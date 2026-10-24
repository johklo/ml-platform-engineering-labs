"""Native optional Feast, Kubeflow Pipelines, and BentoML adapters."""

import hashlib
import importlib.util
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from ml_labs.common import read_json, verify_dataset, write_json
from ml_labs.data import validate_ratings

FEAST_FEATURES = ["user_movie_stats:average_rating", "user_movie_stats:ratings_count"]
FEAST_TABLE = Path("data") / "user_features.parquet"
FEAST_LINEAGE = Path("data") / "lineage.json"
KFP_IMAGE_PLACEHOLDER = "SET-YOUR-REGISTRY/ml-platform-engineering-labs:SET-IMAGE-TAG"
KFP_MLFLOW_ROOT = "/workspace/mlflow"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp(value: str | datetime) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"Timestamp must include a timezone offset: {value}")
    return stamp.tz_convert("UTC")


def build_user_feature_snapshots(ratings: pd.DataFrame) -> pd.DataFrame:
    """Cumulative per-user statistics from training ratings only.

    Each row is the user's state right after one training rating was observed, so
    ``event_timestamp`` is when that value became knowable. Validation/test rows never
    contribute, which keeps the holdout splits out of the feature store.
    """
    validate_ratings(ratings)
    train = ratings[ratings["split"] == "train"].copy()
    if train.empty:
        raise ValueError("The prepared movie dataset has no training rows.")
    train["event_timestamp"] = pd.to_datetime(train["timestamp"], utc=True)
    train = train.sort_values(["user_id", "event_timestamp", "item_id"], kind="stable")
    ratings_by_user = train.groupby("user_id", sort=False)["rating"]
    train["ratings_count"] = ratings_by_user.cumcount().astype("int64") + 1
    train["average_rating"] = (ratings_by_user.cumsum() / train["ratings_count"]).astype("float32")
    snapshots = train.drop_duplicates(["user_id", "event_timestamp"], keep="last")
    snapshots = snapshots[["user_id", "event_timestamp", "average_rating", "ratings_count"]].copy()
    snapshots["created_timestamp"] = snapshots["event_timestamp"]
    return snapshots.reset_index(drop=True)


def feast_init(repo: str | Path, data: str | Path) -> Path:
    """Create a Feast repo whose offline source is derived from a prepared movie dataset."""
    repo, data = Path(repo), Path(data)
    ratings_path = data / "ratings.csv"
    if not ratings_path.is_file():
        raise FileNotFoundError(
            f"Prepared movie ratings not found: {ratings_path}. "
            "Run `ml-labs movies prepare --output <dir>` first and pass it with --data."
        )
    manifest = verify_dataset(data)
    if not manifest["dataset"].endswith("movies"):
        raise ValueError(f"Feast lab expects a movie dataset, got {manifest['dataset']!r}.")
    existing = [
        repo / name for name in ("feature_store.yaml", "features.py", str(FEAST_TABLE))
        if (repo / name).exists()
    ]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing Feast definitions: {existing}")
    snapshots = build_user_feature_snapshots(pd.read_csv(ratings_path))
    (repo / "data").mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(repo / FEAST_TABLE, index=False)
    users = sorted(snapshots["user_id"].unique().tolist())
    write_json(repo / FEAST_LINEAGE, {
        "dataset": manifest,
        "data_path": str(data.resolve()),
        "feature_table": FEAST_TABLE.as_posix(),
        "feature_table_sha256": _sha256(repo / FEAST_TABLE),
        "policy": "train split only; one cumulative snapshot per user training event; "
                  "event_timestamp is the time of the rating that produced the snapshot",
        "snapshot_rows": len(snapshots),
        "users": users,
        "first_event": snapshots["event_timestamp"].min().isoformat(),
        "last_event": snapshots["event_timestamp"].max().isoformat(),
    })
    (repo / "feature_store.yaml").write_text(
        """project: ml_labs
provider: local
registry: data/registry.db
online_store:
  type: sqlite
  path: data/online_store.db
offline_store:
  type: file
entity_key_serialization_version: 3
""",
        encoding="utf-8",
    )
    (repo / "features.py").write_text(
        '''from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

user = Entity(name="user_id", join_keys=["user_id"], description="Synthetic movie user")
source = FileSource(
    name="movie_user_training_snapshots",
    path=str(Path(__file__).resolve().parent / "data" / "user_features.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)
user_features = FeatureView(
    name="user_movie_stats",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="average_rating", dtype=Float32),
        Field(name="ratings_count", dtype=Int64),
    ],
    source=source,
    online=True,
    tags={"dataset": "synthetic-movies", "split": "train-only"},
)
''',
        encoding="utf-8",
    )
    return repo


def _json_value(value):
    if hasattr(value, "item"):
        value = value.item()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _expected_snapshot(snapshots: pd.DataFrame, user: str, at: pd.Timestamp) -> dict:
    rows = snapshots[(snapshots["user_id"] == user) & (snapshots["event_timestamp"] <= at)]
    if rows.empty:
        return {"average_rating": None, "ratings_count": None}
    last = rows.iloc[-1]
    return {
        "average_rating": float(last["average_rating"]),
        "ratings_count": int(last["ratings_count"]),
    }


def feast_run(
    repo: str | Path, action: str, user_id: str | None = None, end_date: str | None = None
) -> dict:
    try:
        from feast import FeatureStore
    except ImportError as exc:
        raise RuntimeError("Install Feast using `uv sync --extra dev --extra feast`.") from exc
    repo = Path(repo).resolve()
    lineage_path, table = repo / FEAST_LINEAGE, repo / FEAST_TABLE
    if not lineage_path.is_file() or not table.is_file():
        raise FileNotFoundError(
            f"{repo} is not an initialized movie Feast repo; run "
            "`ml-labs tools feast init --data <prepared movies> --output <repo>` first."
        )
    lineage = read_json(lineage_path)
    if _sha256(table) != lineage["feature_table_sha256"]:
        raise ValueError(f"Feature table changed after init: {table}")
    user = user_id or lineage["users"][0]
    if user not in lineage["users"]:
        raise ValueError(f"user_id {user!r} has no training snapshots in this Feast repo.")
    store = FeatureStore(repo_path=str(repo))
    base = {"action": action, "repo": str(repo), "dataset_sha256": lineage["dataset"]["sha256"]}
    if action == "apply":
        feature_path = repo / "features.py"
        spec = importlib.util.spec_from_file_location("feast_lab_features", feature_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load Feast feature definitions from {feature_path}.")
        feature_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_module)
        store.apply([feature_module.user, feature_module.user_features])
        return {**base, "status": "applied"}
    if action == "materialize":
        start = _timestamp(lineage["first_event"]) - timedelta(seconds=1)
        end = (_timestamp(end_date) if end_date
               else _timestamp(lineage["last_event"]) + timedelta(seconds=1))
        if end <= start:
            raise ValueError(f"--end-date must be after the first event {lineage['first_event']}.")
        store.materialize(start_date=start.to_pydatetime(), end_date=end.to_pydatetime())
        return {**base, "status": "materialized", "start": start.isoformat(),
                "end": end.isoformat()}
    snapshots = pd.read_parquet(table)
    snapshots["event_timestamp"] = pd.to_datetime(snapshots["event_timestamp"], utc=True)
    if action == "online":
        response = store.get_online_features(
            features=FEAST_FEATURES, entity_rows=[{"user_id": user}]
        ).to_dict()
        values = {name: [_json_value(v) for v in column] for name, column in response.items()}
        if values["average_rating"][0] is None or values["ratings_count"][0] is None:
            raise RuntimeError(
                f"No online features for {user!r}. Run `ml-labs tools feast run --action apply` and "
                "`--action materialize` first (the online store is filled only by materialization)."
            )
        return {**base, "user_id": user, "features": values}
    if action == "historical":
        times = snapshots.loc[snapshots["user_id"] == user, "event_timestamp"].sort_values().tolist()
        query_times = [
            times[0] - timedelta(days=1),
            times[0],
            times[len(times) // 2] + timedelta(hours=12),
            times[-1] + timedelta(days=30),
        ]
        entity_df = pd.DataFrame({"user_id": [user] * len(query_times),
                                  "event_timestamp": query_times})
        result = store.get_historical_features(entity_df=entity_df, features=FEAST_FEATURES).to_df()
        result["event_timestamp"] = pd.to_datetime(result["event_timestamp"], utc=True)
        result["returned_by_feast"] = True
        # Feast's file offline store omits entity rows that have no feature row at or before
        # their timestamp; keep them so the "before any history" case stays visible as null.
        joined = entity_df.merge(result, on=["user_id", "event_timestamp"], how="left")
        joined["returned_by_feast"] = joined["returned_by_feast"].eq(True)
        rows = []
        for record in joined.sort_values("event_timestamp").to_dict(orient="records"):
            at = record["event_timestamp"]
            actual = {
                "average_rating": _json_value(record["average_rating"]),
                "ratings_count": _json_value(record["ratings_count"]),
            }
            if actual["ratings_count"] is not None:
                actual["ratings_count"] = int(actual["ratings_count"])
            expected = _expected_snapshot(snapshots, user, at)
            if expected["average_rating"] is None:
                matches = actual["average_rating"] is None and actual["ratings_count"] is None
            else:
                matches = (
                    actual["ratings_count"] == expected["ratings_count"]
                    and actual["average_rating"] is not None
                    and abs(actual["average_rating"] - expected["average_rating"]) < 1e-4
                )
            rows.append({"user_id": user, "event_timestamp": at.isoformat(), **actual,
                         "returned_by_feast": bool(record["returned_by_feast"]),
                         "expected": expected, "point_in_time_correct": matches})
        if not all(row["point_in_time_correct"] for row in rows):
            raise RuntimeError(f"Feast historical retrieval disagreed with snapshots: {rows}")
        return {**base, "user_id": user, "rows": rows}
    raise ValueError(f"Unsupported Feast action: {action}")


def _validate_image(image: str) -> str:
    if not image or any(char.isspace() for char in image):
        raise ValueError(f"Invalid container image reference: {image!r}")
    return image


def build_kfp_movie_pipeline(image: str):
    """Return the KFP movie pipeline and its components, all running the packaged repo image."""
    try:
        from kfp import dsl
        from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output
    except ImportError as exc:
        raise RuntimeError(
            "Install Kubeflow Pipelines using `uv sync --extra dev --extra pipeline`."
        ) from exc
    image = _validate_image(image)

    @dsl.component(base_image=image, install_kfp_package=False)
    def prepare_movie_data(
        source_revision: str, seed: int, users: int, items: int, dataset: Output[Dataset]
    ):
        import json
        import os

        from ml_labs.data import prepare_movies

        if not source_revision.strip():
            raise ValueError("source_revision is required for provenance.")
        prepare_movies(dataset.path, seed=seed, users=users, items=items)
        with open(os.path.join(dataset.path, "manifest.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)
        dataset.metadata.update({
            "dataset": manifest["dataset"], "sha256": manifest["sha256"],
            "rows": manifest["rows"], "source_revision": source_revision,
        })

    @dsl.component(base_image=image, install_kfp_package=False)
    def train_movie_recommender(
        source_revision: str, dataset: Input[Dataset], rank: int, seed: int, mlflow_root: str,
        model: Output[Model],
    ):
        import json
        import os
        import shutil
        from pathlib import Path

        from ml_labs.movies import train_movies

        if not source_revision.strip():
            raise ValueError("source_revision is required for provenance.")
        root = Path(mlflow_root)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"MLflow scratch root must be empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        os.environ["ML_LABS_CODE_REVISION"] = source_revision
        os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///" + (root / "tracking.db").as_posix()
        previous = os.getcwd()
        os.chdir(root)  # MLflow's default artifact root (./mlruns) lands inside the bundle.
        try:
            candidate_path = train_movies(dataset.path, root / "candidate", rank=rank, seed=seed,
                                          work=root)
        finally:
            os.chdir(previous)
        candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
        shutil.copytree(root, model.path, dirs_exist_ok=True)
        model.metadata.update({
            "run_id": candidate["run_id"], "model_uri": candidate["model_uri"],
            "dataset_sha256": candidate["dataset"]["sha256"], "source_revision": source_revision,
            "mlflow_root": str(root), **candidate["metrics"],
        })

    @dsl.component(base_image=image, install_kfp_package=False)
    def evaluate_movie_recommender(
        source_revision: str, dataset: Input[Dataset], model: Input[Model], max_rmse: float,
        mlflow_root: str, report: Output[Artifact], metrics: Output[Metrics],
        evaluated_model: Output[Model],
    ):
        import os
        import shutil
        from pathlib import Path

        from ml_labs.movies import evaluate_movies

        trained_revision = model.metadata.get("source_revision")
        if trained_revision != source_revision:
            raise ValueError(f"Model revision {trained_revision!r} != {source_revision!r}.")
        root = Path(mlflow_root)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"MLflow scratch root must be empty: {root}")
        # The MLflow store records absolute artifact paths, so restore it at the training root.
        shutil.copytree(model.path, root, dirs_exist_ok=True)
        os.environ["ML_LABS_CODE_REVISION"] = source_revision
        os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///" + (root / "tracking.db").as_posix()
        result = evaluate_movies(root / "candidate" / "candidate.json", dataset.path, report.path,
                                 max_rmse=max_rmse)
        candidate_metrics = result["candidate"]["metrics"]
        metrics.log_metric("validation_rmse", candidate_metrics["validation_rmse"])
        metrics.log_metric("baseline_rmse", candidate_metrics["baseline_rmse"])
        metrics.log_metric("test_relevant_rows", result["checks"]["test_relevant_rows"])
        if result["checks"]["test_recall_at_5"] is not None:
            metrics.log_metric("test_recall_at_5", result["checks"]["test_recall_at_5"])
        report.metadata.update({"passed": result["passed"], "source_revision": source_revision})
        shutil.copytree(root, evaluated_model.path, dirs_exist_ok=True)
        evaluated_model.metadata.update({**model.metadata, "validation_passed": result["passed"]})

    @dsl.pipeline(
        name="ml-labs-movie-recommender",
        description="Synthetic movie data -> SVD recommender (MLflow) -> gated holdout evaluation",
    )
    def movie_pipeline(
        source_revision: str,
        seed: int = 42,
        users: int = 48,
        items: int = 32,
        rank: int = 4,
        max_rmse: float = 1.5,
        mlflow_root: str = KFP_MLFLOW_ROOT,
    ):
        prepared = prepare_movie_data(
            source_revision=source_revision, seed=seed, users=users, items=items
        ).set_display_name("prepare-movie-data")
        trained = train_movie_recommender(
            source_revision=source_revision, dataset=prepared.outputs["dataset"], rank=rank,
            seed=seed, mlflow_root=mlflow_root,
        ).set_display_name("train-svd-recommender")
        evaluate_movie_recommender(
            source_revision=source_revision, dataset=prepared.outputs["dataset"],
            model=trained.outputs["model"], max_rmse=max_rmse, mlflow_root=mlflow_root,
        ).set_display_name("evaluate-holdout")

    components = {
        "prepare": prepare_movie_data,
        "train": train_movie_recommender,
        "evaluate": evaluate_movie_recommender,
    }
    return movie_pipeline, components


def compile_kfp(
    output: str | Path, source_revision: str, image: str = KFP_IMAGE_PLACEHOLDER
) -> dict:
    """Compile (never submit) the movie pipeline; the image must contain this repo's code."""
    if not source_revision or not source_revision.strip():
        raise ValueError("--source-revision is required (for example `git rev-parse HEAD`).")
    pipeline, _components = build_kfp_movie_pipeline(image)
    from kfp import compiler

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=str(output),
        pipeline_parameters={"source_revision": source_revision},
    )
    placeholder = image == KFP_IMAGE_PLACEHOLDER
    result = {
        "output": str(output), "image": image, "source_revision": source_revision,
        "image_is_placeholder": placeholder, "submitted": False,
    }
    if placeholder:
        result["warning"] = (
            "Image is a placeholder; build/push the repo Dockerfile and recompile with "
            "--image <registry>/<name>@sha256:<digest> before submitting."
        )
    return result

def build_bento(candidate_path: str | Path, output: str | Path) -> Path:
    """Import the MLflow run into BentoML and emit a real Bento service/config."""
    try:
        from bentoml.mlflow import import_model
    except ImportError as exc:
        raise RuntimeError("Install BentoML using `pip install -e .[bento]`.") from exc
    candidate = read_json(candidate_path)
    task = candidate.get("task")
    if task not in {"movies", "rag"}:
        raise ValueError("The Bento adapter supports the movie and RAG tabular models.")
    import mlflow

    mlflow.set_tracking_uri(candidate["tracking_uri"])
    imported_model = import_model(
        name=f"ml-labs-{task}",
        model_uri=candidate.get("registered_uri", candidate["model_uri"]),
    )
    model_tag = str(imported_model.tag)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "service.py").write_text(
        f'''import json

import bentoml
import pandas as pd

MODEL_TAG = {str(model_tag)!r}

@bentoml.service(name="ml-labs-{task}")
class PlatformLabService:
    def __init__(self):
        self.model = bentoml.mlflow.load_model(MODEL_TAG)

    @bentoml.api
    def predict(self, request: dict) -> dict:
        if "{task}" == "movies":
            frame = pd.DataFrame({{"user_id": [request["user_id"]]}})
        else:
            frame = pd.DataFrame({{"query": [request["query"]]}})
        row = self.model.predict(frame).iloc[0]
        result = row.to_dict()
        for key, value in list(result.items()):
            if isinstance(value, str) and value.startswith(("[", "{{")):
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
        return result

''',
        encoding="utf-8",
    )
    (output / "bentofile.yaml").write_text(
        f'''service: "service:PlatformLabService"
labels:
  project: "ml-platform-engineering-labs"
  task: "{task}"
include:
  - "service.py"
python:
  packages:
    - bentoml==1.3.10
    - mlflow==2.16.2
    - pandas==2.2.3
    # The MLflow pyfunc unpickles ml_labs code, which imports these at module load.
    - numpy==1.26.4
    - scikit-learn==1.5.2
    - pillow==11.0.0
models:
  - "{model_tag}"
''',
        encoding="utf-8",
    )
    return output / "bentofile.yaml"
