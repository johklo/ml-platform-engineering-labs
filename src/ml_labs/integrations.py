"""Native optional Feast, Kubeflow Pipelines, and BentoML adapters."""

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ml_labs.common import read_json


def feast_init(repo: str | Path) -> Path:
    repo = Path(repo)
    existing = [repo / name for name in ("feature_store.yaml", "features.py") if (repo / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing Feast definitions: {existing}")
    (repo / "data").mkdir(parents=True, exist_ok=True)
    (repo / "feature_store.yaml").write_text(
        """project: ml_labs
provider: local
registry: data/registry.db
online_store:
  type: sqlite
  path: data/online_store.db
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
    name="synthetic_user_features",
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
    tags={"dataset": "synthetic"},
)
''',
        encoding="utf-8",
    )
    return repo


def feast_run(repo: str | Path, action: str) -> dict:
    try:
        from feast import FeatureStore
    except ImportError as exc:
        raise RuntimeError("Install Feast using `pip install -e .[feast]`.") from exc
    repo = Path(repo).resolve()
    data_file = repo / "data" / "user_features.parquet"
    if not data_file.exists():
        ratings_path = Path("data/movies/ratings.csv")
        if ratings_path.exists():
            ratings = pd.read_csv(ratings_path)
            rows = ratings.groupby("user_id").rating.agg(["mean", "count"]).reset_index()
            rows = rows.rename(columns={"mean": "average_rating", "count": "ratings_count"})
        else:
            rows = pd.DataFrame({
                "user_id": ["feast-user-1", "feast-user-2"],
                "average_rating": [3.5, 4.0],
                "ratings_count": [8, 11],
            })
        now = datetime.now(timezone.utc)
        rows["event_timestamp"] = now
        rows["created_timestamp"] = now
        rows.to_parquet(data_file, index=False)
    store = FeatureStore(repo_path=str(repo))
    if action == "apply":
        feature_path = repo / "features.py"
        spec = importlib.util.spec_from_file_location("feast_lab_features", feature_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load Feast feature definitions from {feature_path}.")
        feature_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_module)
        store.apply([feature_module.user, feature_module.user_features])
        return {"action": action, "repo": str(repo), "status": "applied"}
    if action == "materialize":
        store.materialize_incremental(end_date=datetime.now(timezone.utc))
        return {"action": action, "repo": str(repo), "status": "materialized"}
    if action == "online":
        response = store.get_online_features(
            features=["user_movie_stats:average_rating", "user_movie_stats:ratings_count"],
            entity_rows=[{"user_id": "feast-user-1"}],
        ).to_dict()
        return {"action": action, "features": response}
    if action == "historical":
        entity_rows = pd.DataFrame([{
            "user_id": "feast-user-1",
            "event_timestamp": datetime.now(timezone.utc) + timedelta(seconds=1),
        }])
        response = store.get_historical_features(
            entity_df=entity_rows,
            features=["user_movie_stats:average_rating", "user_movie_stats:ratings_count"],
        ).to_df()
        return {"action": action, "rows": response.to_dict(orient="records")}
    raise ValueError(f"Unsupported Feast action: {action}")


def compile_kfp(output: str | Path) -> Path:
    try:
        from kfp import compiler, dsl
        from kfp.dsl import Artifact, Dataset, Input, Model, Output
    except ImportError as exc:
        raise RuntimeError("Install Kubeflow Pipelines using `pip install -e .[pipeline]`.") from exc

    @dsl.component(base_image="python:3.11")
    def prepare(output_data: Output[Dataset]):
        import csv
        import os

        os.makedirs(os.path.dirname(output_data.path), exist_ok=True)
        with open(output_data.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x", "y"])
            for value in range(20):
                writer.writerow([value, 2 * value + 1])

    @dsl.component(
        base_image="python:3.11",
        packages_to_install=["scikit-learn==1.5.2", "joblib==1.4.2"],
    )
    def train(input_data: Input[Dataset], output_model: Output[Model]):
        import csv
        import os

        import joblib
        from sklearn.linear_model import LinearRegression

        with open(input_data.path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        features = [[float(row["x"])] for row in rows]
        labels = [float(row["y"]) for row in rows]
        model = LinearRegression().fit(features, labels)
        os.makedirs(os.path.dirname(output_model.path), exist_ok=True)
        joblib.dump(model, output_model.path)

    @dsl.component(
        base_image="python:3.11",
        packages_to_install=["scikit-learn==1.5.2", "joblib==1.4.2"],
    )
    def evaluate(input_data: Input[Dataset], input_model: Input[Model], report: Output[Artifact]):
        import csv
        import json
        import os

        import joblib
        from sklearn.metrics import mean_squared_error

        with open(input_data.path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        model = joblib.load(input_model.path)
        features = [[float(row["x"])] for row in rows]
        labels = [float(row["y"]) for row in rows]
        rmse = mean_squared_error(labels, model.predict(features)) ** 0.5
        os.makedirs(os.path.dirname(report.path), exist_ok=True)
        with open(report.path, "w", encoding="utf-8") as handle:
            json.dump({"rmse": rmse, "samples": len(rows)}, handle)

    @dsl.pipeline(name="ml-platform-labs-native-kfp")
    def lab_pipeline():
        data = prepare().outputs["output_data"]
        model = train(input_data=data).outputs["output_model"]
        evaluate(input_data=data, input_model=model)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(pipeline_func=lab_pipeline, package_path=str(output))
    return output


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
models:
  - "{model_tag}"
''',
        encoding="utf-8",
    )
    return output / "bentofile.yaml"
