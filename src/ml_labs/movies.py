import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.decomposition import TruncatedSVD

from ml_labs.common import read_json, record_context, setup_tracking, verify_dataset, write_json
from ml_labs.data import validate_ratings


class Recommender(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.metadata = read_json(context.artifacts["metadata"])
        arrays = np.load(context.artifacts["factors"])
        self.scores = arrays["scores"]
        self.seen = arrays["seen"]
        self.popularity = arrays["popularity"]
        self.users = {name: i for i, name in enumerate(self.metadata["users"])}

    def predict(self, context, model_input, params=None):
        results = []
        for value in model_input["user_id"]:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("user_id must be a nonempty string.")
            user = self.users.get(value)
            scores = self.popularity.copy() if user is None else self.scores[user].copy()
            if user is not None:
                scores[self.seen[user]] = -np.inf
            indices = np.argsort(-scores, kind="stable")[: self.metadata["top_k"]]
            recommendations = [
                {"item_id": self.metadata["items"][i], "score": float(scores[i])}
                for i in indices if np.isfinite(scores[i])
            ]
            results.append({
                "user_id": value,
                "status": "cold_start_popularity" if user is None else "personalized",
                "recommendations": json.dumps(recommendations),
            })
        return pd.DataFrame(results)


def train_movies(data, output, rank=4, seed=42, work="work"):
    data, output = Path(data), Path(output)
    manifest = verify_dataset(data)
    output.mkdir(parents=True, exist_ok=True)
    ratings = pd.read_csv(data / "ratings.csv")
    validate_ratings(ratings)
    train = ratings[ratings["split"] == "train"]
    users = sorted(train["user_id"].unique())
    items = sorted(pd.read_csv(data / "items.csv")["item_id"])
    user_index, item_index = {u: i for i, u in enumerate(users)}, {v: i for i, v in enumerate(items)}
    matrix = np.zeros((len(users), len(items)))
    seen = np.zeros_like(matrix, dtype=bool)
    for row in train.itertuples():
        u, i = user_index[row.user_id], item_index[row.item_id]
        matrix[u, i], seen[u, i] = row.rating, True
    counts = seen.sum(axis=0)
    popularity = np.divide(matrix.sum(axis=0), counts, out=np.full(len(items), 3.0), where=counts > 0)
    means = np.divide(matrix.sum(axis=1), seen.sum(axis=1))
    centered = np.where(seen, matrix - means[:, None], 0)
    effective_rank = min(rank, len(items) - 1, len(users) - 1)
    if effective_rank < 1:
        raise ValueError("Need at least two users and items.")
    svd = TruncatedSVD(n_components=effective_rank, random_state=seed)
    scores = svd.fit_transform(centered) @ svd.components_ + means[:, None]
    np.savez_compressed(output / "factors.npz", scores=scores, seen=seen, popularity=popularity)
    write_json(output / "metadata.json", {"users": users, "items": items, "top_k": 5})
    validation = ratings[ratings["split"] == "validation"]
    error = []
    baseline_error = []
    for row in validation.itertuples():
        error.append((scores[user_index[row.user_id], item_index[row.item_id]] - row.rating) ** 2)
        baseline_error.append((popularity[item_index[row.item_id]] - row.rating) ** 2)
    metrics = {
        "validation_rmse": float(np.sqrt(np.mean(error))),
        "baseline_rmse": float(np.sqrt(np.mean(baseline_error))),
    }
    setup_tracking(work)
    with mlflow.start_run() as run:
        record_context(manifest, {"rank": effective_rank, "seed": seed})
        mlflow.log_metrics(metrics)
        example = pd.DataFrame({"user_id": [users[0]]})
        model = Recommender()
        model.load_context(type("Context", (), {"artifacts": {
            "metadata": str(output / "metadata.json"), "factors": str(output / "factors.npz")
        }})())
        mlflow.pyfunc.log_model(
            "model", python_model=model,
            artifacts={"metadata": str(output / "metadata.json"), "factors": str(output / "factors.npz")},
            code_paths=[str(Path(__file__).parent)],
            signature=infer_signature(example, model.predict(None, example)),
            input_example=example,
            pip_requirements=[
                "mlflow==2.16.2", "numpy==1.26.4", "pandas==2.2.3",
                "scikit-learn==1.5.2", "pillow==11.0.0",
            ],
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        write_json(output / "candidate.json", {
            "task": "movies", "run_id": run.info.run_id, "model_uri": model_uri,
            "tracking_uri": mlflow.get_tracking_uri(),
            "dataset": manifest, "metrics": metrics,
            "data_path": str(data.resolve()),
        })
    return output / "candidate.json"


def evaluate_movies(candidate, data, output, max_rmse=1.5):
    candidate = read_json(candidate)
    verify_dataset(data, candidate["dataset"])
    frame = pd.read_csv(Path(data) / "ratings.csv")
    validate_ratings(frame)
    mlflow.set_tracking_uri(candidate["tracking_uri"])
    model = mlflow.pyfunc.load_model(candidate["model_uri"])
    sample_users = frame["user_id"].unique().tolist()
    predictions = model.predict(pd.DataFrame({"user_id": sample_users}))
    for prediction in predictions.itertuples():
        train_items = set(frame.loc[
            (frame.user_id == prediction.user_id) & (frame.split == "train"), "item_id"
        ])
        recommended = {r["item_id"] for r in json.loads(prediction.recommendations)}
        if train_items & recommended:
            raise ValueError("Training items leaked into recommendations.")
    passed = candidate["metrics"]["validation_rmse"] <= max_rmse
    report = {
        "task": "movies", "passed": passed, "max_validation_rmse": max_rmse,
        "candidate": candidate,
        "checks": {"excludes_seen_items": True, "evaluated_users": len(predictions)},
        "test_policy": "Test split remains untouched unless validation passes.",
    }
    if not passed:
        write_json(output, report)
        with mlflow.start_run(run_id=candidate["run_id"]):
            mlflow.log_dict(report, "evaluation/validation.json")
            mlflow.set_tag("validation_passed", "false")
        raise ValueError(f"Model gate rejected validation RMSE > {max_rmse}; report: {output}")

    test = frame[frame["split"] == "test"]
    test_users = sorted(test["user_id"].unique().tolist())
    test_predictions = model.predict(pd.DataFrame({"user_id": test_users}))
    # The packaged model emits top-k recommendations, so the holdout metric is Recall@5.
    hit_count = 0
    for row in test_predictions.itertuples():
        recommendations = {
            item["item_id"] for item in json.loads(row.recommendations)
        }
        relevant = set(test.loc[
            (test["user_id"] == row.user_id) & (test.rating >= 4), "item_id"
        ])
        hit_count += len(recommendations & relevant)
    relevant_count = int((test.rating >= 4).sum())
    test_recall = hit_count / relevant_count if relevant_count else None
    report["checks"].update({
        "test_recall_at_5": test_recall,
        "test_relevant_rows": relevant_count,
        "relevance_threshold": 4,
        "test_rows": len(test),
    })
    report["test_policy"] = "Test split was scored only after the validation gate."
    write_json(output, report)
    with mlflow.start_run(run_id=candidate["run_id"]):
        if test_recall is not None:
            mlflow.log_metric("test_recall_at_5", test_recall)
        mlflow.log_dict(report, "evaluation/validation.json")
        mlflow.set_tag("validation_passed", str(passed).lower())
    return report
