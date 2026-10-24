import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature, infer_signature
from mlflow.types import ColSpec, Schema
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
        for record in model_input.to_dict(orient="records"):
            value = record["user_id"]
            strategy = record.get("strategy", "auto")
            if strategy not in {"auto", "popularity"}:
                raise ValueError("strategy must be auto or popularity.")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("user_id must be a nonempty string.")
            user = self.users.get(value)
            popular = user is None or strategy == "popularity"
            scores = self.popularity.copy() if popular else self.scores[user].copy()
            if user is not None:
                scores[self.seen[user]] = -np.inf
            indices = np.argsort(-scores, kind="stable")[: self.metadata["top_k"]]
            recommendations = [
                {"item_id": self.metadata["items"][i], "score": float(scores[i])}
                for i in indices if np.isfinite(scores[i])
            ]
            results.append({
                "user_id": value,
                "status": "cold_start_popularity" if user is None else (
                    "feature_routed_popularity" if popular else "personalized"
                ),
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
    validation_ranking = ranking_metrics(
        scores, popularity, seen, users, items, validation,
    )
    metrics.update({f"validation_{key}": value for key, value in validation_ranking.items()
                    if isinstance(value, (int, float))})
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
            signature=ModelSignature(
                inputs=Schema([
                    ColSpec("string", "user_id"),
                    ColSpec("string", "strategy", required=False),
                ]),
                outputs=infer_signature(example, model.predict(None, example)).outputs,
            ),
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
            "serving_contract": "movie-routing-v2",
        })
    return output / "candidate.json"


def ranking_metrics(scores, popularity, seen, users, items, holdout, k=5):
    """Macro ranking metrics for relevant (rating >= 4) held-out items, excluding train items."""
    import math

    item_index = {item: index for index, item in enumerate(items)}
    metrics = {"model": [], "baseline": []}
    for user_index, user in enumerate(users):
        relevant = {
            item_index[item] for item in holdout.loc[
                (holdout.user_id == user) & (holdout.rating >= 4), "item_id"
            ] if item in item_index
        }
        if not relevant:
            continue
        for name, values in (("model", scores[user_index]), ("baseline", popularity)):
            values = values.copy()
            values[seen[user_index]] = -np.inf
            top = [int(i) for i in np.argsort(-values, kind="stable")[:k]
                   if np.isfinite(values[i])]
            hits = [int(item in relevant) for item in top]
            dcg = sum(hit / math.log2(position + 2) for position, hit in enumerate(hits))
            ideal = sum(1 / math.log2(position + 2) for position in range(min(k, len(relevant))))
            metrics[name].append((sum(hits) / len(relevant), dcg / ideal))
    result = {"ranking_users": len(metrics["model"])}
    for name, values in metrics.items():
        result[f"{name}_recall_at_5"] = float(np.mean([v[0] for v in values])) if values else None
        result[f"{name}_ndcg_at_5"] = float(np.mean([v[1] for v in values])) if values else None
    return result


def evaluate_movies(candidate, data, output, max_rmse=1.5, min_ndcg_lift=None):
    if not np.isfinite(max_rmse) or max_rmse < 0:
        raise ValueError("max_rmse must be finite and nonnegative.")
    if min_ndcg_lift is not None and not np.isfinite(min_ndcg_lift):
        raise ValueError("min_ndcg_lift must be finite.")
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
    implementation = model.unwrap_python_model()
    validation = frame[frame.split == "validation"]
    user_index = {user: i for i, user in enumerate(implementation.metadata["users"])}
    item_index = {item: i for i, item in enumerate(implementation.metadata["items"])}
    squared_error = [
        (implementation.scores[user_index[row.user_id], item_index[row.item_id]] - row.rating) ** 2
        for row in validation.itertuples()
    ]
    validation_rmse = float(np.sqrt(np.mean(squared_error)))
    ranking = ranking_metrics(
        implementation.scores, implementation.popularity, implementation.seen,
        implementation.metadata["users"], implementation.metadata["items"], validation,
    )
    rmse_passed = validation_rmse <= max_rmse
    lift = (ranking["model_ndcg_at_5"] - ranking["baseline_ndcg_at_5"]
            if ranking["ranking_users"] else None)
    ranking_passed = min_ndcg_lift is None or (lift is not None and lift >= min_ndcg_lift)
    passed = rmse_passed and ranking_passed
    report = {
        "task": "movies", "passed": passed, "max_validation_rmse": max_rmse,
        "candidate": candidate,
        "checks": {"excludes_seen_items": True, "evaluated_users": len(predictions)},
        "test_policy": "Test split remains untouched unless validation passes.",
        "validation_ranking": ranking,
        "observed_validation_rmse": validation_rmse,
        "ranking_gate": {"min_ndcg_lift": min_ndcg_lift, "observed_lift": lift,
                         "passed": ranking_passed},
    }
    if not passed:
        write_json(output, report)
        with mlflow.start_run(run_id=candidate["run_id"]):
            mlflow.log_dict(report, "evaluation/validation.json")
            mlflow.set_tag("validation_passed", "false")
        raise ValueError(f"Model validation gate rejected candidate; report: {output}")

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
    report["test_ranking"] = ranking_metrics(
        implementation.scores, implementation.popularity, implementation.seen,
        implementation.metadata["users"], implementation.metadata["items"], test,
    )
    report["test_policy"] = "Test split was scored only after the validation gate."
    write_json(output, report)
    with mlflow.start_run(run_id=candidate["run_id"]):
        if test_recall is not None:
            mlflow.log_metric("test_recall_at_5", test_recall)
        mlflow.log_dict(report, "evaluation/validation.json")
        mlflow.set_tag("validation_passed", str(passed).lower())
    return report
