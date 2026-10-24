"""An executable failure-and-recovery workshop, using real local platform tools."""

from pathlib import Path

import pandas as pd

from ml_labs.api import create_app
from ml_labs.batch import batch_recommend
from ml_labs.common import read_json, write_json
from ml_labs.data import prepare_movies
from ml_labs.features import FeaturePolicy
from ml_labs.integrations import feast_init, feast_run
from ml_labs.monitoring import create_drift_report
from ml_labs.movies import evaluate_movies, train_movies
from ml_labs.registry import register_candidate
from ml_labs.releases import activate_model, read_release, rollback_model


def run_exercise(work, data=None) -> dict:
    from fastapi.testclient import TestClient

    work = Path(work).resolve()
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"Use a new workshop directory: {work}")
    work.mkdir(parents=True)
    data = Path(data).resolve() if data else prepare_movies(work / "data")
    candidates = [
        train_movies(data, work / f"rank-{rank}", rank=rank, work=work / "tracking")
        for rank in (2, 4)
    ]
    candidates.sort(key=lambda path: (
        -read_json(path)["metrics"].get("validation_model_ndcg_at_5", -1),
        read_json(path)["metrics"]["validation_rmse"],
    ))
    candidate = candidates[0]
    report = work / "evaluation.json"
    evaluation = evaluate_movies(candidate, data, report)
    register_candidate(candidate, "workshop-movies", report)
    releases = work / "release-store"
    first_release = activate_model(releases, candidate, report)
    repo = feast_init(work / "feast", data)
    feast_run(repo, "apply")
    frame = pd.read_csv(data / "ratings.csv")
    train = frame[frame.split == "train"]
    clock = (pd.to_datetime(train.timestamp, utc=True).max() + pd.Timedelta(seconds=1)).isoformat()
    user = train.groupby("user_id").size().idxmax()
    client = TestClient(create_app(
        movie_release=releases, feast_repo=repo, feature_policy=FeaturePolicy(as_of=clock),
    ))
    absent = client.post("/recommend", json={"user_id": user})
    if absent.status_code != 503:
        raise RuntimeError("Workshop expected missing Feast values to reject the request.")
    history = feast_run(repo, "historical", user_id=user)
    feast_run(repo, "materialize")
    good = client.post("/recommend", json={"user_id": user})
    if good.status_code != 200:
        raise RuntimeError(f"Materialized features did not restore serving: {good.text}")
    users = work / "users.csv"
    pd.DataFrame({"user_id": [user, "workshop-new-user"]}).to_csv(users, index=False)
    try:
        batch_recommend(candidate, report, users, work / "batch", "workshop-day", True)
    except RuntimeError as exc:
        if not str(exc).startswith("Injected failure"):
            raise
    else:
        raise RuntimeError("Batch failure injection did not fire.")
    committed = batch_recommend(candidate, report, users, work / "batch", "workshop-day")
    repeated = batch_recommend(candidate, report, users, work / "batch", "workshop-day")
    register_candidate(candidate, "workshop-movies", report)
    try:
        activate_model(releases, candidate, report, fail_before_switch=True)
    except RuntimeError as exc:
        if not str(exc).startswith("Injected failure"):
            raise
    else:
        raise RuntimeError("Release failure injection did not fire.")
    if read_release(releases)["release_id"] != first_release["release_id"]:
        raise RuntimeError("Failed publication changed the active release.")
    next_release = activate_model(releases, candidate, report)
    restored = rollback_model(releases)
    if restored["release_id"] != first_release["release_id"]:
        raise RuntimeError("Rollback did not restore the first release.")
    reference = work / "reference.csv"
    current = work / "current.csv"
    train[["rating"]].to_csv(reference, index=False)
    pd.DataFrame({"rating": [1.0] * len(train)}).to_csv(current, index=False)
    drift = create_drift_report(reference, current, work / "drift.json", backend="evidently")
    evidence = {
        "selected_run": read_json(candidate)["run_id"],
        "selection": "validation NDCG@5, then validation RMSE; no test-based selection",
        "evaluation": evaluation,
        "feature_failure_http": absent.status_code,
        "feature_recovery": good.json(),
        "historical": history,
        "batch": {"committed": committed, "retried": repeated},
        "releases": {
            "first": first_release["release_id"], "next": next_release["release_id"],
            "restored": restored["release_id"],
        },
        "drift": drift, "clock": clock,
        "scope": "Single-host replay; not a distributed cluster availability guarantee.",
    }
    write_json(work / "evidence.json", evidence)
    return evidence
