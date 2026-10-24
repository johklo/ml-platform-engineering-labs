
import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ml_labs.api import create_app
from ml_labs.batch import batch_recommend
from ml_labs.common import read_json
from ml_labs.data import prepare_movies
from ml_labs.features import FeaturePolicy
from ml_labs.movies import evaluate_movies, train_movies
from ml_labs.registry import register_candidate
from ml_labs.releases import activate_model, read_release, rollback_model


@pytest.fixture
def movie_system(tmp_path):
    data = prepare_movies(tmp_path / "data")
    candidate = train_movies(data, tmp_path / "candidate", work=tmp_path / "tracking")
    report = tmp_path / "report.json"
    evaluate_movies(candidate, data, report)
    register_candidate(candidate, "integrated-movies", report)
    return data, candidate, report


def test_ranking_compares_baseline_before_test_gate(movie_system, tmp_path):
    data, candidate, _ = movie_system
    failed = tmp_path / "failed.json"
    with pytest.raises(ValueError, match="validation gate"):
        evaluate_movies(candidate, data, failed, min_ndcg_lift=1.01)
    result = read_json(failed)
    assert result["ranking_gate"]["passed"] is False
    assert result["validation_ranking"]["ranking_users"] > 0
    assert "test_ranking" not in result
    assert "baseline_ndcg_at_5" in result["validation_ranking"]


def test_atomic_release_failure_and_rollback_preserve_served_version(movie_system, tmp_path):
    _, candidate, report = movie_system
    store = tmp_path / "releases"
    initial = activate_model(store, candidate, report)
    # Same model registered as another immutable version provides a distinct release.
    register_candidate(candidate, "integrated-movies", report)
    with pytest.raises(RuntimeError, match="Injected failure"):
        activate_model(store, candidate, report, fail_before_switch=True)
    assert read_release(store)["release_id"] == initial["release_id"]
    second = activate_model(store, candidate, report)
    assert second["release_id"] != initial["release_id"]
    assert activate_model(store, candidate, report)["changed"] is False
    assert rollback_model(store)["release_id"] == initial["release_id"]
    api = TestClient(create_app(movie_release=store))
    response = api.post("/recommend", json={"user_id": "u000"}).json()
    assert response["model_uri"].endswith("/1")
    assert response["release_id"] == initial["release_id"]


def test_batch_retry_is_idempotent_and_rejects_key_reuse(movie_system, tmp_path):
    _, candidate, report = movie_system
    users = tmp_path / "users.csv"
    pd.DataFrame({"user_id": ["u000", "new-user"]}).to_csv(users, index=False)
    output = tmp_path / "batch"
    with pytest.raises(RuntimeError, match="Injected failure"):
        batch_recommend(candidate, report, users, output, "2025-01-15", True)
    assert not list(output.glob("*/manifest.json"))
    committed = batch_recommend(candidate, report, users, output, "2025-01-15")
    assert committed["rows"] == 2 and committed["reused"] is False
    repeated = batch_recommend(candidate, report, users, output, "2025-01-15")
    assert repeated["reused"] is True
    assert len(list(output.glob("*/manifest.json"))) == 1
    pd.DataFrame({"user_id": ["different"]}).to_csv(users, index=False)
    with pytest.raises(ValueError, match="different inputs"):
        batch_recommend(candidate, report, users, output, "2025-01-15")


def test_native_feast_changes_real_http_routing_and_exposes_degradation(movie_system, tmp_path):
    pytest.importorskip("feast")
    from ml_labs.integrations import feast_init, feast_run

    data, candidate, _ = movie_system
    repo = feast_init(tmp_path / "feast", data)
    feast_run(repo, "apply")
    policy = FeaturePolicy(as_of="2025-01-15T00:00:00Z")
    api = TestClient(create_app(movies=candidate, feast_repo=repo, feature_policy=policy))
    assert api.post("/recommend", json={"user_id": "u007"}).status_code == 503
    feast_run(repo, "materialize", end_date="2025-01-05T01:00:00Z")
    early = api.post("/recommend", json={"user_id": "u007"}).json()
    assert early["status"] == "feature_routed_popularity"
    assert early["feature_routing"]["features"]["ratings_count"] == 5
    feast_run(repo, "materialize")
    mature = api.post("/recommend", json={"user_id": "u007"}).json()
    assert mature["status"] == "personalized"
    assert mature["feature_routing"]["features"]["ratings_count"] == 14
    assert mature["recommendations"] != early["recommendations"]
    stale = TestClient(create_app(
        movies=candidate, feast_repo=repo,
        feature_policy=FeaturePolicy(as_of="2026-01-01T00:00:00Z"),
    ))
    response = stale.post("/recommend", json={"user_id": "u007"})
    assert response.status_code == 503 and "stale_features" in response.text
    future = TestClient(create_app(
        movies=candidate, feast_repo=repo,
        feature_policy=FeaturePolicy(as_of="2025-01-01T00:00:00Z"),
    ))
    response = future.post("/recommend", json={"user_id": "u007"})
    assert response.status_code == 503 and "future_features" in response.text
    degraded = TestClient(create_app(
        movies=candidate, feast_repo=repo,
        feature_policy=FeaturePolicy(unavailable="popularity", as_of="2025-01-15T00:00:00Z"),
    ))
    response = degraded.post("/recommend", json={"user_id": "unknown"}).json()
    assert response["feature_routing"]["reason"] == "missing_features"
    assert response["feature_routing"]["degraded"] is True
    metadata = read_json(candidate)
    metadata["dataset"]["sha256"] = "wrong-data"
    from ml_labs.common import write_json

    wrong = write_json(tmp_path / "wrong.json", metadata)
    with pytest.raises(ValueError, match="fingerprints differ"):
        create_app(movies=wrong, feast_repo=repo)
    mlflow.end_run()
