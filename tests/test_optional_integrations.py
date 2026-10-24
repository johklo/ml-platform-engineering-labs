from pathlib import Path

import pandas as pd
import pytest
import yaml


def test_feast_snapshots_use_training_rows_only(tmp_path: Path) -> None:
    from ml_labs.data import prepare_movies
    from ml_labs.integrations import build_user_feature_snapshots

    ratings = pd.read_csv(prepare_movies(tmp_path / "movies", users=4, items=20) / "ratings.csv")
    snapshots = build_user_feature_snapshots(ratings)

    train = ratings[ratings["split"] == "train"]
    assert len(snapshots) == len(train)
    final = snapshots.groupby("user_id").last()
    assert final["ratings_count"].to_dict() == train.groupby("user_id").size().to_dict()
    expected_mean = train.groupby("user_id")["rating"].mean()
    assert (final["average_rating"] - expected_mean).abs().max() < 1e-5
    assert snapshots["event_timestamp"].max() == pd.to_datetime(train["timestamp"], utc=True).max()


def test_feast_init_requires_prepared_movie_data(tmp_path: Path) -> None:
    from ml_labs.integrations import feast_init

    with pytest.raises(FileNotFoundError, match="movies prepare"):
        feast_init(tmp_path / "feast", tmp_path / "missing")


def test_native_feast_point_in_time_and_materialization(tmp_path: Path) -> None:
    pytest.importorskip("feast")
    pytest.importorskip("pyarrow")
    from ml_labs.data import prepare_movies
    from ml_labs.integrations import feast_init, feast_run

    data = prepare_movies(tmp_path / "movies", users=6, items=24)
    ratings = pd.read_csv(data / "ratings.csv")
    user = "u003"
    train = ratings[(ratings["user_id"] == user) & (ratings["split"] == "train")]
    repo = feast_init(tmp_path / "feast", data)

    assert feast_run(repo, "apply", user_id=user)["status"] == "applied"
    with pytest.raises(RuntimeError, match="materialize"):
        feast_run(repo, "online", user_id=user)

    history = feast_run(repo, "historical", user_id=user)["rows"]
    counts = [row["ratings_count"] for row in history]
    assert counts == [None, 1, counts[2], len(train)]
    assert 1 < counts[2] < len(train)
    assert all(row["point_in_time_correct"] for row in history)
    assert [row["returned_by_feast"] for row in history] == [False, True, True, True]

    cutoff = pd.to_datetime(train["timestamp"], utc=True).iloc[4] + pd.Timedelta(hours=1)
    feast_run(repo, "materialize", end_date=cutoff.isoformat())
    partial = feast_run(repo, "online", user_id=user)["features"]
    assert partial["ratings_count"] == [5]

    feast_run(repo, "materialize")
    latest = feast_run(repo, "online", user_id=user)["features"]
    assert latest["ratings_count"] == [len(train)]
    assert abs(latest["average_rating"][0] - train["rating"].mean()) < 1e-4
    with pytest.raises(ValueError, match="no training snapshots"):
        feast_run(repo, "online", user_id="not-a-user")


def _kfp_tasks(spec: dict) -> dict:
    return spec["root"]["dag"]["tasks"]


def test_native_kfp_movie_pipeline_compile_contract(tmp_path: Path) -> None:
    pytest.importorskip("kfp")
    from ml_labs.integrations import KFP_IMAGE_PLACEHOLDER, compile_kfp

    image = "example.azurecr.io/ml-platform-engineering-labs@sha256:" + "a" * 64
    result = compile_kfp(tmp_path / "pipeline.yaml", "abc123", image)
    spec = yaml.safe_load(Path(result["output"]).read_text(encoding="utf-8"))

    assert result["submitted"] is False and result["image_is_placeholder"] is False
    tasks = _kfp_tasks(spec)
    assert set(tasks) == {
        "prepare-movie-data", "train-movie-recommender", "evaluate-movie-recommender"
    }
    train_inputs = tasks["train-movie-recommender"]["inputs"]["artifacts"]
    assert train_inputs["dataset"]["taskOutputArtifact"]["producerTask"] == "prepare-movie-data"
    eval_inputs = tasks["evaluate-movie-recommender"]["inputs"]["artifacts"]
    assert eval_inputs["model"]["taskOutputArtifact"]["producerTask"] == "train-movie-recommender"
    assert eval_inputs["dataset"]["taskOutputArtifact"]["outputArtifactKey"] == "dataset"
    parameters = spec["root"]["inputDefinitions"]["parameters"]
    assert parameters["source_revision"]["defaultValue"] == "abc123"
    executors = spec["deploymentSpec"]["executors"].values()
    assert {executor["container"]["image"] for executor in executors} == {image}
    commands = "\n".join(" ".join(executor["container"]["command"]) for executor in executors)
    assert "from ml_labs.movies import train_movies" in commands
    assert "pip install" not in commands
    assert "LinearRegression" not in commands

    placeholder = compile_kfp(tmp_path / "placeholder.yaml", "abc123")
    assert placeholder["image"] == KFP_IMAGE_PLACEHOLDER and "warning" in placeholder
    with pytest.raises(ValueError, match="source-revision"):
        compile_kfp(tmp_path / "missing.yaml", " ", image)


def test_kfp_component_bodies_run_movie_workflow_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("kfp")
    import shutil

    from kfp import dsl

    from ml_labs.integrations import build_kfp_movie_pipeline

    for name in ("MLFLOW_TRACKING_URI", "ML_LABS_CODE_REVISION"):
        monkeypatch.setenv(name, "restored-by-monkeypatch")
    monkeypatch.chdir(tmp_path)
    _pipeline, components = build_kfp_movie_pipeline("local/test:image")

    def artifact(kind, name):
        value = kind(name=name, uri=str(tmp_path / "artifacts" / name), metadata={})
        value.path = str(tmp_path / "artifacts" / name)
        return value

    dataset, model = artifact(dsl.Dataset, "dataset"), artifact(dsl.Model, "model")
    report, metrics = artifact(dsl.Artifact, "report.json"), artifact(dsl.Metrics, "metrics")
    evaluated = artifact(dsl.Model, "evaluated_model")
    root = tmp_path / "pod-mlflow"

    components["prepare"].python_func(source_revision="rev-1", seed=42, users=48, items=32,
                                      dataset=dataset)
    components["train"].python_func(source_revision="rev-1", dataset=dataset, rank=4, seed=42,
                                    mlflow_root=str(root), model=model)
    # Simulate a fresh evaluate pod: release MLflow's cached SQLite engine, then wipe the root
    # so evaluation must restore the MLflow store from the KFP model artifact.
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

    for engine in SqlAlchemyStore._db_uri_sql_alchemy_engine_map.values():
        engine.dispose()
    SqlAlchemyStore._db_uri_sql_alchemy_engine_map.clear()
    shutil.rmtree(root)
    components["evaluate"].python_func(
        source_revision="rev-1", dataset=dataset, model=model, max_rmse=1.5,
        mlflow_root=str(root), report=report, metrics=metrics, evaluated_model=evaluated,
    )

    assert dataset.metadata["source_revision"] == "rev-1"
    assert model.metadata["dataset_sha256"] == dataset.metadata["sha256"]
    assert report.metadata["passed"] is True
    assert "test_recall_at_5" in metrics.metadata
    assert (Path(evaluated.path) / "tracking.db").is_file()


def test_native_evidently_report_writes_html_and_json(tmp_path: Path) -> None:
    pytest.importorskip("evidently")
    from ml_labs.monitoring import create_drift_report

    reference = tmp_path / "reference.csv"
    current = tmp_path / "current.csv"
    pd.DataFrame({"score": [1, 2, 3, 4, 5]}).to_csv(reference, index=False)
    pd.DataFrame({"score": [2, 3, 4, 5, 6]}).to_csv(current, index=False)

    result = create_drift_report(reference, current, tmp_path / "drift.json")

    assert result["backend"] == "evidently"
    assert Path(result["html_report"]).is_file()
    assert Path(result["json_report"]).is_file()


def test_promptfoo_config_is_a_native_eval_suite(tmp_path: Path) -> None:
    from ml_labs.llmops import create_promptfoo_config

    config_path = create_promptfoo_config(tmp_path / "promptfooconfig.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["providers"] == ["openai:chat:gpt-4o-mini"]
    assert len(config["tests"]) == 2
    assert config["tests"][1]["assert"][0]["type"] == "llm-rubric"


def test_promptfoo_runner_handles_windows_command_shim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    from types import SimpleNamespace

    from ml_labs import llmops

    if os.name != "nt":
        pytest.skip("Windows .cmd launcher behavior.")
    config = tmp_path / "promptfooconfig.yaml"
    config.write_text("providers: []\n", encoding="utf-8")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="completed", stderr="")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llmops.shutil, "which", lambda _name: r"C:\tools\promptfoo.cmd")
    monkeypatch.setattr(llmops.subprocess, "run", fake_run)

    result = llmops.run_promptfoo(config)

    assert result["stdout"] == "completed"
    assert captured["shell"] is True
    assert "--config" in captured["command"]


def test_llmops_candidate_guard_rejects_non_rag(tmp_path: Path) -> None:
    from ml_labs.common import write_json
    from ml_labs.llmops import check_candidate

    candidate = write_json(tmp_path / "candidate.json", {"task": "movies"})

    with pytest.raises(ValueError, match="RAG candidate"):
        check_candidate(candidate)


def test_redisvl_adapter_builds_and_queries_native_index_without_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    redisvl = pytest.importorskip("redisvl.index")
    pytest.importorskip("redisvl.query")
    from ml_labs.llmops import redisvl_index, redisvl_query

    class FakeIndex:
        def __init__(self) -> None:
            self.loaded = []
            self.query_value = None

        def create(self, overwrite: bool = False) -> None:
            assert overwrite is False

        def load(self, records, id_field: str) -> None:
            assert id_field == "doc_id"
            self.loaded.extend(records)

        def query(self, query):
            self.query_value = query
            return [{"doc_id": "doc-mlflow-registry", "title": "Model registry"}]

    fake = FakeIndex()
    loaded = []

    def create_without_redis(index, overwrite: bool = False) -> None:
        assert overwrite is False
        assert index.schema.index.name == "ml-labs-documents"

    def load_without_redis(index, records, id_field: str) -> None:
        assert id_field == "doc_id"
        loaded.extend(records)

    monkeypatch.setattr(redisvl.SearchIndex, "create", create_without_redis)
    monkeypatch.setattr(redisvl.SearchIndex, "load", load_without_redis)
    monkeypatch.setattr(
        redisvl.SearchIndex, "from_existing", lambda name, **kwargs: fake
    )
    from ml_labs.rag import prepare_documents

    corpus = prepare_documents(tmp_path / "documents")
    state = tmp_path / "redisvl-state.joblib"

    indexed = redisvl_index(corpus, state, redis_url="redis://localhost:6379")
    results = redisvl_query(
        "How does model registration work?", state, redis_url="redis://localhost:6379"
    )

    assert indexed["documents"] == 10
    assert len(loaded) == 10
    assert results["results"][0]["doc_id"] == "doc-mlflow-registry"
    assert fake.query_value._vector_field_name == "embedding"
