from pathlib import Path

import pandas as pd
import pytest
import yaml


def test_native_feast_apply_materialize_online_and_historical(tmp_path: Path) -> None:
    pytest.importorskip("feast")
    pytest.importorskip("pyarrow")
    from ml_labs.integrations import feast_init, feast_run

    repo = feast_init(tmp_path / "feast")
    assert feast_run(repo, "apply")["status"] == "applied"
    assert feast_run(repo, "materialize")["status"] == "materialized"
    assert feast_run(repo, "online")["features"]["average_rating"] == [3.5]
    assert feast_run(repo, "historical")["rows"][0]["ratings_count"] == 8


def test_native_kfp_graph_compiles_without_cluster(tmp_path: Path) -> None:
    pytest.importorskip("kfp")
    from ml_labs.integrations import compile_kfp

    pipeline = compile_kfp(tmp_path / "pipeline.yaml")

    assert pipeline.is_file()
    assert "train" in pipeline.read_text(encoding="utf-8")


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
            assert overwrite is True

        def load(self, records, id_field: str) -> None:
            assert id_field == "doc_id"
            self.loaded.extend(records)

        def query(self, query):
            self.query_value = query
            return [{"doc_id": "doc-mlflow-registry", "title": "Model registry"}]

    fake = FakeIndex()
    loaded = []

    def create_without_redis(index, overwrite: bool = False) -> None:
        assert overwrite is True
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
