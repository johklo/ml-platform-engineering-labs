from pathlib import Path

import pytest
import yaml

from ml_labs.azure_jobs import render_pipeline
from ml_labs.cli import make_parser
from ml_labs.common import write_json
from ml_labs.registry import register_candidate


def test_cli_exposes_each_learning_path() -> None:
    parser = make_parser()
    cards = parser.parse_args(["cards", "train", "--backend", "color_component_baseline"])
    rag = parser.parse_args(["rag", "query", "--query", "What is MLflow?"])
    feast = parser.parse_args(["tools", "feast", "run", "--action", "historical"])

    assert cards.backend == "color_component_baseline"
    assert rag.provider == "local"
    assert feast.action == "historical"


def test_azure_pipeline_wires_each_data_model_and_report_artifact(tmp_path: Path) -> None:
    path = render_pipeline(tmp_path / "pipeline.yml", compute="azureml:existing-cpu")
    pipeline = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert set(pipeline["jobs"]) == {
        "prepare_cards", "train_cards", "evaluate_cards",
        "prepare_movies", "train_movies", "evaluate_movies",
        "prepare_rag", "train_rag", "evaluate_rag",
    }
    assert pipeline["jobs"]["train_movies"]["inputs"]["data"] == (
        "${{parent.jobs.prepare_movies.outputs.data}}"
    )
    assert pipeline["jobs"]["evaluate_rag"]["inputs"]["model"] == (
        "${{parent.jobs.train_rag.outputs.model}}"
    )
    assert pipeline["jobs"]["evaluate_cards"]["outputs"]["report"]["type"] == "uri_folder"
    assert pipeline["settings"]["default_compute"] == "azureml:existing-cpu"


def test_registry_rejects_missing_or_failed_gate_before_mlflow_access(tmp_path: Path) -> None:
    candidate = write_json(tmp_path / "candidate.json", {
        "task": "movies",
        "run_id": "run-1",
        "model_uri": "runs:/run-1/model",
        "tracking_uri": "sqlite:///not-opened.db",
    })
    report = write_json(tmp_path / "report.json", {
        "task": "movies", "passed": False, "candidate": {"run_id": "run-1"},
    })

    with pytest.raises(ValueError, match="passing"):
        register_candidate(candidate, "should-not-register", report)


def test_azure_sdk_loads_rendered_components_without_contacting_workspace(tmp_path: Path) -> None:
    azure = pytest.importorskip("azure.ai.ml")
    path = render_pipeline(tmp_path / "pipeline.yml", cards_backend="yolov8")
    job = azure.load_job(source=str(path))
    assert not job._validate().error_messages
    assert len(job.jobs) == 9
    component = yaml.safe_load(
        (tmp_path / "pipeline-components" / "train_cards.yml").read_text(encoding="utf-8")
    )
    assert "--backend yolov8" in component["command"]
    assert component["environment"]["conda_file"].endswith("environment-vision.yml")
    assert "ML_LABS_CODE_REVISION=" in component["command"]
