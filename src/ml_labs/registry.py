"""MLflow model registry promotion and reload checks."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from ml_labs.common import read_json, write_json


def register_candidate(candidate_path: str | Path, model_name: str, report_path: str | Path,
                       alias: str = "champion") -> dict:
    candidate_path = Path(candidate_path)
    candidate = read_json(candidate_path)
    if candidate.get("task") not in {"cards", "movies", "rag"}:
        raise ValueError(f"Unsupported candidate task: {candidate.get('task')!r}")
    report = read_json(report_path)
    report_candidate = report.get("candidate", {})
    if (
        report.get("task") != candidate["task"]
        or report.get("passed") is not True
        or report_candidate.get("run_id") != candidate["run_id"]
        or report_candidate.get("model_uri") != candidate["model_uri"]
        or report_candidate.get("tracking_uri") != candidate["tracking_uri"]
    ):
        raise ValueError(
            f"Registry promotion requires a passing {candidate['task']} evaluation report."
        )
    mlflow.set_tracking_uri(candidate["tracking_uri"])
    registered = mlflow.register_model(candidate["model_uri"], model_name)
    client = MlflowClient()
    uri = f"models:/{model_name}/{registered.version}"
    model = mlflow.pyfunc.load_model(uri)
    task = candidate["task"]
    if task == "cards":
        image = next((Path(candidate["data_path"]) / "images" / "test").glob("*.png"))
        probe = pd.DataFrame({"image_path": [str(image.resolve())]})
    elif task == "movies":
        probe = pd.DataFrame({"user_id": ["registry-cold-start-probe"]})
    else:
        probe = pd.DataFrame({"query": ["model registry version"]})
    result = model.predict(probe)
    if result.empty:
        raise RuntimeError("Registered model reload returned no prediction rows.")
    client.set_registered_model_alias(model_name, alias, registered.version)
    candidate.update({
        "registered_model": model_name,
        "registered_version": str(registered.version),
        "registered_alias": alias,
        "registered_uri": uri,
    })
    write_json(candidate_path, candidate)
    with mlflow.start_run(run_id=candidate["run_id"]):
        mlflow.set_tags({
            "registered_model": model_name,
            "registered_version": str(registered.version),
            "registered_alias": alias,
            "registry_reload_verified": "true",
        })
        mlflow.log_dict(
            {
                "model_uri": uri, "probe_task": task, "rows": len(result),
                "evaluation_report": str(Path(report_path).resolve()),
            },
            "registry/reload-check.json",
        )
    return candidate
