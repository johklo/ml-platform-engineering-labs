"""Azure ML SDK v2 renderers; rendering is local and never provisions resources."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

from ml_labs.common import read_json

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"


def render_pipeline(output: str | Path, compute: str = "azureml:cpu-cluster") -> Path:
    """Render cards, recommendation, and RAG jobs with explicit output dependencies."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    component_dir = output.parent / f"{output.stem}-components"
    component_dir.mkdir(parents=True, exist_ok=True)
    code_path = os.path.relpath(ROOT, component_dir.resolve()).replace("\\", "/")
    environment_path = os.path.relpath(
        ROOT / "infra" / "azure" / "environment.yml", component_dir.resolve()
    ).replace("\\", "/")
    jobs = {}
    for task in ("cards", "movies", "rag"):
        prepare_job = f"prepare_{task}"
        train_job = f"train_{task}"
        evaluate_job = f"evaluate_{task}"
        prepare_component = component_dir / f"{prepare_job}.yml"
        _write_component(
            prepare_component,
            name=prepare_job.replace("_", "-"),
            command=(
                f"PYTHONPATH=src python -m ml_labs.cli {task} prepare "
                "--output ${{outputs.data}}"
            ),
            code_path=code_path,
            environment_path=environment_path,
            outputs={"data": {"type": "uri_folder"}},
        )
        jobs[prepare_job] = {
            "type": "command",
            "component": f"{component_dir.name}/{prepare_component.name}",
            "compute": compute,
            "outputs": {"data": {"type": "uri_folder"}},
        }
        train_command = (
            f"PYTHONPATH=src python -m ml_labs.cli {task} "
            f"{'index' if task == 'rag' else 'train'} "
            "--data ${{inputs.data}} --output ${{outputs.model}}"
        )
        if task == "cards":
            train_command += " --backend color_component_baseline"
        train_component = component_dir / f"{train_job}.yml"
        _write_component(
            train_component,
            name=train_job.replace("_", "-"),
            command=train_command,
            code_path=code_path,
            environment_path=environment_path,
            inputs={"data": {"type": "uri_folder"}},
            outputs={"model": {"type": "uri_folder"}},
        )
        jobs[train_job] = {
            "type": "command",
            "component": f"{component_dir.name}/{train_component.name}",
            "compute": compute,
            "inputs": {"data": f"${{{{parent.jobs.{prepare_job}.outputs.data}}}}"},
            "outputs": {"model": {"type": "uri_folder"}},
        }
        evaluate_command = (
            f"PYTHONPATH=src python -m ml_labs.cli {task} evaluate "
            f"--candidate ${{{{inputs.model}}}}/candidate.json "
            "--report ${{outputs.report}}/report.json"
        )
        if task != "rag":
            evaluate_command += " --data ${{inputs.data}}"
        evaluate_component = component_dir / f"{evaluate_job}.yml"
        evaluate_inputs = {"model": {"type": "uri_folder"}}
        if task != "rag":
            evaluate_inputs["data"] = {"type": "uri_folder"}
        _write_component(
            evaluate_component,
            name=evaluate_job.replace("_", "-"),
            command=evaluate_command,
            code_path=code_path,
            environment_path=environment_path,
            inputs=evaluate_inputs,
            outputs={"report": {"type": "uri_folder"}},
        )
        jobs[evaluate_job] = {
            "type": "command",
            "component": f"{component_dir.name}/{evaluate_component.name}",
            "compute": compute,
            "inputs": {
                "model": f"${{{{parent.jobs.{train_job}.outputs.model}}}}",
                **(
                    {"data": f"${{{{parent.jobs.{prepare_job}.outputs.data}}}}"}
                    if task != "rag" else {}
                ),
            },
            "outputs": {"report": {"type": "uri_folder"}},
        }
    job = {
        "$schema": "https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json",
        "type": "pipeline",
        "display_name": "ml-platform-engineering-labs",
        "experiment_name": "ml-platform-engineering-labs",
        "description": "Synthetic cards, movie recommendation, and local extractive RAG.",
        "settings": {"default_compute": compute},
        "jobs": jobs,
    }
    output.write_text(yaml.safe_dump(job, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output


def _write_component(path: Path, name: str, command: str, code_path: str,
                     environment_path: str, inputs: dict | None = None,
                     outputs: dict | None = None) -> None:
    component = {
        "$schema": "https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json",
        "name": name,
        "type": "command",
        "is_deterministic": False,
        "code": code_path,
        "command": command,
        "environment": {
            "image": ENVIRONMENT_IMAGE,
            "conda_file": environment_path,
        },
    }
    if inputs:
        component["inputs"] = inputs
    if outputs:
        component["outputs"] = outputs
    path.write_text(yaml.safe_dump(component, sort_keys=False, allow_unicode=True), encoding="utf-8")


def render_endpoint(candidate_path: str | Path, output: str | Path,
                    report_path: str | Path) -> Path:
    """Export an MLflow artifact and write an explicit opt-in endpoint deployment script."""
    import mlflow

    candidate = read_json(candidate_path)
    report = read_json(report_path)
    report_candidate = report.get("candidate", {})
    if (
        report.get("task") != candidate.get("task")
        or report.get("passed") is not True
        or report_candidate.get("run_id") != candidate.get("run_id")
    ):
        raise ValueError("Endpoint rendering requires a passing evaluation report for the candidate.")
    mlflow.set_tracking_uri(candidate["tracking_uri"])
    model_source = Path(mlflow.artifacts.download_artifacts(candidate["model_uri"]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "model"
    if model_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing model export: {model_dir}")
    shutil.copytree(model_source, model_dir)
    script = output / "deploy_endpoint.py"
    script.write_text(
        '''"""Explicit-cost Azure ML endpoint deployment; review before running.
Requires an existing workspace, network access, permissions, and paid online compute.
Run: python deploy_endpoint.py --endpoint-name <globally-unique-name>
"""
import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint, Model
from azure.identity import DefaultAzureCredential

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint-name", required=True)
parser.add_argument("--instance-type", default="Standard_DS3_v2")
args = parser.parse_args()
client = MLClient.from_config(credential=DefaultAzureCredential())
endpoint = ManagedOnlineEndpoint(name=args.endpoint_name, auth_mode="key")
client.online_endpoints.begin_create_or_update(endpoint).result()
model = Model(path="./model", type=AssetTypes.MLFLOW)
deployment = ManagedOnlineDeployment(
    name="blue", endpoint_name=args.endpoint_name, model=model,
    instance_type=args.instance_type, instance_count=1,
)
client.online_deployments.begin_create_or_update(deployment).result()
endpoint = client.online_endpoints.get(args.endpoint_name)
endpoint.traffic = {"blue": 100}
client.online_endpoints.begin_create_or_update(endpoint).result()
print(f"Deployment ready: {args.endpoint_name}/blue")
''',
        encoding="utf-8",
    )
    return script


def submit_existing_workspace(pipeline_path: str | Path) -> dict[str, str]:
    """Submit to a workspace already selected by Azure ML config; creates no workspace."""
    try:
        from azure.ai.ml import MLClient, load_job
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        raise RuntimeError("Install the optional Azure SDK with `pip install -e .[azure]`.") from exc
    client = MLClient.from_config(credential=DefaultAzureCredential())
    workspace = client.workspaces.get(client.workspace_name)
    job = load_job(source=str(pipeline_path))
    tracking_uri = workspace.mlflow_tracking_uri
    if not tracking_uri:
        raise RuntimeError("The selected Azure ML workspace did not return an MLflow tracking URI.")
    for component_job in job.jobs.values():
        variables = dict(component_job.environment_variables or {})
        variables["MLFLOW_TRACKING_URI"] = tracking_uri
        component_job.environment_variables = variables
    submitted = client.jobs.create_or_update(job)
    return {"job_name": submitted.name, "status": submitted.status}
