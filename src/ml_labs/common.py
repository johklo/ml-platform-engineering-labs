import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(paths: list[str | Path]) -> str:
    digest = hashlib.sha256()
    normalized = [Path(path).resolve() for path in paths]
    if not normalized:
        raise ValueError("Cannot fingerprint an empty file list.")
    root = Path(os.path.commonpath([str(path.parent) for path in normalized]))
    for path in sorted(normalized, key=lambda item: item.as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit() -> str:
    revision = os.environ.get("ML_LABS_CODE_REVISION")
    if revision:
        return revision
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode or Path(result.stdout.strip()).resolve() != root:
        raise RuntimeError(
            "Code provenance is missing. Clone this repository or set ML_LABS_CODE_REVISION "
            "to the exact source commit before running an uploaded/containerized job."
        )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def setup_tracking(work: str | Path) -> None:
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    else:
        database = Path(work).resolve() / "tracking.db"
        database.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///" + database.as_posix())
    if not mlflow.active_run() and not os.environ.get("AZUREML_RUN_ID"):
        mlflow.set_experiment("ml-platform-labs")


def verify_dataset(data: str | Path, expected: dict | None = None) -> dict:
    data = Path(data)
    manifest = read_json(data / "manifest.json")
    dataset = manifest["dataset"]
    if dataset.endswith("movies") or dataset.startswith("movielens"):
        files = [data / "ratings.csv", data / "items.csv"]
    elif dataset == "synthetic-cards":
        files = list(data.glob("images/*/*.png")) + list(data.glob("labels/*/*.txt"))
    elif dataset == "synthetic-ml-platform-documents":
        files = [data / "documents.jsonl"]
    else:
        raise ValueError(f"Unsupported dataset manifest: {dataset}")
    if fingerprint(files) != manifest["sha256"]:
        raise ValueError("Dataset fingerprint mismatch: data changed after preparation.")
    if expected and expected["sha256"] != manifest["sha256"]:
        raise ValueError("Candidate and evaluation dataset fingerprints differ.")
    return manifest


def record_context(dataset: dict[str, Any], parameters: dict[str, Any]) -> None:
    mlflow.log_params(parameters)
    mlflow.set_tags({
        "git_commit": git_commit(),
        "dataset_sha256": dataset["sha256"],
        "dataset_name": dataset["dataset"],
    })
    mlflow.log_dict(dataset, "lineage/dataset.json")
