"""Single-host release records and atomic pointers for reproducible serving and rollback."""

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd

from ml_labs.common import read_json


def content_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def writer_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".writer.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another writer holds {lock}. After a crashed writer, verify no writer is running "
            "before removing this single lock file."
        ) from exc
    try:
        os.write(descriptor, str(os.getpid()).encode())
        yield
    finally:
        os.close(descriptor)
        lock.unlink()


def checked_candidate(candidate_path, report_path) -> tuple[dict, dict]:
    candidate, report = read_json(candidate_path), read_json(report_path)
    source = report.get("candidate", {})
    if report.get("passed") is not True or any(
        source.get(key) != candidate.get(key) for key in (
            "task", "run_id", "model_uri", "tracking_uri", "dataset",
        )
    ):
        raise ValueError("Release requires this candidate's passing evaluation report.")
    version = candidate.get("registered_version")
    uri = f"models:/{candidate.get('registered_model')}/{version}"
    if not version or candidate.get("registered_uri") != uri:
        raise ValueError("Register and reload a fixed model version before releasing it.")
    if candidate["task"] not in {"movies", "rag"}:
        raise ValueError("Release lifecycle currently supports movies and RAG models.")
    return candidate, report


def _probe(candidate: dict):
    mlflow.set_tracking_uri(candidate["tracking_uri"])
    model = mlflow.pyfunc.load_model(candidate["registered_uri"])
    example = ({"user_id": ["release-cold-start"]} if candidate["task"] == "movies"
               else {"query": ["model registry"]})
    if model.predict(pd.DataFrame(example)).empty:
        raise RuntimeError("Release probe produced no output.")
    return model


def read_release(store, release_id=None) -> dict:
    store = Path(store)
    selected = release_id or read_json(store / "active.json")["release_id"]
    if not isinstance(selected, str) or len(selected) != 64 or any(
        char not in "0123456789abcdef" for char in selected
    ):
        raise ValueError("Invalid release identifier.")
    record = read_json(store / "releases" / f"{selected}.json")
    body = {key: value for key, value in record.items() if key != "release_id"}
    if content_hash(body) != selected:
        raise ValueError("Release record checksum mismatch.")
    return record


def activate_model(store, candidate_path, report_path, fail_before_switch=False) -> dict:
    store = Path(store)
    candidate, report = checked_candidate(candidate_path, report_path)
    _probe(candidate)
    body = {
        "candidate": candidate, "report_sha256": content_hash(report),
        "dataset_sha256": candidate["dataset"]["sha256"],
    }
    record = {**body, "release_id": content_hash(body)}
    with writer_lock(store):
        target = store / "releases" / f"{record['release_id']}.json"
        if not target.exists():
            atomic_json(target, record)
        if fail_before_switch:
            raise RuntimeError("Injected failure before active release pointer replacement.")
        previous = read_json(store / "active.json") if (store / "active.json").exists() else None
        if previous and previous["release_id"] == record["release_id"]:
            return {**record, "changed": False}
        atomic_json(store / "active.json", {
            "release_id": record["release_id"],
            "previous_release_id": previous["release_id"] if previous else None,
            "activated_at": datetime.now(timezone.utc).isoformat(),
        })
    return {**record, "changed": True}


def rollback_model(store, release_id=None) -> dict:
    store = Path(store)
    with writer_lock(store):
        current = read_json(store / "active.json")
        target_id = release_id or current.get("previous_release_id")
        if not target_id:
            raise ValueError("There is no previous release to restore.")
        target = read_release(store, target_id)
        _probe(target["candidate"])
        atomic_json(store / "active.json", {
            "release_id": target["release_id"], "previous_release_id": current["release_id"],
            "activated_at": datetime.now(timezone.utc).isoformat(),
        })
    return target
