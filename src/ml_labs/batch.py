"""Retry-safe batch inference: publish a completed result directory only once."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from ml_labs.common import read_json, write_json
from ml_labs.releases import checked_candidate, content_hash, writer_lock


def batch_recommend(candidate, report, users, output, job_key, fail_before_commit=False):
    candidate, _ = checked_candidate(candidate, report)
    if candidate["task"] != "movies":
        raise ValueError("Batch recommendations require a movie candidate.")
    if not job_key.strip():
        raise ValueError("A nonempty business job key is required.")
    users = Path(users)
    frame = pd.read_csv(users, dtype={"user_id": str})
    if (set(frame.columns) != {"user_id"} or frame.empty or frame.user_id.isna().any()
            or frame.user_id.duplicated().any() or frame.user_id.str.strip().eq("").any()):
        raise ValueError("Input must have exactly user_id, unique nonempty user rows.")
    identity = {
        "job_key": job_key, "model_uri": candidate["registered_uri"],
        "tracking_uri": candidate["tracking_uri"],
        "users_sha256": hashlib.sha256(users.read_bytes()).hexdigest(),
        "dataset_sha256": candidate["dataset"]["sha256"],
    }
    output = Path(output)
    destination = output / content_hash({"job_key": job_key})
    with writer_lock(output):
        if destination.exists():
            committed = read_json(destination / "manifest.json")
            if committed["identity"] != identity:
                raise ValueError("Business job key already belongs to different inputs or model.")
            predictions = destination / "predictions.json"
            if hashlib.sha256(predictions.read_bytes()).hexdigest() != committed["output_sha256"]:
                raise ValueError("Committed batch output checksum mismatch.")
            return {"output": str(destination), "reused": True, **committed}
        mlflow.set_tracking_uri(candidate["tracking_uri"])
        model = mlflow.pyfunc.load_model(candidate["registered_uri"])
        result = model.predict(frame).to_dict(orient="records")
        for row in result:
            row["recommendations"] = json.loads(row["recommendations"])
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=output) as staging:
            pending = Path(staging)
            predictions = write_json(pending / "predictions.json", result)
            manifest = {
                "identity": identity, "rows": len(result),
                "output_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
            }
            write_json(pending / "manifest.json", manifest)
            if fail_before_commit:
                raise RuntimeError("Injected failure after inference, before batch publication.")
            # Rename on the same filesystem exposes either the complete bundle or nothing.
            os.rename(pending, destination)
    return {"output": str(destination), "reused": False, **manifest}
