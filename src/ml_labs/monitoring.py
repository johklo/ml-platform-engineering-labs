"""Reference/current data drift reports with native Evidently or an explicit PSI baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_labs.common import write_json


def _psi(reference: pd.Series, current: pd.Series) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if ref.size < 2 or cur.size < 2:
        raise ValueError("PSI needs at least two non-null numeric observations per population.")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if edges.size < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    epsilon = 1e-6
    ref_share = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
    cur_share = np.clip(cur_counts / cur_counts.sum(), epsilon, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def create_drift_report(reference_path: str | Path, current_path: str | Path,
                        output: str | Path, backend: str = "evidently") -> dict:
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    numeric_columns = sorted(
        set(reference.select_dtypes(include="number").columns)
        & set(current.select_dtypes(include="number").columns)
    )
    if not numeric_columns:
        raise ValueError("Reference and current CSVs need at least one shared numeric feature.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if backend == "psi":
        metrics = {column: _psi(reference[column], current[column]) for column in numeric_columns}
        result = {
            "backend": "psi",
            "reference_rows": len(reference),
            "current_rows": len(current),
            "features": metrics,
            "drifted_features_psi_gt_0_2": [
                column for column, value in metrics.items() if value > 0.2
            ],
            "interpretation": "PSI is a monitoring signal, not proof of model failure.",
        }
        write_json(output, result)
        return result
    if backend != "evidently":
        raise ValueError("backend must be 'evidently' or 'psi'.")
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError as exc:
        raise RuntimeError("Install Evidently with `pip install -e .[monitor]`.") from exc
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    html_path = output.with_suffix(".html")
    report.save_html(str(html_path))
    result = report.as_dict()
    write_json(output.with_suffix(".json"), result)
    return {
        "backend": "evidently",
        "html_report": str(html_path),
        "json_report": str(output.with_suffix(".json")),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "monitoring_note": "Drift is a signal, not proof of model failure.",
    }
