from pathlib import Path

import pandas as pd
import pytest

from ml_labs.monitoring import create_drift_report
from ml_labs.rag import (
    evaluate_retrieval,
    index_documents,
    prepare_documents,
    query_azure,
    query_local,
)


def test_local_rag_logs_retrieval_and_returns_explicit_extractive_mode(tmp_path: Path) -> None:
    data = prepare_documents(tmp_path / "documents")
    candidate = index_documents(data, tmp_path / "rag-model", work=tmp_path / "tracking")

    response = query_local(candidate, "How does a model registry promote a candidate?")
    report = evaluate_retrieval(candidate, tmp_path / "retrieval.json")

    assert response["answer_mode"] == "extractive_offline"
    assert "doc-mlflow-registry" in response["citations"]
    assert report["passed"] is True
    assert report["retrieval_recall_at_3"] == 1.0


def test_psi_report_detects_distribution_shift(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    current = tmp_path / "current.csv"
    pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4, 0.5]}).to_csv(reference, index=False)
    pd.DataFrame({"score": [4.1, 4.2, 4.3, 4.4, 4.5]}).to_csv(current, index=False)

    result = create_drift_report(reference, current, tmp_path / "drift.json", backend="psi")

    assert result["features"]["score"] > 0.2
    assert result["drifted_features_psi_gt_0_2"] == ["score"]


def test_azure_rag_requires_explicit_existing_resource_configuration(monkeypatch) -> None:
    for name in (
        "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_INDEX",
        "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="AZURE_SEARCH_ENDPOINT"):
        query_azure("What is model monitoring?")
