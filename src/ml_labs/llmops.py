"""Optional native LLMOps integrations; external services are always explicit."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from ml_labs.common import read_json
from ml_labs.rag import _load_documents, query_local


def create_promptfoo_config(output: str | Path) -> Path:
    """Write a real Promptfoo eval config; execution requires its CLI and OpenAI credentials."""
    config = {
        "description": "ML Platform Labs grounded-answer quality starter",
        "prompts": [
            (
                "Answer the question using only the context. If unsupported, say so.\n"
                "Context: {{context}}\nQuestion: {{question}}"
            )
        ],
        "providers": ["openai:chat:gpt-4o-mini"],
        "tests": [
            {
                "vars": {
                    "question": "What does an MLflow model registry do?",
                    "context": (
                        "A model registry versions named models and supports aliases such as "
                        "champion. A candidate should pass validation before promotion."
                    ),
                },
                "assert": [{"type": "contains", "value": "registry"}],
            },
            {
                "vars": {
                    "question": "What is the capital of Mars?",
                    "context": "The local synthetic corpus contains no information about Mars.",
                },
                "assert": [{"type": "llm-rubric", "value": "The response states it cannot answer."}],
            },
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output


def run_promptfoo(config: str | Path) -> dict[str, Any]:
    executable = shutil.which("promptfoo")
    if not executable:
        raise RuntimeError(
            "Promptfoo CLI is not installed. Install it with `npm install -g promptfoo`, "
            "set OPENAI_API_KEY, then run this command again."
        )
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Promptfoo's OpenAI provider needs OPENAI_API_KEY.")
    config = Path(config).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    command = [executable, "eval", "--config", str(config)]
    if os.name == "nt" and Path(executable).suffix.lower() in {".cmd", ".bat"}:
        result = subprocess.run(
            subprocess.list2cmdline(command),
            shell=True,
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Promptfoo evaluation failed:\n{result.stderr or result.stdout}")
    return {"config": str(config), "stdout": result.stdout.strip()}


def validate_with_guardrails(answer: str, citations: list[str]) -> dict[str, Any]:
    """Validate an answer object with Guardrails AI's local Pydantic guard."""
    try:
        from guardrails import Guard
        from pydantic import BaseModel, ConfigDict, Field
    except ImportError as exc:
        raise RuntimeError("Install Guardrails AI using `uv sync --extra guardrails`.") from exc
    from guardrails.settings import settings

    class GroundedAnswer(BaseModel):
        model_config = ConfigDict(extra="forbid")
        answer: str = Field(min_length=1)
        citations: list[str]

    settings.rc.enable_metrics = False
    from guardrails.utils.hub_telemetry_utils import HubTelemetry

    if HubTelemetry._instance is not None:
        HubTelemetry._instance._enabled = False
    guard = Guard.for_pydantic(GroundedAnswer)
    guard.configure(allow_metrics_collection=False)
    result = guard.parse(json.dumps({"answer": answer, "citations": citations}))
    return {
        "valid": bool(result.validation_passed),
        "validated_output": result.validated_output,
    }


def evaluate_with_deepeval(
    candidate_path: str | Path, question: str, threshold: float = 0.7
) -> dict[str, Any]:
    """Run DeepEval's model-based answer relevance metric (requires an LLM API key)."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("DeepEval's default evaluator needs OPENAI_API_KEY.")
    try:
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise RuntimeError("Install DeepEval using `uv sync --extra deepeval`.") from exc
    response = query_local(candidate_path, question)
    case = LLMTestCase(input=question, actual_output=response["answer"])
    metric = AnswerRelevancyMetric(threshold=threshold, include_reason=True)
    metric.measure(case)
    report = {
        "framework": "deepeval",
        "metric": "answer_relevancy",
        "question": question,
        "answer_mode": response["answer_mode"],
        "score": float(metric.score),
        "threshold": threshold,
        "passed": bool(metric.is_successful()),
        "reason": metric.reason,
        "limitation": "LLM-judged score; evaluator model and prompt can affect results.",
    }
    return report


def trace_with_langfuse(candidate_path: str | Path, question: str) -> dict[str, Any]:
    """Trace local retrieval as a span; it does not claim an LLM generation occurred."""
    if not (os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY")):
        raise RuntimeError("Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY to send a trace.")
    try:
        from langfuse import Langfuse
    except ImportError as exc:
        raise RuntimeError("Install Langfuse using `uv sync --extra langfuse`.") from exc
    client = Langfuse()
    trace = client.trace(
        name="ml-labs-local-rag",
        input={"query": question},
    )
    span = trace.span(
        name="tfidf-retrieval",
        input={"query": question},
    )
    answer = query_local(candidate_path, question)
    span.end(output={"answer": answer["answer"], "citations": answer["citations"]})
    trace.update(output={"answer_mode": answer["answer_mode"], "citations": answer["citations"]})
    client.flush()
    return {
        "trace_id": trace.id,
        "answer_mode": answer["answer_mode"],
        "sent_to_langfuse": True,
    }


def redisvl_index(
    data_path: str | Path,
    state_path: str | Path,
    index_name: str = "ml-labs-documents",
    redis_url: str | None = None,
) -> dict[str, Any]:
    """Create a RedisVL vector index from the synthetic corpus and local TF-IDF vectors."""
    try:
        from redisvl.index import SearchIndex
    except ImportError as exc:
        raise RuntimeError("Install RedisVL using `uv sync --extra redisvl`.") from exc
    documents = _load_documents(Path(data_path) / "documents.jsonl")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform([doc["text"] for doc in documents]).toarray().astype(np.float32)
    schema = {
        "index": {
            "name": index_name,
            "prefix": f"{index_name}:",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "doc_id", "type": "tag"},
            {"name": "title", "type": "text"},
            {"name": "content", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": int(matrix.shape[1]),
                    "algorithm": "flat",
                    "distance_metric": "cosine",
                    "datatype": "float32",
                },
            },
        ],
    }
    index = SearchIndex.from_dict(schema, redis_url=redis_url or os.environ.get(
        "REDIS_URL", "redis://localhost:6379"
    ))
    index.create(overwrite=False)
    records = [
        {
            "doc_id": document["id"],
            "title": document["title"],
            "content": document["text"],
            "embedding": vector.tolist(),
        }
        for document, vector in zip(documents, matrix, strict=True)
    ]
    index.load(records, id_field="doc_id")
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "index_name": index_name}, state_path)
    return {
        "index": index_name,
        "documents": len(records),
        "dimensions": int(matrix.shape[1]),
        "state": str(state_path),
    }


def redisvl_query(
    query: str,
    state_path: str | Path,
    top_k: int = 3,
    redis_url: str | None = None,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    try:
        from redisvl.index import SearchIndex
        from redisvl.query import VectorQuery
    except ImportError as exc:
        raise RuntimeError("Install RedisVL using `uv sync --extra redisvl`.") from exc
    state = joblib.load(state_path)
    vector = state["vectorizer"].transform([query]).toarray()[0].astype(np.float32)
    index = SearchIndex.from_existing(
        state["index_name"],
        redis_url=redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379"),
    )
    results = index.query(VectorQuery(
        vector=vector.tolist(),
        vector_field_name="embedding",
        num_results=top_k,
        return_fields=["doc_id", "title", "content"],
    ))
    return {"query": query, "backend": "redisvl", "results": results}


def check_candidate(candidate_path: str | Path) -> dict[str, Any]:
    candidate = read_json(candidate_path)
    if candidate.get("task") != "rag":
        raise ValueError("LLMOps adapters in this module require a RAG candidate.")
    return candidate
