"""Local document retrieval and explicitly extractive offline RAG baseline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ml_labs.common import (
    fingerprint,
    read_json,
    record_context,
    setup_tracking,
    verify_dataset,
    write_json,
)

_DOCUMENTS = [
    {
        "id": "doc-mlflow-tracking",
        "title": "MLflow experiment tracking",
        "text": "MLflow tracks parameters, metrics, tags, artifacts, and model files in runs. "
        "A run can link its dataset fingerprint and source revision for reproducible lineage.",
    },
    {
        "id": "doc-mlflow-registry",
        "title": "Model registry and promotion",
        "text": "A model registry versions named models and supports aliases such as champion. "
        "A candidate should pass validation before promotion and deployment.",
    },
    {
        "id": "doc-feast",
        "title": "Feature stores",
        "text": "Feast separates feature definitions from materialized feature values. "
        "An offline store supports historical training retrieval and an online store serves "
        "low-latency latest values.",
    },
    {
        "id": "doc-kubeflow",
        "title": "Kubeflow Pipelines",
        "text": "Kubeflow Pipelines composes containerized components into a workflow DAG. "
        "Components should exchange declared inputs and outputs instead of hidden local files.",
    },
    {
        "id": "doc-bentoml",
        "title": "BentoML serving",
        "text": "BentoML packages a model and its service into a versioned Bento. "
        "A service defines typed API endpoints and can be run locally or deployed.",
    },
    {
        "id": "doc-evidently",
        "title": "Data and model drift",
        "text": "Evidently compares current data with a reference dataset to produce data "
        "quality and drift reports. Drift is a monitoring signal and is not proof of model failure.",
    },
    {
        "id": "doc-azure-ml",
        "title": "Azure Machine Learning",
        "text": "Azure Machine Learning pipelines use registered components and named inputs "
        "and outputs. Compute and workspace resources must exist before cloud submission.",
    },
    {
        "id": "doc-identity-safety",
        "title": "Synthetic computer vision data",
        "text": "Synthetic ID-card-shaped rectangles teach object detection without real identity "
        "documents, personal data, or optical character recognition.",
    },
    {
        "id": "doc-movie-splits",
        "title": "Recommendation evaluation",
        "text": "Recommendation events should use a per-user chronological train, validation, "
        "and test split. Keep test events out of model selection and exclude training items.",
    },
    {
        "id": "doc-llm-eval",
        "title": "LLMOps evaluation",
        "text": "LLM evaluation checks retrieval relevance, groundedness, answer quality, latency, "
        "and safety. Deterministic offline tests are useful but do not replace model-based evaluation.",
    },
]


def prepare_documents(output: str | Path) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    document_file = output / "documents.jsonl"
    document_file.write_text(
        "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in _DOCUMENTS),
        encoding="utf-8",
    )
    write_json(output / "manifest.json", {
        "dataset": "synthetic-ml-platform-documents",
        "rows": len(_DOCUMENTS),
        "sha256": fingerprint([document_file]),
        "limitations": "Synthetic educational reference snippets; no external sources.",
    })
    return output


def _load_documents(path: str | Path) -> list[dict[str, str]]:
    documents = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not documents:
        raise ValueError(f"Document corpus is empty: {path}")
    ids = [doc.get("id") for doc in documents]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Every document must have a unique, nonempty id.")
    if any(not doc.get("text", "").strip() for doc in documents):
        raise ValueError("Every document must contain nonempty text.")
    return documents


class LocalRAG(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.documents = _load_documents(context.artifacts["documents"])
        self.vectorizer = joblib.load(context.artifacts["vectorizer"])
        self.matrix = self.vectorizer.transform([doc["text"] for doc in self.documents])
        self.top_k = int(read_json(context.artifacts["config"])["top_k"])

    def predict(self, context, model_input, params=None):
        outputs = []
        for query in model_input["query"]:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a nonempty string.")
            vector = self.vectorizer.transform([query.strip()])
            scores = cosine_similarity(vector, self.matrix)[0]
            indices = np.argsort(-scores, kind="stable")[: self.top_k]
            retrieved = [
                {
                    "id": self.documents[index]["id"],
                    "title": self.documents[index]["title"],
                    "score": float(scores[index]),
                    "text": self.documents[index]["text"],
                }
                for index in indices
                if scores[index] > 0
            ]
            if not retrieved:
                outputs.append({
                    "query": query,
                    "answer_mode": "extractive_offline",
                    "answer": "No relevant passage was found in the local corpus.",
                    "citations": "[]",
                })
                continue
            # This is quote-based retrieval, not a language model or generated answer.
            answer = " ".join(f"[{item['id']}] {item['text']}" for item in retrieved)
            outputs.append({
                "query": query,
                "answer_mode": "extractive_offline",
                "answer": answer,
                "citations": json.dumps([item["id"] for item in retrieved]),
            })
        return pd.DataFrame(outputs)


def index_documents(data: str | Path, output: str | Path, top_k: int = 3,
                    work: str | Path = "work") -> Path:
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    data, output = Path(data), Path(output)
    manifest = verify_dataset(data)
    documents = _load_documents(data / "documents.jsonl")
    output.mkdir(parents=True, exist_ok=True)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    vectorizer.fit([doc["text"] for doc in documents])
    vectorizer_path = output / "vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    config_path = write_json(output / "config.json", {"top_k": top_k})
    docs_path = data / "documents.jsonl"

    setup_tracking(work)
    with mlflow.start_run() as run:
        record_context(manifest, {"top_k": top_k, "retriever": "tfidf"})
        mlflow.log_param("answer_mode", "extractive_offline")
        model = LocalRAG()
        model.load_context(type("Context", (), {"artifacts": {
            "documents": str(docs_path), "vectorizer": str(vectorizer_path),
            "config": str(config_path),
        }})())
        example = pd.DataFrame({"query": ["How does the model registry work?"]})
        mlflow.pyfunc.log_model(
            "model",
            python_model=model,
            artifacts={
                "documents": str(docs_path),
                "vectorizer": str(vectorizer_path),
                "config": str(config_path),
            },
            code_paths=[str(Path(__file__).parent)],
            signature=infer_signature(example, model.predict(None, example)),
            input_example=example,
            pip_requirements=[
                "mlflow==2.16.2", "numpy==1.26.4", "pandas==2.2.3",
                "scikit-learn==1.5.2", "joblib==1.4.2",
            ],
        )
        candidate = {
            "task": "rag",
            "answer_mode": "extractive_offline",
            "retriever": "tfidf",
            "run_id": run.info.run_id,
            "model_uri": f"runs:/{run.info.run_id}/model",
            "tracking_uri": mlflow.get_tracking_uri(),
            "data_path": str(data.resolve()),
            "dataset": manifest,
        }
        write_json(output / "candidate.json", candidate)
    return output / "candidate.json"


def query_local(candidate_path: str | Path, query: str) -> dict[str, Any]:
    candidate = read_json(candidate_path)
    if candidate.get("answer_mode") != "extractive_offline":
        raise ValueError("Local query requires an extractive_offline candidate.")
    mlflow.set_tracking_uri(candidate["tracking_uri"])
    result = mlflow.pyfunc.load_model(candidate["model_uri"]).predict(
        pd.DataFrame({"query": [query]})
    ).iloc[0]
    return {
        "query": result["query"],
        "answer_mode": result["answer_mode"],
        "answer": result["answer"],
        "citations": json.loads(result["citations"]),
    }


def evaluate_retrieval(candidate_path: str | Path, output: str | Path) -> dict[str, Any]:
    candidate = read_json(candidate_path)
    mlflow.set_tracking_uri(candidate["tracking_uri"])
    model = mlflow.pyfunc.load_model(candidate["model_uri"])
    cases = [
        ("How does model registration work?", "doc-mlflow-registry"),
        ("What is a feature store?", "doc-feast"),
        ("per-user chronological train validation test split", "doc-movie-splits"),
        ("What does data drift mean?", "doc-evidently"),
    ]
    rows = model.predict(pd.DataFrame({"query": [query for query, _ in cases]}))
    hits = [
        expected in json.loads(row.citations)
        for row, (_, expected) in zip(rows.itertuples(index=False), cases, strict=True)
    ]
    report = {
        "task": "rag",
        "passed": all(hits),
        "retrieval_recall_at_3": float(np.mean(hits)),
        "candidate": candidate,
        "cases": [
            {"query": query, "expected_document": expected, "hit": hit}
            for (query, expected), hit in zip(cases, hits, strict=True)
        ],
        "answer_mode": "extractive_offline",
        "limitation": "Retrieval smoke metric only; does not evaluate generated LLM answers.",
    }
    write_json(output, report)
    with mlflow.start_run(run_id=candidate["run_id"]):
        mlflow.log_metric("retrieval_recall_at_3", report["retrieval_recall_at_3"])
        mlflow.log_dict(report, "evaluation/retrieval.json")
        mlflow.set_tag("retrieval_passed", str(report["passed"]).lower())
    if not report["passed"]:
        raise ValueError(f"Retrieval quality gate failed; report: {output}")
    return report


def query_azure(query: str, top_k: int = 3) -> dict[str, Any]:
    """Query existing Azure AI Search and Azure OpenAI resources; creates nothing."""
    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    index_name = os.environ.get("AZURE_SEARCH_INDEX")
    openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not all((endpoint, index_name, openai_endpoint, deployment)):
        raise RuntimeError(
            "Azure RAG requires AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX, "
            "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT."
        )
    try:
        from azure.identity import DefaultAzureCredential
        from azure.search.documents import SearchClient
        from openai import AzureOpenAI
    except ImportError as exc:
        raise RuntimeError("Install Azure integrations with `pip install -e .[azure]`.") from exc
    credential = DefaultAzureCredential()
    search = SearchClient(endpoint, index_name, credential)
    hits = list(search.search(search_text=query, top=top_k))
    passages = [
        {"id": hit.get("id", hit.get("document_id")), "text": hit.get("content", "")}
        for hit in hits
    ]
    if not passages:
        return {
            "query": query, "answer_mode": "azure_openai_grounded",
            "answer": "No relevant passage was returned by the configured search index.",
            "citations": [],
        }
    client = AzureOpenAI(
        azure_endpoint=openai_endpoint,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        azure_ad_token_provider=lambda: _azure_openai_token(credential),
    )
    prompt_context = "\n\n".join(
        f"[{item['id']}] {item['text']}" for item in passages
    )
    completion = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": "Answer only from the provided passages. Cite passage IDs. "
                "If the passages do not support an answer, say so.",
            },
            {"role": "user", "content": f"Passages:\n{prompt_context}\n\nQuestion: {query}"},
        ],
        temperature=0,
    )
    return {
        "query": query,
        "answer_mode": "azure_openai_grounded",
        "answer": completion.choices[0].message.content,
        "citations": [item["id"] for item in passages],
    }


def create_azure_search_index(documents_path: str | Path) -> dict[str, Any]:
    """Create/update an index in an existing AI Search service and upload the local corpus."""
    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    index_name = os.environ.get("AZURE_SEARCH_INDEX")
    if not endpoint or not index_name:
        raise RuntimeError("Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX first.")
    try:
        from azure.identity import DefaultAzureCredential
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.indexes.models import (
            SearchableField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
        )
    except ImportError as exc:
        raise RuntimeError("Install Azure integrations with `pip install -e .[azure]`.") from exc
    documents = _load_documents(Path(documents_path) / "documents.jsonl")
    credential = DefaultAzureCredential()
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    index = SearchIndex(
        name=index_name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
        ],
    )
    index_client.create_or_update_index(index)
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
    results = search_client.upload_documents([
        {"id": doc["id"], "title": doc["title"], "content": doc["text"]}
        for doc in documents
    ])
    failed = [item.key for item in results if not item.succeeded]
    if failed:
        raise RuntimeError(f"AI Search rejected document uploads: {failed}")
    return {"index": index_name, "uploaded_documents": len(results), "service": endpoint}


def _azure_openai_token(credential) -> str:
    return credential.get_token("https://cognitiveservices.azure.com/.default").token
