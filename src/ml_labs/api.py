"""Small HTTP inference surface for the tracked local models."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from ml_labs.common import read_json

REQUESTS = Counter("ml_labs_inference_requests_total", "Inference requests", ["task", "status"])
LATENCY = Histogram("ml_labs_inference_seconds", "Inference latency", ["task"])


class ImageRequest(BaseModel):
    image_path: str


class RecommendationRequest(BaseModel):
    user_id: str = Field(min_length=1)


class RAGRequest(BaseModel):
    query: str = Field(min_length=1)


def _load_candidate(path: str | Path, task: str):
    candidate = read_json(path)
    if candidate.get("task") != task:
        raise ValueError(f"Candidate {path} is not a {task} model.")
    mlflow.set_tracking_uri(candidate["tracking_uri"])
    return mlflow.pyfunc.load_model(candidate.get("registered_uri", candidate["model_uri"]))


def create_app(cards: str | Path | None = None, movies: str | Path | None = None,
               rag: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="ML Platform Engineering Labs", version="0.1.0")
    models = {
        "cards": _load_candidate(cards, "cards") if cards else None,
        "movies": _load_candidate(movies, "movies") if movies else None,
        "rag": _load_candidate(rag, "rag") if rag else None,
    }

    @app.get("/health")
    def health():
        return {"status": "ok", "models": {key: value is not None for key, value in models.items()}}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict/cards")
    def predict_cards(request: ImageRequest):
        if models["cards"] is None:
            raise HTTPException(status_code=503, detail="Cards model was not configured.")
        with LATENCY.labels("cards").time():
            try:
                result = models["cards"].predict(pd.DataFrame({"image_path": [request.image_path]}))
            except (ValueError, FileNotFoundError) as exc:
                REQUESTS.labels("cards", "error").inc()
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        REQUESTS.labels("cards", "success").inc()
        row = result.iloc[0]
        return {
            "image_path": row["image_path"],
            "detected": bool(row["detected"]),
            "box_xyxy": None if not row["detected"] else [
                float(row["x_min"]), float(row["y_min"]),
                float(row["x_max"]), float(row["y_max"]),
            ],
            "backend": row["backend"],
        }

    @app.post("/recommend")
    def recommend(request: RecommendationRequest):
        if models["movies"] is None:
            raise HTTPException(status_code=503, detail="Movie model was not configured.")
        with LATENCY.labels("movies").time():
            result = models["movies"].predict(pd.DataFrame({"user_id": [request.user_id]}))
        REQUESTS.labels("movies", "success").inc()
        row = result.iloc[0]
        return {
            "user_id": row["user_id"],
            "status": row["status"],
            "recommendations": json.loads(row["recommendations"]),
        }

    @app.post("/rag/query")
    def rag_query(request: RAGRequest):
        if models["rag"] is None:
            raise HTTPException(status_code=503, detail="RAG model was not configured.")
        with LATENCY.labels("rag").time():
            result = models["rag"].predict(pd.DataFrame({"query": [request.query]}))
        REQUESTS.labels("rag", "success").inc()
        row = result.iloc[0]
        return {
            "query": row["query"],
            "answer_mode": row["answer_mode"],
            "answer": row["answer"],
            "citations": json.loads(row["citations"]),
        }

    return app
