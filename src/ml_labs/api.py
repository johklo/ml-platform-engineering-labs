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
from ml_labs.features import FeaturePolicy, FeatureUnavailable, OnlineMovieFeatures

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
               rag: str | Path | None = None, feast_repo: str | Path | None = None,
               feature_policy: FeaturePolicy | None = None,
               movie_release: str | Path | None = None,
               rag_release: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="ML Platform Engineering Labs", version="0.1.0")
    release = None
    if movie_release:
        from ml_labs.releases import read_release

        if movies:
            raise ValueError("Select a movie candidate or a release store, not both.")
        release = read_release(movie_release)
        candidate = release["candidate"]
        if candidate["task"] != "movies":
            raise ValueError("Movie release points at a different task.")
        mlflow.set_tracking_uri(candidate["tracking_uri"])
        movie_model = mlflow.pyfunc.load_model(candidate["registered_uri"])
    else:
        candidate = read_json(movies) if movies else None
        movie_model = _load_candidate(movies, "movies") if movies else None
    feature_source = None
    if feast_repo:
        if candidate is None:
            raise ValueError("Feast serving requires a movie model.")
        if candidate.get("serving_contract") != "movie-routing-v2":
            raise ValueError("Retrain this movie model with the feature-routing serving contract.")
        feature_source = OnlineMovieFeatures(
            feast_repo, candidate["dataset"]["sha256"], feature_policy or FeaturePolicy(),
        )
    if rag and rag_release:
        raise ValueError("Select a RAG candidate or a release store, not both.")
    models = {
        "cards": _load_candidate(cards, "cards") if cards else None,
        "movies": movie_model,
        "rag": _load_candidate(rag, "rag") if rag else None,
    }

    @app.get("/health")
    def health():
        return {
            "status": "ok", "models": {key: value is not None for key, value in models.items()},
            "movie_release_id": release["release_id"] if release else None,
            "feast_enabled": feature_source is not None,
            "rag_release_configured": rag_release is not None,
        }

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
            routing = None
            try:
                routing = feature_source.route(request.user_id) if feature_source else None
            except FeatureUnavailable as exc:
                REQUESTS.labels("movies", "feature_unavailable").inc()
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            frame = {"user_id": [request.user_id]}
            if routing:
                frame["strategy"] = [routing["strategy"]]
            result = models["movies"].predict(pd.DataFrame(frame))
        REQUESTS.labels("movies", "success").inc()
        row = result.iloc[0]
        return {
            "user_id": row["user_id"],
            "status": row["status"],
            "recommendations": json.loads(row["recommendations"]),
            "feature_routing": routing,
            "model_uri": (candidate.get("registered_uri", candidate["model_uri"])
                          if candidate else None),
            "release_id": release["release_id"] if release else None,
        }

    @app.post("/rag/query")
    def rag_query(request: RAGRequest):
        if rag_release:
            from ml_labs.rag_lifecycle import query_release

            with LATENCY.labels("rag").time():
                result = query_release(rag_release, request.query)
            REQUESTS.labels("rag", "success").inc()
            return result
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
