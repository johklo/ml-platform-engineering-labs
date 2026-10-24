import base64
from pathlib import Path

import mlflow
import pandas as pd
from fastapi.testclient import TestClient

from ml_labs.api import create_app
from ml_labs.cards import evaluate_cards, train_cards
from ml_labs.common import read_json
from ml_labs.data import prepare_cards, prepare_movies
from ml_labs.movies import evaluate_movies, train_movies
from ml_labs.rag import evaluate_retrieval, index_documents, prepare_documents
from ml_labs.registry import register_candidate


def test_local_train_evaluate_register_reload_and_http(tmp_path: Path) -> None:
    work = tmp_path / "work"
    cards_data = prepare_cards(tmp_path / "cards", train=18, validation=8, test=8)
    movies_data = prepare_movies(tmp_path / "movies", users=12, items=14)
    documents = prepare_documents(tmp_path / "documents")

    cards = train_cards(cards_data, work / "cards", work=work / "tracking")
    movies = train_movies(movies_data, work / "movies", work=work / "tracking")
    rag = index_documents(documents, work / "rag", work=work / "tracking")
    cards_report = work / "cards-test.json"
    movies_report = work / "movies-test.json"
    rag_report = work / "rag-test.json"
    evaluate_cards(cards, cards_data, cards_report, min_iou=0.6)
    evaluate_movies(movies, movies_data, movies_report)
    evaluate_retrieval(rag, rag_report)
    reports = {"cards": cards_report, "movies": movies_report, "rag": rag_report}
    for task, candidate in (("cards", cards), ("movies", movies), ("rag", rag)):
        register_candidate(candidate, f"test-{task}", reports[task])

    app = create_app(cards, movies, rag)
    client = TestClient(app)
    image = next((cards_data / "images" / "test").glob("*.png"))
    metadata = read_json(cards)
    mlflow.set_tracking_uri(metadata["tracking_uri"])
    packaged = mlflow.pyfunc.load_model(metadata["registered_uri"])
    byte_request = pd.DataFrame({
        "image_base64": [base64.b64encode(image.read_bytes()).decode("ascii")],
    })
    assert len(packaged.predict(byte_request)) == 1
    assert client.get("/health").json()["models"] == {
        "cards": True, "movies": True, "rag": True,
    }
    assert client.post("/predict/cards", json={"image_path": str(image)}).status_code == 200
    recommendation = client.post("/recommend", json={"user_id": "unknown"}).json()
    assert recommendation["status"] == "cold_start_popularity"
    answer = client.post("/rag/query", json={"query": "What is Feast?"}).json()
    assert answer["answer_mode"] == "extractive_offline"
    assert client.get("/metrics").status_code == 200
