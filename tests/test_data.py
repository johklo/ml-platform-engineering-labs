from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from ml_labs.cards import detect_color_component, evaluate_card_split
from ml_labs.data import prepare_cards, prepare_movies, validate_ratings


def test_synthetic_card_baseline_detects_boxes_and_blank_images(tmp_path: Path) -> None:
    data = prepare_cards(tmp_path / "cards", seed=19, train=18, validation=10, test=10)

    score = evaluate_card_split(data, "test")

    assert score["images"] == 10
    assert score["detection_rate"] >= 0.7
    assert score["mean_iou"] >= 0.75
    blank = Image.open(data / "images" / "test" / "test-0000.png")
    assert detect_color_component(blank) is None


def test_movie_data_has_valid_per_user_temporal_splits(tmp_path: Path) -> None:
    data = prepare_movies(tmp_path / "movies", users=8, items=12)
    frame = pd.read_csv(data / "ratings.csv")

    validate_ratings(frame)

    assert set(frame["split"]) == {"train", "validation", "test"}
    assert frame["user_id"].nunique() == 8
    assert (data / "manifest.json").is_file()


def test_rating_validator_rejects_duplicate_pairs() -> None:
    frame = pd.DataFrame([
        {"user_id": "u", "item_id": "m", "rating": 4, "timestamp": "2025-01-01T00:00:00Z", "split": "train"},
        {"user_id": "u", "item_id": "m", "rating": 3, "timestamp": "2025-01-02T00:00:00Z", "split": "validation"},
    ])
    with pytest.raises(ValueError, match="only once"):
        validate_ratings(frame)
