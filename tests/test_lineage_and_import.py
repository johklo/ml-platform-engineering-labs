import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from ml_labs.common import verify_dataset
from ml_labs.data import prepare_movies, validate_ratings
from ml_labs.movielens import import_movielens


def test_mutated_dataset_and_different_candidate_are_rejected(tmp_path: Path):
    data = prepare_movies(tmp_path / "movies", users=8, items=12)
    manifest = verify_dataset(data)
    with pytest.raises(ValueError, match="fingerprints differ"):
        verify_dataset(data, {**manifest, "sha256": "different"})
    with (data / "ratings.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="changed after preparation"):
        verify_dataset(data)


def test_movielens_import_preserves_timestamp_groups_and_rating_range(tmp_path: Path):
    rows = [
        {"userId": user, "movieId": item, "rating": 0.5 if item == 0 else 5,
         "timestamp": 1700000000 + item // 2}
        for user in (1, 2) for item in range(20)
    ]
    archive = tmp_path / "movies.zip"
    with ZipFile(archive, "w") as source:
        source.writestr("ml-latest-small/ratings.csv", pd.DataFrame(rows).to_csv(index=False))
        source.writestr("ml-latest-small/movies.csv", pd.DataFrame({
            "movieId": range(20), "title": [f"Film {i}" for i in range(20)],
        }).to_csv(index=False))
    result = import_movielens(archive, tmp_path / "imported")
    frame = pd.read_csv(result / "ratings.csv")
    validate_ratings(frame)
    assert frame.rating.min() == 1
    assert frame.rating.max() == 5
    assert frame.groupby(["user_id", "timestamp"]).split.nunique().max() == 1
    assert verify_dataset(result)["dataset"] == "movielens-latest-small"
    with pytest.raises(FileExistsError):
        import_movielens(archive, result)
    assert json.loads((result / "manifest.json").read_text())["archive_sha256"]
