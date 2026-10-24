"""Import a user-downloaded MovieLens latest-small archive without redistributing it."""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from ml_labs.common import fingerprint, write_json
from ml_labs.data import validate_ratings


def import_movielens(archive: str | Path, output: str | Path, min_events: int = 10) -> Path:
    if min_events < 5:
        raise ValueError("min_events must be at least 5.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output directory: {output}")
    with ZipFile(archive) as source:
        with source.open("ml-latest-small/ratings.csv") as handle:
            ratings = pd.read_csv(handle)
        with source.open("ml-latest-small/movies.csv") as handle:
            catalog = pd.read_csv(handle)
    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})
    if ratings.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Repeated user/movie events require an explicit deduplication policy.")
    groups = []
    for _, events in ratings.groupby("user_id"):
        if len(events) < min_events:
            continue
        events = events.sort_values("timestamp").copy()
        times = sorted(events.timestamp.unique())
        if len(times) < 3:
            continue
        # Split whole timestamp groups, not rows: simultaneous ratings cannot cross a boundary.
        first = times[min(len(times) - 2, max(1, int(len(times) * 0.8)))]
        second = times[min(len(times) - 1, max(times.index(first) + 1, int(len(times) * 0.9)))]
        events["split"] = "train"
        events.loc[events.timestamp >= first, "split"] = "validation"
        events.loc[events.timestamp >= second, "split"] = "test"
        groups.append(events)
    if len(groups) < 2:
        raise ValueError("Need at least two users with sufficient distinct event timestamps.")
    frame = pd.concat(groups, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, unit="s", utc=True).astype(str)
    frame["user_id"] = "u" + frame.user_id.astype(str)
    frame["item_id"] = "m" + frame.item_id.astype(str)
    # MovieLens permits half-star ratings; map them affinely to the lab's [1, 5] contract.
    frame["rating"] = 1 + (frame.rating - 0.5) * (4 / 4.5)
    validate_ratings(frame)
    catalog = catalog.rename(columns={"movieId": "item_id"})
    catalog["item_id"] = "m" + catalog.item_id.astype(str)
    output.mkdir(parents=True)
    frame.to_csv(output / "ratings.csv", index=False)
    catalog[["item_id", "title"]].to_csv(output / "items.csv", index=False)
    write_json(output / "manifest.json", {
        "dataset": "movielens-latest-small",
        "rows": len(frame),
        "sha256": fingerprint([output / "ratings.csv", output / "items.csv"]),
        "archive_sha256": fingerprint([archive]),
        "source": "https://grouplens.org/datasets/movielens/latest/",
        "rating_transform": "1 + (original_rating - 0.5) * 4 / 4.5",
        "split": "per-user chronological timestamp groups, 80/10/10 of unique timestamps",
        "license": "MovieLens terms apply; archive and derived data must not be committed.",
    })
    return output
