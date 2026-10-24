from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from ml_labs.common import fingerprint, write_json


def prepare_movies(output, seed=42, users=48, items=32):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    user_factors = rng.normal(size=(users, 4))
    item_factors = rng.normal(size=(items, 4))
    rows = []
    origin = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for user in range(users):
        selected = rng.choice(items, size=min(18, items), replace=False)
        for offset, item in enumerate(selected):
            rating = np.clip(
                3 + user_factors[user] @ item_factors[item] / 2 + rng.normal(0, 0.1), 1, 5
            )
            rows.append({
                "user_id": f"u{user:03}",
                "item_id": f"m{item:03}",
                "rating": round(float(rating), 3),
                "timestamp": (origin + timedelta(days=offset)).isoformat(),
                "split": "train" if offset < len(selected) - 4 else (
                    "validation" if offset < len(selected) - 2 else "test"
                ),
            })
    frame = pd.DataFrame(rows)
    validate_ratings(frame)
    frame.to_csv(output / "ratings.csv", index=False)
    pd.DataFrame({
        "item_id": [f"m{i:03}" for i in range(items)],
        "title": [f"Synthetic Movie {i:03}" for i in range(items)],
    }).to_csv(output / "items.csv", index=False)
    write_json(output / "manifest.json", {
        "dataset": "synthetic-movies", "seed": seed, "rows": len(frame),
        "sha256": fingerprint([output / "ratings.csv", output / "items.csv"]),
        "split": "per-user chronological train / validation / test; no real people",
    })
    return output


def validate_ratings(frame):
    required = {"user_id", "item_id", "rating", "timestamp", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing rating columns: {sorted(required - set(frame.columns))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Ratings contain null values.")
    if frame.duplicated(["user_id", "item_id"]).any():
        raise ValueError("A user/item pair must occur only once.")
    if not frame["rating"].between(1, 5).all():
        raise ValueError("Ratings must be within [1, 5].")
    if not set(frame["split"]).issubset({"train", "validation", "test"}):
        raise ValueError("Unknown rating split.")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    for user, group in frame.assign(event_time=timestamps).groupby("user_id"):
        previous = None
        for split in ("train", "validation", "test"):
            times = group.loc[group["split"] == split, "event_time"]
            if times.empty:
                raise ValueError(f"{user} has no {split} observations.")
            if previous is not None and times.min() <= previous:
                raise ValueError(f"Temporal leakage in {user}/{split}.")
            previous = times.max()


def prepare_cards(output, seed=42, train=80, validation=20, test=20):
    output = Path(output).resolve()
    rng = np.random.default_rng(seed)
    for split, count in (("train", train), ("val", validation), ("test", test)):
        images, labels = output / "images" / split, output / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for number in range(count):
            width = height = 128
            background = rng.integers(10, 90, size=(height, width, 3), dtype=np.uint8)
            image = Image.fromarray(background)
            draw = ImageDraw.Draw(image)
            label = ""
            if number % 5 != 0:
                left, top = int(rng.integers(8, 40)), int(rng.integers(12, 55))
                card_width, card_height = int(rng.integers(55, 80)), int(rng.integers(35, 55))
                right, bottom = min(left + card_width, 123), min(top + card_height, 123)
                color = tuple(int(x) for x in rng.integers(150, 245, size=3))
                draw.rectangle((left, top, right, bottom), fill=color, outline="white", width=2)
                draw.rectangle((left + 5, top + 7, left + 19, bottom - 6), fill=(90, 120, 150))
                for line in range(3):
                    y = top + 9 + line * 7
                    draw.line((left + 25, y, right - 5, y), fill=(50, 60, 70), width=2)
                label = (
                    f"0 {(left + right) / (2 * width):.6f} "
                    f"{(top + bottom) / (2 * height):.6f} "
                    f"{(right - left) / width:.6f} {(bottom - top) / height:.6f}\n"
                )
            name = f"{split}-{number:04}"
            image.save(images / f"{name}.png")
            (labels / f"{name}.txt").write_text(label, encoding="utf-8")
    import yaml

    (output / "dataset.yaml").write_text(yaml.safe_dump({
        "path": str(output), "train": "images/train", "val": "images/val",
        "test": "images/test", "names": {0: "synthetic_id_card"},
    }), encoding="utf-8")
    files = list(output.glob("images/*/*.png")) + list(output.glob("labels/*/*.txt"))
    write_json(output / "manifest.json", {
        "dataset": "synthetic-cards", "seed": seed, "sha256": fingerprint(files),
        "counts": {"train": train, "validation": validation, "test": test},
        "limitations": "Axis-aligned generated cards, independent images, no identity data; not OCR.",
    })
    return output
