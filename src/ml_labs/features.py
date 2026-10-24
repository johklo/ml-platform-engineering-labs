"""Read actual online Feast values and make explicit serving decisions."""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml_labs.common import read_json
from ml_labs.integrations import FEAST_FEATURES, FEAST_LINEAGE, FEAST_TABLE, _sha256


class FeatureUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class FeaturePolicy:
    min_ratings: int = 10
    max_age_seconds: int = 30 * 86400
    unavailable: str = "reject"
    as_of: str | None = None

    def __post_init__(self):
        if self.min_ratings < 1 or self.max_age_seconds < 1:
            raise ValueError("Feature count and age limits must be positive.")
        if self.unavailable not in {"reject", "popularity"}:
            raise ValueError("Feature unavailable policy must be reject or popularity.")
        if self.as_of is not None and pd.Timestamp(self.as_of).tzinfo is None:
            raise ValueError("Feature as-of must include a timezone.")


class OnlineMovieFeatures:
    def __init__(self, repo: str | Path, dataset_sha256: str, policy: FeaturePolicy):
        from feast import FeatureStore

        self.repo = Path(repo).resolve()
        lineage = read_json(self.repo / FEAST_LINEAGE)
        if lineage["dataset"]["sha256"] != dataset_sha256:
            raise ValueError("Model and Feast dataset fingerprints differ.")
        if _sha256(self.repo / FEAST_TABLE) != lineage["feature_table_sha256"]:
            raise ValueError("Feast feature table changed after preparation.")
        self.store = FeatureStore(repo_path=str(self.repo))
        schema = {field.name for field in self.store.get_feature_view("user_movie_stats").schema}
        if not {"ratings_count", "average_rating", "observed_at_unix"}.issubset(schema):
            raise ValueError(
                "Feast schema predates the serving contract. Initialize and apply a new feature repo."
            )
        self.policy = policy
        self.dataset_sha256 = dataset_sha256

    def route(self, user_id: str) -> dict:
        response = self.store.get_online_features(
            features=FEAST_FEATURES, entity_rows=[{"user_id": user_id}],
        ).to_dict()
        values = {key: response[key][0] for key in (
            "average_rating", "ratings_count", "observed_at_unix",
        )}
        reason = None
        if any(value is None or (isinstance(value, float) and not math.isfinite(value))
               for value in values.values()):
            reason = "missing_features"
        else:
            now = (pd.Timestamp(self.policy.as_of).timestamp() if self.policy.as_of else
                   datetime.now(timezone.utc).timestamp())
            age = now - int(values["observed_at_unix"])
            if age < 0:
                reason = "future_features"
            elif age > self.policy.max_age_seconds:
                reason = "stale_features"
            values["age_seconds"] = age
            if not (1 <= values["average_rating"] <= 5 and values["ratings_count"] >= 1):
                reason = "invalid_features"
        if reason and self.policy.unavailable == "reject":
            raise FeatureUnavailable(f"{reason}: online features for {user_id!r} are unusable.")
        strategy = ("popularity" if reason or
                    int(values["ratings_count"]) < self.policy.min_ratings else "auto")
        return {
            "strategy": strategy,
            "reason": reason or ("insufficient_history" if strategy == "popularity"
                                 else "sufficient_history"),
            "features": values,
            "dataset_sha256": self.dataset_sha256,
            "clock": self.policy.as_of or "system_utc",
            "degraded": reason is not None,
        }
