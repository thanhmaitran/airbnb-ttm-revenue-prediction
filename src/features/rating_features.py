"""Group A — Composite Rating features.

rating_avg / rating_std summarise the 6 review-rating dimensions into a
holistic quality signal; std captures consistency (inconsistent hosts tend
to lose repeat guests).
"""
from __future__ import annotations

import pandas as pd

RATING_COLUMNS = [
    "rating_accuracy", "rating_checkin", "rating_cleanliness",
    "rating_communication", "rating_location", "rating_value",
]


def create_rating_features(df: pd.DataFrame, rating_cols=RATING_COLUMNS) -> pd.DataFrame:
    df = df.copy()
    df["rating_avg"] = df[rating_cols].mean(axis=1)
    df["rating_std"] = df[rating_cols].std(axis=1)
    return df
