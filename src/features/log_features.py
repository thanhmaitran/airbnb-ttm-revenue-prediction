"""Group C — Log Transforms.

log1p compresses the right-skew present in rate, review-count and fee
distributions, which improves the fit of the linear (Ridge) baseline in
particular.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def create_log_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_avg_rate"] = np.log1p(df["ttm_avg_rate"])
    df["log_reviews"] = np.log1p(df["num_reviews"])
    df["log_cleaning"] = np.log1p(df["cleaning_fee"].fillna(0))
    return df
