"""Group D (price ratio) and Group E (interaction terms).

revpar_proxy in particular is the single most important predictor in both
the Random Forest and XGBoost models (see reports/figures/figure7 and
figure8) — it mirrors the hotel-industry RevPAR = rate x occupancy KPI.
"""
from __future__ import annotations

import pandas as pd


def create_price_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cleaning_fee_ratio"] = df["cleaning_fee"].fillna(0) / (df["ttm_avg_rate"].fillna(100) + 1e-9)
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # log_avg_rate / log_reviews must already exist (see log_features.py)
    if "log_avg_rate" not in df.columns or "log_reviews" not in df.columns:
        raise KeyError(
            "create_interaction_features requires log_avg_rate/log_reviews; "
            "run create_log_features first (FeatureEngineer.transform handles "
            "ordering automatically)."
        )
    df["rate_x_reviews"] = df["log_avg_rate"] * df["log_reviews"]
    df["revpar_proxy"] = df["ttm_occupancy"] * df["ttm_avg_rate"]
    df["size_x_rating"] = df["bedrooms"].fillna(1) * df["rating_avg"].fillna(4.5)
    return df
