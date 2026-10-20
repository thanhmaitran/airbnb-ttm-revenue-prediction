"""Group F — Market-Relative features, and Group G — country grouping.

price_vs_city_mean / occ_vs_city_mean provide contextual normalisation: the
same absolute price can be a premium or a budget offer depending on the
local market, which is why these rank among the top features.

Both functions accept *pre-fitted* statistics (city means, top countries).
FeatureEngineer.fit() learns them on the training rows only and
FeatureEngineer.transform() passes them in, so (a) nothing about the test
rows leaks into the features and (b) a single record scored by predict.py
is compared against the training-time city mean rather than against itself.
If no statistics are given they are computed from the frame itself (handy
for notebooks and unit tests).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def fit_market_statistics(df: pd.DataFrame) -> dict:
    """City-level mean rate / occupancy, plus global fall-back means."""
    return {
        "city_mean_rate": df.groupby("city")["ttm_avg_rate"].mean(),
        "city_mean_occ": df.groupby("city")["ttm_occupancy"].mean(),
        "global_mean_rate": float(df["ttm_avg_rate"].mean()),
        "global_mean_occ": float(df["ttm_occupancy"].mean()),
    }


def create_market_relative_features(df: pd.DataFrame, stats: Optional[dict] = None) -> pd.DataFrame:
    """Create city-relative pricing and occupancy features.

    Features
    --------
    price_vs_city_mean : ttm_avg_rate / mean ttm_avg_rate of the listing's city
    occ_vs_city_mean   : ttm_occupancy / mean ttm_occupancy of the listing's city

    Cities never seen during fit fall back to the global mean.

    Returns
    -------
    DataFrame
    """
    df = df.copy()
    stats = stats or fit_market_statistics(df)
    city_rate = df["city"].map(stats["city_mean_rate"]).fillna(stats["global_mean_rate"])
    city_occ = df["city"].map(stats["city_mean_occ"]).fillna(stats["global_mean_occ"])
    df["price_vs_city_mean"] = df["ttm_avg_rate"] / (city_rate + 1e-9)
    df["occ_vs_city_mean"] = df["ttm_occupancy"] / (city_occ + 1e-9)
    return df


def fit_top_countries(df: pd.DataFrame, top_n: int = 10) -> list:
    return df["country"].value_counts().head(top_n).index.tolist()


def create_country_grouping(df: pd.DataFrame, top_n: int = 10, top_countries: Optional[list] = None) -> pd.DataFrame:
    """Collapse the long tail of countries into the top-N + 'Other' so the
    one-hot space stays small and unseen countries fall into 'Other'.
    """
    df = df.copy()
    top_countries = top_countries if top_countries is not None else fit_top_countries(df, top_n)
    df["country_grp"] = df["country"].where(df["country"].isin(top_countries), "Other")
    return df
