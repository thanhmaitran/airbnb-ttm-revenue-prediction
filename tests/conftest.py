"""Shared pytest fixtures — a small synthetic listings frame that mirrors
the real schema (see 01_eda_preprocessing.py / the technical report) so
tests run instantly and never depend on the private dataset being present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_listings_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "guests": rng.integers(1, 8, n).astype(float),
        "bedrooms": rng.integers(1, 5, n).astype(float),
        "beds": rng.integers(1, 6, n).astype(float),
        "baths": rng.integers(1, 3, n).astype(float),
        "min_nights": rng.integers(1, 14, n).astype(float),
        "num_reviews": rng.integers(0, 300, n).astype(float),
        "rating_overall": rng.uniform(3.5, 5.0, n),
        "rating_accuracy": rng.uniform(3.5, 5.0, n),
        "rating_checkin": rng.uniform(3.5, 5.0, n),
        "rating_cleanliness": rng.uniform(3.5, 5.0, n),
        "rating_communication": rng.uniform(3.5, 5.0, n),
        "rating_location": rng.uniform(3.5, 5.0, n),
        "rating_value": rng.uniform(3.5, 5.0, n),
        "cleaning_fee": rng.uniform(0, 100, n),
        "extra_guest_fee": rng.uniform(0, 30, n),
        "photos_count": rng.integers(1, 40, n).astype(float),
        "ttm_avg_rate": rng.uniform(20, 400, n),
        "ttm_occupancy": rng.uniform(0, 1, n),
        "superhost": rng.choice([True, False], n),
        "professional_management": rng.choice([True, False], n),
        "room_type": rng.choice(["Entire home", "Private room", "Shared room"], n),
        "country": rng.choice(["United Kingdom", "Portugal", "Spain", "Iceland", "Malta"], n),
        "city": rng.choice(["London", "Lisbon", "Madrid", "Reykjavik", "Valletta"], n),
    })
    # target correlated with rate * occupancy so the pipeline has real signal to learn
    df["ttm_revenue"] = (
        df["ttm_avg_rate"] * df["ttm_occupancy"] * 300
        + rng.normal(0, 500, n)
    ).clip(lower=100)
    return df
