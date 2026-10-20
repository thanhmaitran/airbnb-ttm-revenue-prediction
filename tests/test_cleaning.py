"""Tests for src/preprocessing/cleaning.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.cleaning import clean_booleans, convert_numeric, remove_outliers


def test_convert_numeric_coerces_strings_to_nan(raw_listings_df):
    df = raw_listings_df.astype({"guests": object})
    df.loc[0, "guests"] = "not_a_number"
    out = convert_numeric(df, columns=["guests"])
    assert pd.isna(out.loc[0, "guests"])
    assert out["guests"].dtype.kind == "f"


def test_clean_booleans_maps_to_zero_one(raw_listings_df):
    out = clean_booleans(raw_listings_df, columns=["superhost", "professional_management"])
    assert set(out["superhost"].unique()) <= {0.0, 1.0}
    assert out["superhost"].dtype.kind == "f"


def test_remove_outliers_keeps_occupancy_in_bounds(raw_listings_df):
    df = raw_listings_df.copy()
    df.loc[0, "ttm_occupancy"] = 1.5  # inject an out-of-range value
    out = remove_outliers(df)
    assert out["ttm_occupancy"].between(0, 1).all()


def test_remove_outliers_drops_non_positive_revenue(raw_listings_df):
    df = raw_listings_df.copy()
    df.loc[0, "ttm_revenue"] = -10
    out = remove_outliers(df)
    assert (out["ttm_revenue"] > 0).all()


def test_remove_outliers_caps_top_quantile(raw_listings_df):
    df = raw_listings_df.copy()
    df.loc[0, "ttm_revenue"] = df["ttm_revenue"].max() * 100  # extreme outlier
    out = remove_outliers(df, revenue_upper_quantile=0.99)
    assert out["ttm_revenue"].max() < df["ttm_revenue"].max() * 100


def test_clean_booleans_is_case_insensitive():
    df = pd.DataFrame({"superhost": ["true", "False", True, "TRUE", None, "weird"],
                       "professional_management": [1, 0, "true", "false", "t", "no"]})
    out = clean_booleans(df)
    assert out["superhost"].tolist() == [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert out["professional_management"].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_drop_malformed_rows_removes_non_numeric_ids():
    from src.preprocessing.cleaning import drop_malformed_rows

    df = pd.DataFrame({"listing_id": ["123", "Yalikavak Bodrum", '(B2)"', "456"], "x": [1, 2, 3, 4]})
    out = drop_malformed_rows(df)
    assert out["listing_id"].tolist() == ["123", "456"]
