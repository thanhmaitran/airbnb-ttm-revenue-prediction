"""Loading, type coercion, boolean mapping and outlier filtering.

This module intentionally does NOT touch feature engineering or model
training — see src/features/ and src/models/ respectively. Keeping this
separation makes each step independently testable (see tests/test_cleaning.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

NUMERIC_COLUMNS = [
    "ttm_revenue", "ttm_occupancy", "ttm_avg_rate", "ttm_revpar",
    "l90d_revenue", "l90d_occupancy", "l90d_avg_rate",
    "guests", "bedrooms", "beds", "baths",
    "num_reviews", "rating_overall", "rating_accuracy",
    "rating_checkin", "rating_cleanliness",
    "rating_communication", "rating_location", "rating_value",
    "cleaning_fee", "extra_guest_fee", "min_nights",
    "photos_count", "ttm_reserved_days", "ttm_available_days", "ttm_total_days",
]

BOOLEAN_COLUMNS = ["superhost", "professional_management"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the raw listings table (.parquet or .csv, chosen by extension)."""
    from src.utils.io import read_table

    return read_table(Path(path))


_TRUE = {"true", "1", "yes", "t"}
_FALSE = {"false", "0", "no", "f"}


def to_binary(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Map bool / 'True' / 'true' / 1 / 0 ... to {0.0, 1.0}; unknown -> fill_value.

    Case-insensitive on purpose: the raw CSV export stores 'true'/'false'
    while the parquet export stores real booleans.
    """
    def _map(v):
        if isinstance(v, (bool, int, float)) and v == v:
            return 1.0 if v else 0.0
        text = str(v).strip().lower()
        if text in _TRUE:
            return 1.0
        if text in _FALSE:
            return 0.0
        return fill_value

    return series.map(_map).astype(float)


def drop_malformed_rows(df: pd.DataFrame, id_col: str = "listing_id") -> pd.DataFrame:
    """Drop rows whose id is not numeric.

    A few hundred rows in the raw CSV export are column-shifted (free text
    with embedded line breaks spilled into neighbouring columns). Their id
    column holds text, which is a reliable marker for the corruption.
    """
    if id_col not in df.columns:
        return df
    ok = pd.to_numeric(df[id_col], errors="coerce").notna()
    return df[ok].copy()


def convert_numeric(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_COLUMNS) -> pd.DataFrame:
    """Coerce known-numeric columns from string/object dtype to float, turning
    unparseable values into NaN rather than raising.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_booleans(df: pd.DataFrame, columns: Iterable[str] = BOOLEAN_COLUMNS,
                    fill_value: float = 0.0) -> pd.DataFrame:
    """Map string/bool representations of booleans to {0.0, 1.0}."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = to_binary(df[col], fill_value)
    return df


def remove_outliers(
    df: pd.DataFrame,
    revenue_col: str = "ttm_revenue",
    rate_col: str = "ttm_avg_rate",
    occupancy_col: str = "ttm_occupancy",
    revenue_upper_quantile: float = 0.99,
    rate_upper_quantile: float = 0.99,
    occupancy_range: tuple[float, float] = (0, 1),
) -> pd.DataFrame:
    """Cap the top 1% of revenue and rate, constrain occupancy to [0, 1],
    and drop rows with a missing/non-positive target.

    Matches the filtering described in Section 2.1 of the technical report:
    93,423 of 95,898 listings are retained after this step.
    """
    df = df.copy()
    df = df[df[revenue_col] > 0]
    df = df[df[revenue_col] < df[revenue_col].quantile(revenue_upper_quantile)]
    df = df[df[occupancy_col].between(*occupancy_range)]
    df = df[df[rate_col] > 0]
    df = df[df[rate_col] < df[rate_col].quantile(rate_upper_quantile)]
    df = df.dropna(subset=[revenue_col])
    return df


def clean_pipeline(path: str | Path) -> pd.DataFrame:
    """End-to-end cleaning: load -> convert numeric -> clean booleans -> filter outliers."""
    df = load_dataset(path)
    df = drop_malformed_rows(df)
    df = convert_numeric(df)
    df = clean_booleans(df)
    df = remove_outliers(df)
    return df
