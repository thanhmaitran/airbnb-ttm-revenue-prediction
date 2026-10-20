"""Post-cleaning data validation / sanity checks.

Used both as a standalone report (validate_dataset) and inside tests to
guard against silent data-quality regressions.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def missing_value_report(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum().sort_values(ascending=False)


def dtype_report(df: pd.DataFrame) -> pd.Series:
    return df.dtypes


def revenue_range_report(df: pd.DataFrame, revenue_col: str = "ttm_revenue") -> Dict[str, Any]:
    return {
        "min": float(df[revenue_col].min()),
        "max": float(df[revenue_col].max()),
        "mean": float(df[revenue_col].mean()),
        "median": float(df[revenue_col].median()),
        "std": float(df[revenue_col].std()),
    }


def validate_dataset(df: pd.DataFrame, revenue_col: str = "ttm_revenue") -> Dict[str, Any]:
    """Run all checks and return a single summary dict — used by
    `python -m src.preprocessing.validation` and by CI.
    """
    assert revenue_col in df.columns, f"Missing target column '{revenue_col}'"
    assert (df[revenue_col] > 0).all(), "Target contains non-positive values after cleaning"
    if "ttm_occupancy" in df.columns:
        assert df["ttm_occupancy"].between(0, 1).all(), "Occupancy outside [0, 1]"

    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "missing_values": missing_value_report(df).head(15).to_dict(),
        "revenue": revenue_range_report(df, revenue_col),
    }
