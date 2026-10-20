#!/usr/bin/env python
"""Data preparation CLI: raw listings -> cleaned dataset + dataset summary.

    python prepare_data.py                       # uses configs/paths.yaml (raw_listings -> processed_dataset)
    python prepare_data.py --input data/raw/listings.csv --output data/processed/data_clean.csv

Applies the documented cleaning (numeric coercion, boolean mapping, outlier
filtering), validates the result, and writes reports/tables/dataset_summary.csv
and reports/tables/eda_stats.json used by the technical report.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.preprocessing.cleaning import clean_pipeline, load_dataset
from src.preprocessing.validation import validate_dataset
from src.utils.config import resolve_path
from src.utils.io import ensure_dir, save_json, write_table


def main(input_path, output_path) -> None:
    raw = load_dataset(input_path)
    print(f"raw: {raw.shape}")
    df = clean_pipeline(input_path)
    print(f"cleaned: {df.shape}")
    report = validate_dataset(df)

    rev = df["ttm_revenue"]
    summary = pd.DataFrame([
        ("Raw listings", len(raw)), ("Raw columns", raw.shape[1]),
        ("Listings after cleaning", len(df)), ("Countries", df["country"].nunique()),
        ("Cities", df["city"].nunique()), ("TTM revenue mean (USD)", round(rev.mean())),
        ("TTM revenue median (USD)", round(rev.median())), ("TTM revenue std (USD)", round(rev.std())),
        ("Mean occupancy", round(df["ttm_occupancy"].mean(), 3)),
        ("Mean nightly rate (USD)", round(df["ttm_avg_rate"].mean(), 2)),
    ], columns=["metric", "value"])
    tables = ensure_dir(resolve_path("table_dir"))
    summary.to_csv(tables / "dataset_summary.csv", index=False)

    corr_cols = ["ttm_revenue", "ttm_occupancy", "ttm_avg_rate", "guests", "bedrooms", "beds", "baths",
                 "num_reviews", "rating_overall", "cleaning_fee", "min_nights"]
    corr = df[corr_cols].dropna().corr()["ttm_revenue"].drop("ttm_revenue").round(3).to_dict()
    stats = {
        "skew_raw": float(rev.skew()), "skew_log1p": float(np.log1p(rev).skew()),
        "corr_with_revenue": corr, "validation": report,
        "country_median_revenue_top12": df.groupby("country")["ttm_revenue"].median()
            .sort_values(ascending=False).head(12).round(0).to_dict(),
    }
    save_json(stats, tables / "eda_stats.json")

    write_table(df, output_path)
    print(summary.to_string(index=False))
    print(f"skew raw={stats['skew_raw']:.2f} log1p={stats['skew_log1p']:.2f}")
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    a = ap.parse_args()
    main(a.input or resolve_path("raw_listings"), a.output or resolve_path("processed_dataset"))
