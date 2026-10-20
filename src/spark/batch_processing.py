"""Batch processing (Section 5.1): simulates a data-lake architecture where
listings arrive as partitioned Parquet files by chunking the dataset into
fixed-size batches and computing per-batch summary statistics.

This does NOT require a Spark cluster — it is a pandas-only demonstration
of chunked/out-of-core processing. See src/spark/spark_train.py for the
actual PySpark job.
"""
from __future__ import annotations

import time

import pandas as pd


def run_batch_processing(df: pd.DataFrame, chunk_size: int = 10_000) -> pd.DataFrame:
    """Process `df` in `chunk_size`-row batches and return one summary row
    per batch (rows, mean revenue/occupancy/rate, median reviews).
    """
    for col in ("ttm_avg_rate", "ttm_revenue", "ttm_occupancy", "num_reviews"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    batch_results = []
    t0 = time.time()
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        batch_results.append({
            "batch": i // chunk_size,
            "rows": len(chunk),
            "mean_revenue": chunk["ttm_revenue"].mean(),
            "mean_occupancy": chunk["ttm_occupancy"].mean(),
            "mean_rate": chunk["ttm_avg_rate"].mean(),
            "median_reviews": chunk["num_reviews"].median(),
        })
    elapsed = time.time() - t0
    result = pd.DataFrame(batch_results)
    result.attrs["elapsed_seconds"] = elapsed
    return result
