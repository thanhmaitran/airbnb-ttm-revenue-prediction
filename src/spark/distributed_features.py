"""Distributed feature engineering + aggregation in PySpark (Section 5, and
originally 04_pyspark.py). Requires a SparkSession — import lazily so the
rest of the codebase works without PySpark installed.
"""
from __future__ import annotations


def add_engineered_columns(df):
    """df: pyspark.sql.DataFrame with the cleaned listing schema."""
    from pyspark.sql.functions import col, log1p

    df = df.withColumn("log_avg_rate", log1p(col("ttm_avg_rate")))
    df = df.withColumn("log_reviews", log1p(col("num_reviews")))
    df = df.withColumn("bed_per_guest", col("beds") / (col("guests") + 1e-9))

    rating_cols = [
        "rating_accuracy", "rating_checkin", "rating_cleanliness",
        "rating_communication", "rating_location", "rating_value",
    ]
    df = df.withColumn(
        "rating_avg",
        sum(col(c) for c in rating_cols) / len(rating_cols),
    )
    return df


def aggregate_by_country(df):
    from pyspark.sql.functions import mean, stddev

    return df.groupBy("country").agg(
        mean("ttm_revenue").alias("avg_revenue"),
        mean("ttm_occupancy").alias("avg_occupancy"),
        stddev("ttm_revenue").alias("revenue_std"),
    )
