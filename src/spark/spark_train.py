"""Distributed Spark ML training job (Section 3.7.2 / 5). Run with:

    spark-submit src/spark/spark_train.py \\
        --input data/raw/data_clean.parquet \\
        --hdfs   # optional: read from hdfs:// instead of local parquet

This mirrors the local scikit-learn pipeline conceptually (feature
engineering -> assemble -> train -> evaluate) but uses Spark MLlib so it
scales to data that doesn't fit on one machine. Spark MLlib supports fewer
model families than scikit-learn/XGBoost (see docs/big_data.md for the
trade-off discussion) so this job trains a single Linear Regression model
as a distributed-scale sanity check, not a replacement for the local
benchmark.
"""
from __future__ import annotations

import argparse


def main(input_path: str, use_hdfs: bool = False) -> None:
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression
    from pyspark.sql import SparkSession

    from src.spark.distributed_features import add_engineered_columns

    spark = SparkSession.builder.appName("Airbnb Big Data Pipeline").getOrCreate()

    path = input_path if not use_hdfs else f"hdfs://{input_path}"
    df = spark.read.parquet(path)
    print(f"Dataset loaded with {df.count()} rows")

    df = add_engineered_columns(df)

    feature_cols = [
        "guests", "bedrooms", "beds", "baths",
        "num_reviews", "ttm_avg_rate", "ttm_occupancy",
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_ml = df.select(feature_cols + ["ttm_revenue"]).dropna()
    df_ml = assembler.transform(df_ml)

    train, test = df_ml.randomSplit([0.8, 0.2], seed=42)
    lr = LinearRegression(featuresCol="features", labelCol="ttm_revenue")
    model = lr.fit(train)
    predictions = model.transform(test)

    evaluator = RegressionEvaluator(
        labelCol="ttm_revenue", predictionCol="prediction", metricName="r2",
    )
    r2 = evaluator.evaluate(predictions)
    print(f"Spark Linear Regression R2: {r2:.4f}")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/data_clean.parquet")
    parser.add_argument("--hdfs", action="store_true")
    args = parser.parse_args()
    main(args.input, args.hdfs)
