# Big Data & Distributed Processing

The core pipeline (`train.py`/`benchmark.py`) runs comfortably on a single
machine for this dataset size, but the codebase includes four separate
techniques that demonstrate how the same approach scales past what fits in
memory on one node — this is what `src/spark/` and the batch/streaming
simulations are for.

## 1. Batch processing (`src/spark/batch_processing.py`)

Simulates a data-lake architecture where listings arrive as partitioned
Parquet files: the dataset is processed in fixed-size (10,000-row) chunks,
computing per-batch summary statistics without ever loading the full
dataset into memory at once. This is a pandas-only demonstration of
chunked/out-of-core processing — it needs no Spark cluster. Comparing batch
means is also a data-quality check, and on this export it **fails**: batch mean
revenue ranges from $13.6k to $20.1k, so the file is not randomly ordered and
chunks are not exchangeable (shuffle or stratify before chunked training).

## 2. Streaming simulation (`src/spark/stream_processing.py`)

Processes the 1.1M-row `past_rates.parquet` time series month by month, as
if records were arriving incrementally from a Kafka / Spark Structured
Streaming source. Each monthly window computes aggregate metrics — this is
what would power a real-time revenue/occupancy dashboard, and reveals the
seasonal summer-peak pattern: total revenue rises from $83M (Feb 2025) to
$261M (Aug 2025) and falls to $58M (Jan 2026).

## 3. CPU multi-core parallelism

`RandomForestRegressor(n_jobs=-1)` (`src/models/random_forest.py`) uses
scikit-learn's joblib-backed parallelism across every available core. The
committed `reports/tables/cpu_parallelism.csv` holds the timings from the
original 8-core run (14.6 s / 4.1 s / 2.6 s for 1 / 4 / 8 jobs); the build
sandbox had one core, so run `python -m src.spark.cpu_parallelism` on a
multi-core machine to replace them with live measurements.
For a training set that doesn't fit on one machine, the equivalent would be
`Ray`/`Dask-ML` partitioning training across nodes, analogous to a parameter
server architecture.

## 4. GPU acceleration (`src/models/xgboost_model.py`)

`device="cuda"` is auto-detected (`detect_device()`); the codebase falls
back to CPU transparently when no GPU is present. On an Oracle VM GPU
instance (e.g. NVIDIA V100/A10), XGBoost's GPU histogram method is commonly
reported to give an order-of-magnitude speed-up. **This project has not
measured it** (no GPU was available); treat the figure as a hypothesis to test
with `python train.py --model xgboost`. A PyTorch MLP or RAPIDS `cuML` Random
Forest would follow the same `.to("cuda")` / `cuDF` pattern.

| Component | CPU implementation | GPU (CUDA) equivalent |
|---|---|---|
| XGBoost | `XGBRegressor()` | `XGBRegressor(device="cuda")` |
| Random Forest | `RandomForestRegressor(n_jobs=-1)` | `cuml.ensemble.RandomForestRegressor` |
| MLP | `MLPRegressor()` (sklearn) | `torch.nn.Sequential(...).to("cuda")` |
| Data pipeline | pandas `DataFrame` | cuDF `DataFrame` (RAPIDS) |

## 5. PySpark distributed pipeline (`src/spark/`)

`distributed_features.py` mirrors the pandas feature engineering (log
transforms, rating aggregation) as Spark DataFrame operations;
`spark_train.py` assembles features with `VectorAssembler` and trains a
Spark MLlib `LinearRegression` model, evaluated with `RegressionEvaluator`.
Run it with:

```bash
spark-submit src/spark/spark_train.py --input data/processed/data_clean.parquet
```

(Not executed in the build environment - no Spark runtime was available.)

Spark MLlib supports fewer model families than scikit-learn/XGBoost, so this
job is a distributed-scale sanity check (comparable to the Ridge baseline),
not a replacement for the main benchmark — the practical workflow is:
prototype locally with pandas/scikit-learn, tune with `GridSearchCV`, then
port the winning approach's feature logic to Spark once data volume actually
requires a cluster.

## Hardware considerations

The full `GridSearchCV` grids in `configs/model_config.yaml` were tuned to
run in a reasonable time on a laptop-class CPU; on a machine without a GPU,
override the CV fold count for a quicker pass:

```bash
python train.py --model xgboost --cv 3
python benchmark.py --cv 3
```
