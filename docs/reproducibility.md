# Reproducibility

## Environment

- **Python:** 3.10 or 3.11 (see `.github/workflows/ci.yml`, tested on both)
- **Install:** `pip install -r requirements.txt`, or `conda env create -f environment.yml`
- **CUDA:** optional — `src/models/xgboost_model.py::detect_device()` auto-detects
  GPU availability and falls back to CPU; nothing else in the pipeline requires it.

## Determinism

A single seed (`configs/training_config.yaml: seed`, default `42`) drives
every source of randomness:

- `src/utils/seed.py::set_global_seed` seeds Python's `random`, NumPy, and
  `PYTHONHASHSEED` — called once at the start of `train.py`/`benchmark.py`.
- Every estimator (`Ridge`, `RandomForestRegressor`, `XGBRegressor`,
  `MLPRegressor`) is constructed with `random_state=seed`
  (`src/models/pipeline.py::create_pipeline`).
- `make_train_test_split` (`src/preprocessing/splitting.py`) takes the same
  `random_state`, so the same seed always produces the same train/test rows
  — this is what lets `evaluate.py` re-score a model saved by `train.py` on
  the exact same held-out set without saving the split itself.

## Runtime (measured: 1 CPU core, 3 GB RAM)

| Model | Grid-search wall-clock |
|---|---|
| Ridge | 20 s |
| HistGradientBoosting | 67 s |
| MLP | 229 s |
| Random Forest | 430 s |

(`reports/tables/benchmark_results.csv`; excludes the extra 5-fold CV pass.)
Runtimes scale with the grid in `configs/model_config.yaml`; `--cv N` overrides
the fold count.

## Data

Raw files are not committed (`.gitignore`). A 3,000-row sample is in
`data/sample/`. To reproduce the full results:

1. Put `listings.csv` (or `.parquet`) and `past_rates.csv` under `data/raw/`.
2. `python prepare_data.py --input data/raw/listings.csv --output data/processed/data_clean.csv`
   (92,937 rows from the CSV export; the CSV contains ~570 column-shifted rows
   that are dropped).
3. `export AIRBNB_PROCESSED_DATASET=data/processed/data_clean.csv` and run the
   commands in the README. Any key in `configs/paths.yaml` can be overridden
   with `AIRBNB_<KEY>`.

## What's checked in CI (`.github/workflows/ci.yml`)

- `ruff check .` — linting
- `pytest -v --cov=src` — the full test suite (`tests/`), including the
  no-target-leakage assertion, the fit-on-train-only checks for the
  `FeatureEngineer`, and the single-record-equals-batch check for `predict.py`
- Config files (`configs/*.yaml`) load without error, on Python 3.10 and 3.11

## Known limitations affecting interpretation

- Realised occupancy and rate dominate the model; without them R² falls from
  0.99 to about 0.5 (`reports/technical_report.md`, Section 5).
- XGBoost, SHAP, PySpark and the multi-core timing were not executed in the
  environment that produced the committed results.
