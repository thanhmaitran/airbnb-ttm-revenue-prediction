# Project Inventory

## Datasets
| File | Rows | Notes |
|---|---|---|
| `listings.csv` / `listings.parquet` (raw, not committed) | 95,898 x 61 | CSV export has ~570 column-shifted rows |
| `past_rates.csv` / `past_rates.parquet` (raw, not committed) | 1,115,174 | monthly rates, Feb 2025 - Jan 2026 |
| `data/processed/data_clean.csv` (generated, not committed) | 92,937 | `python prepare_data.py` |
| `data/sample/sample_listings.csv` | 3,000 | committed; for trying the code only |

## Code
`train.py`, `evaluate.py`, `predict.py`, `benchmark.py`, `prepare_data.py`,
`make_figures.py`; library code under `src/` (preprocessing, features, models,
evaluation, explainability, visualization, spark, utils); configs under
`configs/`; tests under `tests/` (35 tests).

## Outputs (committed)
* `outputs/` - `metrics.csv`, `predictions.csv` (best model, test rows),
  `feature_importance.csv`, `cv_scores.csv`
* `reports/tables/` - benchmark (full / no_performance / pre_booking),
  dataset summary, EDA stats, permutation importance, PDP curves, error
  analysis, batch / streaming / memory / CPU tables
* `reports/figures/` - 19 PNG figures (`fig01`-`fig19`)
* `models/*_metrics.json` - per-model metrics; fitted `*.pkl` are regenerated
  (`python train.py --model hist_gb`, ~1 min) and not committed

## Reproducibility checklist
| Item | Value |
|---|---|
| Python used for results | 3.12 (CI: 3.10, 3.11) |
| Key versions | scikit-learn 1.8, pandas 3.0, numpy 2.4 |
| Seed | 42 (split, estimators, global) |
| Hardware | 1 CPU core, 3 GB RAM, no GPU |
| Total compute for all committed results | ~25 min |
| Not executed there | XGBoost, SHAP, PySpark, multi-core timing |
