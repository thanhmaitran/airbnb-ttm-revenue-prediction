# Architecture

## Pipeline overview

```
raw listings (csv / parquet)
   |  prepare_data.py -> src/preprocessing/cleaning.py
   |     drop malformed rows -> coerce numerics -> map booleans -> filter outliers
   v
train / test split (seed 42)
   v
Pipeline  (src/models/pipeline.py :: get_feature_engineered_pipeline)
   |  features    FeatureEngineer     fit: city means + top countries (train rows only)
   |  preprocess  ColumnTransformer   impute + scale / impute + one-hot
   |  regressor   ridge | random_forest | xgboost | hist_gb | mlp
   v
GridSearchCV (configs/model_config.yaml)
   v
models/<model>_pipeline.pkl  (+ <model>_metrics.json)
   |- evaluate.py -> outputs/metrics.csv, outputs/predictions.csv
   |- predict.py  -> JSON prediction from raw records
   '- make_figures.py / src/explainability -> reports/figures, reports/tables
```

## Design principles

**Separation of concerns.** Cleaning (`src/preprocessing/`), feature
engineering (`src/features/`) and model assembly (`src/models/`) are three
independent modules. None of them trains a model; `train.py`/`benchmark.py`
are the only places a `.fit()` call on a full pipeline happens. This keeps
each layer independently testable (see `tests/`).

**No hardcoded paths or hyperparameters.** Every path lives in
`configs/paths.yaml`, every hyperparameter grid in `configs/model_config.yaml`,
every training setting (seed, test split, feature list) in
`configs/training_config.yaml`. `src/utils/config.py` is the only place that
reads these files.

**Train/serve symmetry.** The `FeatureEngineer` is the first step of the saved
`Pipeline`, so `predict.py` simply calls `pipe.predict(raw_records)`: there is
no second implementation of the features. This matters because two features
(`price_vs_city_mean`, `occ_vs_city_mean`) are relative to a city mean. They
use the mean learned in `fit()` on training rows; computing it from the frame
being scored would give exactly 1.0 for any single record.
`tests/test_prediction.py::test_single_record_matches_batch_prediction`
guards this.

**Leakage prevention.** `FeatureEngineer` (city means, top countries) and the
`ColumnTransformer` (imputer/scaler/encoder) are only ever `.fit()` inside
`GridSearchCV` or `cross_val_score`, both of which fit exclusively on the
training fold of each split — see
`tests/test_pipeline.py::test_city_statistics_never_see_test_rows`.
No engineered feature is derived from `ttm_revenue` — see
`tests/test_features.py::test_feature_engineer_does_not_touch_target`.

## Component map

| Layer | Module | Responsibility |
|---|---|---|
| Cleaning | `src/preprocessing/cleaning.py` | type coercion, boolean mapping, outlier filtering |
| Splitting | `src/preprocessing/splitting.py` | train/test split, CV folds |
| Validation | `src/preprocessing/validation.py` | post-cleaning sanity checks |
| Features | `src/features/*.py` | the 11 feature groups + stateful `FeatureEngineer` |
| Explainability | `src/explainability/analysis.py` | permutation importance, partial dependence, error analysis |
| Models | `src/models/*.py` | per-model factories + `create_pipeline` |
| Evaluation | `src/evaluation/metrics.py` | MAE/RMSE/R² on the original USD scale |
| Visualization | `src/visualization/plots.py` | shared publication-style plotting |
| Big data | `src/spark/*.py` | batch/streaming simulation + PySpark job |
| Shared training | `src/training_utils.py` | the routine `train.py` and `benchmark.py` both call |

See also `docs/feature_engineering.md`, `docs/modelling.md`,
`docs/big_data.md` and `docs/reproducibility.md` for the detail behind each
layer.
