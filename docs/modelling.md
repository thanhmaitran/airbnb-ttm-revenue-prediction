# Modelling

Five regressors are available (four were benchmarked in the committed results) and trained on identical inputs (same `FeatureEngineer`
output, same `ColumnTransformer`, same train/test split) so their scores are
directly comparable.

| Model | Why used | Grid searched (configs/model_config.yaml) | Module |
|---|---|---|---|
| Ridge | Linear, interpretable baseline | alpha in {0.1, 1, 10, 100, 500}, 5-fold | `src/models/ridge.py` |
| Random Forest | Bagging ensemble | 200 trees, depth 10, min leaf 2 (one configuration), 3-fold | `src/models/random_forest.py` |
| XGBoost | Regularised gradient boosting, optional GPU | lr / depth / n_estimators grid, 3-fold (**not run** in the committed results) | `src/models/xgboost_model.py` |
| HistGradientBoosting | scikit-learn boosted trees; needs no extra dependency | lr {0.05, 0.1} x leaves {31, 63}, 3-fold | `src/models/hist_gb.py` |
| MLP | Neural baseline | (128, 64), early stopping, 3-fold | `src/models/mlp.py` |

The grids are deliberately small (they are the grids that were run); put a
larger grid in `configs/model_config.yaml` to search more - no code change.

## Target transform

The target (`ttm_revenue`) is right-skewed, so training happens on
`log1p(ttm_revenue)` (`configs/training_config.yaml: log_transform_target`).
`src/evaluation/metrics.py::evaluate_predictions` applies `expm1` before
computing MAE/RMSE, so reported errors are in USD, and computes R² directly
on the log scale (matching the CV scoring metric).

## GPU support

`src/models/xgboost_model.py::detect_device()` probes for CUDA once (fitting
a 2-estimator model on a tiny array) and falls back to CPU automatically —
`configs/model_config.yaml: xgboost.device: "auto"` controls this. `xgboost`
itself is imported lazily inside the module's functions rather than at
import time, so importing `src.models.pipeline` (and therefore training
`ridge`/`random_forest`/`mlp`) never requires xgboost to be installed —
only actually training an XGBoost model does.

## Model selection

`train.py` trains and saves one model. `benchmark.py` trains all four on the
identical split (`src/training_utils.load_training_frame`) and writes a
comparison table (`reports/tables/benchmark_results.csv`) ranked by test R².
Both write the fitted pipeline to `models/<model>_pipeline.pkl` and a
metrics JSON alongside it — the file `predict.py` and `evaluate.py` load.

## Reproducing the headline result

```bash
python benchmark.py --models ridge hist_gb random_forest mlp
python evaluate.py --model hist_gb
```

reproduces `reports/tables/benchmark_results.csv` (HistGradientBoosting
R² 0.9913, MAE $906). `evaluate.py` re-scores a saved pipeline on the same
held-out rows (same seed => same split) and matches the benchmark exactly.

## Ablation

`--feature-set no_performance` removes realised price/occupancy and everything
derived from them; `--feature-set pre_booking` additionally removes
reviews/ratings. Results and interpretation: `reports/technical_report.md`,
Section 5.
