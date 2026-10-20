# Airbnb TTM Revenue Prediction

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-brightgreen)](https://xgboost.readthedocs.io/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5%2B-orange)](https://spark.apache.org/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A leak-free, config-driven machine-learning pipeline that predicts an Airbnb
listing's **trailing-twelve-month (TTM) revenue** from 92,937 listings in 36
countries, plus a big-data track (batch processing, streaming simulation,
CPU/GPU parallelism, PySpark) and an honest ablation of what the model
actually relies on.

![Model comparison](reports/figures/fig09_model_comparison.png)

## Headline results

| | |
|---|---|
| Best model (test set, 18,588 listings) | **HistGradientBoosting - R² 0.991, MAE $906, RMSE $1,819** |
| Random Forest / MLP / Ridge | R² 0.991 / 0.990 / 0.893 |
| Same model **without** realised price & occupancy | **R² 0.51** (MAE $7,561) |
| Same model with **pre-booking information only** | **R² 0.36** (MAE $8,872) |

**Read the last two rows before trusting the first.** Almost all of the
signal comes from two columns the model is given: realised occupancy and
realised average nightly rate. Revenue is, almost by definition, rate x
booked nights, so the 0.99 R² is an *estimation of revenue from observed
performance*, not a forecast for a new listing. See
[Ablation](#what-the-model-actually-relies-on) and
[`reports/technical_report.md`](reports/technical_report.md).

## Quick start (no dataset needed)

A 3,000-row sample is included in `data/sample/`.

```bash
pip install -r requirements.txt
export AIRBNB_PROCESSED_DATASET=data/sample/sample_listings.csv   # point the pipeline at the sample
python train.py --model ridge --cv 3
python predict.py examples/listing.json --model ridge
# {"prediction": 18162.5}   (a ridge model trained on the 3,000-row sample)
```

> The sample is for trying the code only (its metrics are not comparable to the
> full-data results below). Check the data provider's terms before
> redistributing it publicly.

## Reproduce the full results

The raw export is the `listings.csv` / `past_rates.csv` (or `.parquet`) pair.
Put them under `data/raw/` and run:

```bash
python prepare_data.py --input data/raw/listings.csv --output data/processed/data_clean.csv
export AIRBNB_PROCESSED_DATASET=data/processed/data_clean.csv
export AIRBNB_RAW_PAST_RATES=data/raw/past_rates.csv

python benchmark.py --models ridge hist_gb random_forest mlp    # + xgboost if installed
python benchmark.py --models ridge hist_gb --feature-set no_performance
python benchmark.py --models ridge hist_gb --feature-set pre_booking
for m in ridge hist_gb random_forest mlp; do python evaluate.py --model $m; done
python make_figures.py --stage all
```

Every path can be overridden with an `AIRBNB_<KEY>` environment variable
(keys in `configs/paths.yaml`), so nothing is hard-coded. `prepare_data.py`
accepts `.csv` or `.parquet`.

## Pipeline

```
raw listings (csv / parquet)
   |  prepare_data.py          drop malformed rows, coerce types, filter outliers
   v
train / test split (seed 42)   18,588 held-out listings
   |
   v   Pipeline ------------------------------------------------------------
   |   FeatureEngineer        11 feature groups; city means + top countries
   |                          learned from TRAINING rows only (fit)
   |   ColumnTransformer      median-impute + scale / mode-impute + one-hot
   |   Regressor              Ridge | RandomForest | XGBoost | HistGB | MLP
   |   -----------------------------------------------------------------
   v
GridSearchCV (configs/model_config.yaml) -> models/<model>_pipeline.pkl
   |-> evaluate.py -> outputs/metrics.csv, outputs/predictions.csv
   '-> predict.py  -> {"prediction": ...}   (same fitted pipeline, raw JSON in)
```

Because the `FeatureEngineer` sits **inside** the `Pipeline`, it is re-fit on
each CV training fold, and the saved pickle scores raw records directly - there
is no second feature code path to drift from training. A regression test
(`tests/test_prediction.py`) checks that scoring a record alone equals scoring
it inside a batch. Details: [`docs/architecture.md`](docs/architecture.md).

## Results

All numbers below come from the committed tables in `reports/tables/` and were
produced on **one CPU core, 3 GB RAM**.

| Model | MAE | RMSE | R² test | R² CV (5-fold) | R² train | Grid-search time |
|---|---|---|---|---|---|---|
| **HistGradientBoosting** | **$906** | **$1,819** | **0.9913** | 0.9913 ± 0.0004 | 0.9923 | 67 s |
| Random Forest | $963 | $1,866 | 0.9907 | 0.9905 ± 0.0005 | 0.9928 | 430 s |
| MLP | $1,096 | $2,217 | 0.9902 | 0.9899 ± 0.0009 | 0.9914 | 229 s |
| Ridge | $4,392 | $9,685 | 0.8929 | 0.8916 ± 0.0011 | 0.8922 | 20 s |

Train/test/CV scores agree closely for every model (no overfitting signal).
Hyper-parameter grids are the small grids defined in
[`configs/model_config.yaml`](configs/model_config.yaml); larger grids need no code change.

> **XGBoost:** the XGBoost implementation (`--model xgboost`, optional GPU) is
> included and unit-tested, but it was **not executed** in the environment that
> produced these results (the package was unavailable). `HistGradientBoosting`
> is a different implementation of the same model family and is always reported
> under its own name. The original coursework run reported XGBoost at R² 0.990,
> MAE $961 - consistent with the numbers above but not reproduced here.

<table><tr>
<td><img src="reports/figures/fig10_best_model_pred_vs_actual.png"></td>
</tr></table>

### What the model actually relies on

![Ablation](reports/figures/fig17_ablation.png)

| Feature set | Features | HistGB R² | HistGB MAE | Ridge R² |
|---|---|---|---|---|
| Full | 29 | 0.991 | $906 | 0.893 |
| No realised price/occupancy | 21 | 0.506 | $7,561 | 0.443 |
| Pre-booking only (also no reviews/ratings) | 15 | 0.364 | $8,872 | 0.299 |

Permutation importance agrees: shuffling occupancy drops test R² by 1.43 and
shuffling nightly rate by 0.60; **every other input costs under 0.004**.
In Random Forest, `revpar_proxy` (occupancy x rate) carries 99.7% of impurity
importance.

![Permutation importance](reports/figures/fig08_permutation_importance.png)

Partial dependence (what-if curves, recomputing all derived features):

![Partial dependence](reports/figures/fig18_partial_dependence.png)

Error analysis - relative error is ~5% for the cheapest listings and ~3% for
the most expensive; the hardest markets are Croatia (7.1% median error) and
Turkey (5.8%), the easiest the UK (2.6%) and Germany (3.0%):

![Error analysis](reports/figures/fig19_error_analysis.png)

## Dataset

| | |
|---|---|
| Raw listings / columns | 95,898 / 61 |
| After cleaning | 92,937 |
| Countries / cities | 36 / 373 |
| TTM revenue median / mean | $11,647 / $16,865 |
| Mean occupancy / nightly rate | 31.2% / $145.18 |
| Monthly rate history | 1,115,174 rows |

Cleaning drops ~570 column-shifted rows present in the CSV export, then caps the
top 1% of revenue and rate and constrains occupancy to [0, 1]
(`src/preprocessing/cleaning.py`). The original coursework run, which read the
parquet export, retained 93,423 listings; the 0.5% gap is attributable to those
CSV rows.

![Revenue distribution](reports/figures/fig01_revenue_distribution.png)

## Feature engineering

Eleven groups, none of which reads the target (enforced by a test):

| Group | Features |
|---|---|
| Composite rating | `rating_avg`, `rating_std` |
| Capacity ratios | `bed_per_guest`, `bath_per_guest` |
| Log transforms | `log_avg_rate`, `log_reviews`, `log_cleaning` |
| Price ratio | `cleaning_fee_ratio` |
| Interactions | `rate_x_reviews`, `revpar_proxy`, `size_x_rating` |
| Market-relative | `price_vs_city_mean`, `occ_vs_city_mean` |
| Country grouping | `country_grp` |

Formulas and rationale: [`docs/feature_engineering.md`](docs/feature_engineering.md).

## Big-data track

| Technique | Where | Outcome on this data |
|---|---|---|
| Batch processing (10,000-row chunks) | `src/spark/batch_processing.py` | Batch mean revenue ranges $13.6k-$20.1k: the file is **not randomly ordered**, so chunks are not exchangeable ([fig12](reports/figures/fig12_batch_processing.png)) |
| Streaming simulation (1.1M rows, month by month) | `src/spark/stream_processing.py` | Clear July-August peak (~$260M, occupancy ~0.50) vs January low ([fig13](reports/figures/fig13_streaming_trend.png)) |
| CPU parallelism | `src/spark/cpu_parallelism.py` | Committed timings are from the original 8-core run; re-measure with `python -m src.spark.cpu_parallelism` ([fig14](reports/figures/fig14_cpu_parallelism.png)) |
| float64 -> float32 | `reports/tables/memory_efficiency.csv` | 158.5 -> 141.5 MB (-10.7%) |
| PySpark job | `src/spark/spark_train.py` | Implemented; **not executed** in the build environment (no Spark runtime) |

Details and the GPU strategy: [`docs/big_data.md`](docs/big_data.md).

![Streaming](reports/figures/fig13_streaming_trend.png)

## Project structure

```
configs/       paths, training settings, hyper-parameter grids
src/           preprocessing/ features/ models/ evaluation/ explainability/
               visualization/ spark/ utils/ training_utils.py
tests/         pytest suite (cleaning, features, pipeline, prediction, io)
notebooks/     EDA, features, benchmark, SHAP, big data
reports/       figures/ (19 PNG), tables/ (CSV/JSON), technical_report.md
outputs/       metrics.csv, predictions.csv, feature_importance.csv, cv_scores.csv
docs/          architecture, feature_engineering, modelling, big_data, reproducibility
data/sample/   3,000-row example
train.py evaluate.py predict.py benchmark.py prepare_data.py make_figures.py
```

## Testing

```bash
pytest -v --cov=src
```

Covers cleaning (incl. malformed rows), feature engineering (no target
leakage; statistics learned at `fit`; unseen cities), the full pipeline,
train/serve consistency of `predict.py`, and IO/config overrides. CI runs it
on Python 3.10 and 3.11.

## Limitations and future work

- Realised occupancy and rate dominate; for new listings use the pre-booking
  set (R² 0.36) - or add genuinely pre-booking signals: amenities text,
  photos, listing descriptions.
- XGBoost, SHAP and PySpark code paths are included but were not executed
  where the results were produced; run them on your machine
  (`notebooks/04_SHAP.ipynb`, `src/spark/spark_train.py`).
- Temporal modelling over the 1.1M-row rate history, ensemble stacking, and
  distributed training beyond a single node.

## License

[MIT](LICENSE)
