# Feature Engineering

Eleven groups of engineered features, all constructed exclusively from
supply-side and non-target attributes — none of them reads `ttm_revenue`,
which is enforced by `tests/test_features.py::test_feature_engineer_does_not_touch_target`.

| Group | Module | Features | Formula | Business intuition |
|---|---|---|---|---|
| A. Composite Rating | `rating_features.py` | `rating_avg`, `rating_std` | mean / std of 6 rating dimensions | Holistic quality signal; std captures consistency — inconsistent hosts lose repeat guests |
| B. Capacity Ratios | `capacity_features.py` | `bed_per_guest`, `bath_per_guest` | beds / guests, baths / guests | Comfort proxy — under-bedded listings earn less per guest capacity |
| C. Log Transforms | `log_features.py` | `log_avg_rate`, `log_reviews`, `log_cleaning` | `log1p(x)` | Compresses right-skew in rate/review-count distributions, improving the linear baseline's fit |
| D. Price Ratio | `interaction_features.py` | `cleaning_fee_ratio` | cleaning_fee / avg_rate | A high fee relative to nightly rate reduces booking conversion |
| E. Interaction Terms | `interaction_features.py` | `rate_x_reviews`, `revpar_proxy`, `size_x_rating` | log_rate × log_reviews; occupancy × avg_rate; beds × rating_avg | Captures non-additive joint effects; `revpar_proxy` mirrors the hotel-industry RevPAR KPI and is the single most important predictor for both tree models |
| F. Market-Relative | `market_features.py` | `price_vs_city_mean`, `occ_vs_city_mean` | feature / city mean | Contextual normalisation — the same absolute price is a premium or a budget offer depending on the local market |
| G. Country Grouping | `market_features.py` | `country_grp` | top-N countries + 'Other' | Keeps the one-hot categorical space small and generalises to unseen countries at inference time |

## Why this design

**One function per feature group, one file per group.** Each transform is a
pure function `df -> df` with a docstring stating its formula and rationale
(see the module list above). This makes every group independently
unit-testable (`tests/test_features.py`) and easy to extend — adding a
twelfth group means adding one function and one line in
`FeatureEngineer.transform`, not touching the others.

**`FeatureEngineer` is the single entry point.** It is a scikit-learn
`BaseEstimator`/`TransformerMixin`, so it can be dropped directly into a
`Pipeline` (see `get_feature_engineered_pipeline` in
`src/models/pipeline.py`) or called standalone via `.fit_transform(df)`
(what `train.py`/`evaluate.py`/`benchmark.py`/`predict.py` all do).

**Stateful, so train and serve agree.** `FeatureEngineer.fit()` learns two
things from the rows it receives - the per-city mean rate / occupancy (used by
Group F) and the top-N countries (Group G) - and `transform()` reuses them.
Inside `GridSearchCV` this means they come from the training fold only;
unseen cities fall back to the global mean and unseen countries to "Other".
No target value is ever used.

**Robust to partial input.** `transform` adds any missing raw column as `NaN`
before running the feature groups, so a sparse JSON record sent to
`predict.py` still produces every engineered feature (as `NaN`, which the
`ColumnTransformer`'s imputer handles exactly as it would a missing training
value). Booleans are parsed case-insensitively (`true`/`True`/`1`).

## Ordering matters

`FeatureEngineer.transform` calls the groups in a specific order because
Group E depends on columns Group A and Group C create
(`rating_avg`, `log_avg_rate`, `log_reviews`):

```
A (rating) → B (capacity) → C (log) → D (price ratio) → E (interaction) → F (market) → G (country)
```

`create_interaction_features` raises a clear `KeyError` if called before
`create_log_features` — this is intentional, so a reordering bug fails loudly
in `tests/test_pipeline.py` rather than silently producing `NaN` interaction
columns.
