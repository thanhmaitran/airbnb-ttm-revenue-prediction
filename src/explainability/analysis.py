"""Model explainability without extra dependencies (scikit-learn only).

Three views of a fitted *full* pipeline (raw columns in, USD out):

* permutation_importance_table - how much test R^2 drops when one raw input
  column is shuffled (model-agnostic, measured on held-out rows)
* partial_dependence_usd       - what-if curves: set one raw column to a grid
  of values for every listing and average the predicted revenue. Because the
  FeatureEngineer sits inside the pipeline, derived features (revpar_proxy,
  price_vs_city_mean, ...) are recomputed for every grid value, so the curve
  reflects the full effect of changing the input.
* error_analysis_tables        - where the model is wrong: by revenue decile
  and by country.

SHAP (notebooks/04_SHAP.ipynb) complements these once `shap` is installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_table(pipe, X_test: pd.DataFrame, y_test_log: np.ndarray,
                                 n_repeats: int = 3, max_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=min(max_rows, len(X_test)), replace=False)
    X_s, y_s = X_test.iloc[idx], y_test_log[idx]
    res = permutation_importance(pipe, X_s, y_s, scoring="r2", n_repeats=n_repeats,
                                 random_state=seed, n_jobs=1)
    out = pd.DataFrame({
        "feature": X_test.columns,
        "r2_drop_mean": res.importances_mean,
        "r2_drop_std": res.importances_std,
    })
    return out.sort_values("r2_drop_mean", ascending=False).reset_index(drop=True)


def partial_dependence_usd(pipe, X_test: pd.DataFrame, column: str, n_grid: int = 25,
                           max_rows: int = 3000, seed: int = 42,
                           quantile_range=(0.02, 0.98)) -> pd.DataFrame:
    """Average predicted revenue (USD) as `column` sweeps its 2nd-98th percentile."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=min(max_rows, len(X_test)), replace=False)
    X_s = X_test.iloc[idx].copy()
    values = pd.to_numeric(X_s[column], errors="coerce")
    grid = np.linspace(values.quantile(quantile_range[0]), values.quantile(quantile_range[1]), n_grid)

    means = []
    for v in grid:
        X_mod = X_s.copy()
        X_mod[column] = v
        means.append(float(np.expm1(pipe.predict(X_mod)).mean()))
    return pd.DataFrame({column: grid, "avg_predicted_revenue_usd": means})


def error_analysis_tables(actual_usd: np.ndarray, predicted_usd: np.ndarray, countries: np.ndarray,
                          top_countries: int = 10):
    df = pd.DataFrame({"actual": actual_usd, "pred": predicted_usd, "country": countries})
    df["abs_err"] = (df["pred"] - df["actual"]).abs()
    df["ape"] = df["abs_err"] / df["actual"].clip(lower=1)

    df["revenue_decile"] = pd.qcut(df["actual"], 10, labels=[f"D{i}" for i in range(1, 11)])
    by_decile = df.groupby("revenue_decile", observed=True).agg(
        n=("actual", "size"), mean_actual=("actual", "mean"),
        MAE=("abs_err", "mean"), median_ape=("ape", "median")).reset_index()

    top = df["country"].value_counts().head(top_countries).index
    by_country = (df[df["country"].isin(top)].groupby("country")
                  .agg(n=("actual", "size"), MAE=("abs_err", "mean"), median_ape=("ape", "median"))
                  .sort_values("n", ascending=False).reset_index())
    return by_decile, by_country
