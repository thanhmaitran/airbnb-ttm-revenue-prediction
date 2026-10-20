"""Histogram-based Gradient Boosting (scikit-learn) — dependency-free
gradient-boosted-tree model.

Why it exists alongside XGBoost: it needs nothing beyond scikit-learn, so
the whole benchmark can run (and CI can test it) on machines where
`xgboost` is not installed. It is a *different implementation* of the same
model family (histogram-binned gradient boosting, LightGBM-style) and is
always reported under its own name — never as XGBoost.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor

from src.utils.config import load_model_config


def get_hist_gb_model(random_state: int = 42, **kwargs) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=random_state, **kwargs)


def get_param_grid() -> dict:
    return load_model_config()["hist_gb"]["param_grid"]


def get_cv_folds() -> int:
    return load_model_config()["hist_gb"].get("cv", 3)
