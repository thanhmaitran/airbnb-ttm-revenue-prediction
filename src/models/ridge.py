"""Ridge Regression — linear baseline (Section 3.4.1 of the technical report)."""
from __future__ import annotations

from sklearn.linear_model import Ridge

from src.utils.config import load_model_config


def get_ridge_model(**kwargs) -> Ridge:
    return Ridge(**kwargs)


def get_param_grid() -> dict:
    return load_model_config()["ridge"]["param_grid"]


def get_cv_folds() -> int:
    return load_model_config()["ridge"].get("cv", 5)
