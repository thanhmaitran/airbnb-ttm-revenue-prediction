"""Random Forest Regressor — bagging ensemble baseline (Section 3.4.2)."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from src.utils.config import load_model_config


def get_rf_model(random_state: int = 42, **kwargs) -> RandomForestRegressor:
    cfg = load_model_config()["random_forest"]
    return RandomForestRegressor(
        random_state=random_state,
        n_jobs=cfg.get("n_jobs", -1),
        **kwargs,
    )


def get_param_grid() -> dict:
    return load_model_config()["random_forest"]["param_grid"]


def get_cv_folds() -> int:
    return load_model_config()["random_forest"].get("cv", 3)
