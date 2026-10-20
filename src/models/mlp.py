"""MLP Neural Network baseline (Section 3.4.4)."""
from __future__ import annotations

from sklearn.neural_network import MLPRegressor

from src.utils.config import load_model_config


def get_mlp_model(random_state: int = 42, **kwargs) -> MLPRegressor:
    cfg = load_model_config()["mlp"]
    return MLPRegressor(
        random_state=random_state,
        early_stopping=cfg.get("early_stopping", True),
        validation_fraction=cfg.get("validation_fraction", 0.1),
        n_iter_no_change=cfg.get("n_iter_no_change", 20),
        **kwargs,
    )


def get_param_grid() -> dict:
    return load_model_config()["mlp"]["param_grid"]


def get_cv_folds() -> int:
    return load_model_config()["mlp"].get("cv", 3)
