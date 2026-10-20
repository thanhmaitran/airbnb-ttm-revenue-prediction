#!/usr/bin/env python
"""Training CLI.

    python train.py --model xgboost
    python train.py --model hist_gb --feature-set pre_booking
    python train.py --model random_forest --cv 5

Trains one model end to end (feature engineering -> pipeline -> GridSearchCV)
and writes the fitted pipeline to models/ plus a metrics JSON beside it.
"""
from __future__ import annotations

import argparse

from src.models.pipeline import MODEL_FACTORIES
from src.training_utils import (
    get_param_grid, grid_size, load_training_frame, save_model_artifacts, train_one_model,
)
from src.utils.config import load_training_config
from src.utils.seed import set_global_seed


def main(model_name: str, cv_override: int | None, n_jobs: int, feature_set: str) -> None:
    set_global_seed(load_training_config()["seed"])
    X_train, X_test, y_train, y_test = load_training_frame(feature_set)

    print(f"Training {model_name} [{feature_set} features] | grid size={grid_size(get_param_grid(model_name))}")
    r = train_one_model(model_name, X_train, X_test, y_train, y_test,
                        cv_override=cv_override, n_jobs=n_jobs, feature_set=feature_set)

    print(f"Best params: {r['best_params']}")
    print(f"Test  -> MAE=${r['MAE']:,.0f}  RMSE=${r['RMSE']:,.0f}  R2={r['R2_test']:.4f}")
    print(f"CV(5) -> R2={r['R2_cv']:.4f} +/- {r['R2_cv_std']:.4f}   train R2={r['R2_train']:.4f}")
    print(f"Training time: {r['time_seconds']:.1f}s")

    save_model_artifacts(r)
    print("Saved fitted pipeline to models/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a single model end to end.")
    ap.add_argument("--model", required=True, choices=list(MODEL_FACTORIES))
    ap.add_argument("--feature-set", default="full", choices=["full", "no_performance", "pre_booking"])
    ap.add_argument("--cv", type=int, default=None, help="Override GridSearchCV fold count")
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    main(a.model, a.cv, a.n_jobs, a.feature_set)
