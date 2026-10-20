"""Shared training routine used by train.py (single model), benchmark.py
(all models) and make_figures.py — kept in one place so they can never
drift apart on how data is split, a model is trained or scored.
"""
from __future__ import annotations

import pickle
import time
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.evaluation.metrics import evaluate_predictions
from src.features.feature_engineer import ALL_RAW_COLUMNS
from src.models.pipeline import get_feature_engineered_pipeline
from src.preprocessing.splitting import make_train_test_split
from src.utils.config import load_model_config, load_training_config, resolve_path
from src.utils.io import ensure_dir, read_table, save_json


def get_param_grid(model_name: str) -> dict:
    return load_model_config()[model_name]["param_grid"]


def get_cv_folds(model_name: str) -> int:
    return load_model_config()[model_name].get("cv", 3)


def grid_size(param_grid: dict) -> int:
    return len(list(product(*param_grid.values())))


def feature_lists(feature_set: str = "full") -> Tuple[List[str], List[str]]:
    """(numeric, categorical) feature names for 'full' or an ablation set."""
    cfg = load_training_config()
    numeric, categorical = list(cfg["numeric_features"]), list(cfg["categorical_features"])
    if feature_set != "full":
        ablations = cfg["ablations"]
        if feature_set not in ablations:
            raise ValueError(f"Unknown feature set '{feature_set}'. Choose from full, {list(ablations)}")
        drop = set(ablations[feature_set])
        numeric = [c for c in numeric if c not in drop]
    return numeric, categorical


def load_training_frame(feature_set: str = "full", return_meta: bool = False):
    """cleaned data -> *raw-column* train/test split.

    X holds only raw listing columns; FeatureEngineer runs inside the
    Pipeline (fit on training rows only). Same seed => same split in every
    CLI. With return_meta=True also returns country/city of the test rows.
    """
    cfg = load_training_config()
    df = read_table(resolve_path("processed_dataset"))
    target = pd.to_numeric(df[cfg["target"]], errors="coerce")
    df = df[target > 0]
    target = target[target > 0]

    X = df[ALL_RAW_COLUMNS].copy()
    y = np.log1p(target.values) if cfg["log_transform_target"] else target.values
    X_train, X_test, y_train, y_test = make_train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["seed"],
    )
    if return_meta:
        return X_train, X_test, y_train, y_test, X_test[["country", "city"]]
    return X_train, X_test, y_train, y_test


def train_one_model(
    model_name: str,
    X_train, X_test, y_train, y_test,
    cv_override: Optional[int] = None,
    n_jobs: int = -1,
    seed: Optional[int] = None,
    feature_set: str = "full",
) -> Dict[str, Any]:
    """GridSearchCV one model; return the fitted pipeline and every Table-4
    metric (MAE, RMSE, R2 test, R2 CV mean/std, R2 train, training time)."""
    cfg = load_training_config()
    seed = seed if seed is not None else cfg["seed"]
    numeric, categorical = feature_lists(feature_set)

    pipe = get_feature_engineered_pipeline(model_name, random_state=seed,
                                           numeric_features=numeric, categorical_features=categorical,
                                           top_n_countries=cfg["top_n_countries"])
    cv = cv_override or get_cv_folds(model_name)
    gs = GridSearchCV(pipe, get_param_grid(model_name), cv=cv, scoring="r2", n_jobs=n_jobs)

    t0 = time.time()
    gs.fit(X_train, y_train)
    elapsed = time.time() - t0

    cv_scores = cross_val_score(gs.best_estimator_, X_train, y_train, cv=5, scoring="r2", n_jobs=n_jobs)
    test_metrics = evaluate_predictions(y_test, gs.predict(X_test))
    train_metrics = evaluate_predictions(y_train, gs.predict(X_train))

    return {
        "model": model_name,
        "feature_set": feature_set,
        "n_features": len(numeric) + len(categorical),
        "best_estimator": gs.best_estimator_,
        "best_params": gs.best_params_,
        "gridsearch_cv_r2": float(gs.best_score_),
        "MAE": test_metrics["MAE"],
        "RMSE": test_metrics["RMSE"],
        "R2_test": test_metrics["R2"],
        "R2_cv": float(cv_scores.mean()),
        "R2_cv_std": float(cv_scores.std()),
        "R2_train": train_metrics["R2"],
        "time_seconds": round(elapsed, 2),
    }


def model_path(model_name: str, feature_set: str = "full"):
    suffix = "" if feature_set == "full" else f"_{feature_set}"
    return resolve_path("model_dir") / f"{model_name}{suffix}_pipeline.pkl"


def save_model_artifacts(result: Dict[str, Any]) -> None:
    """Persist the fitted pipeline + a metrics JSON under models/."""
    ensure_dir(resolve_path("model_dir"))
    name, fset = result["model"], result.get("feature_set", "full")
    with open(model_path(name, fset), "wb") as f:
        pickle.dump(result["best_estimator"], f)

    suffix = "" if fset == "full" else f"_{fset}"
    save_json({
        "model": name,
        "feature_set": fset,
        "best_params": result["best_params"],
        "test_metrics": {"MAE": result["MAE"], "RMSE": result["RMSE"], "R2": result["R2_test"]},
        "cv_mean": result["R2_cv"],
        "cv_std": result["R2_cv_std"],
        "train_time_seconds": result["time_seconds"],
    }, resolve_path("model_dir") / f"{name}{suffix}_metrics.json")
