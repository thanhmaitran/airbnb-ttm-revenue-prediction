"""Evaluation protocol (Section 3.6): MAE, RMSE and R^2 computed on the
original revenue scale (expm1 applied to log-predictions), plus 5-fold CV
mean/std for stability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def evaluate_predictions(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """y_* are on the log1p(revenue) scale; MAE/RMSE are reported back on
    the original USD scale via expm1, matching Table 4 of the report.
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true_log, y_pred_log),
    }


def cross_validate_r2(estimator, X, y, cv: int = 5) -> tuple[float, float]:
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="r2")
    return float(scores.mean()), float(scores.std())


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results)
