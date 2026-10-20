#!/usr/bin/env python
"""Evaluation CLI.

    python evaluate.py --model xgboost

Loads a pipeline saved by train.py/benchmark.py, re-scores it on the same
held-out split (same seed => same rows) and upserts MAE/RMSE/R2 into
outputs/metrics.csv, plus writes row-level predictions to
outputs/predictions.csv for the evaluated model.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_predictions
from src.models.pipeline import MODEL_FACTORIES
from src.training_utils import load_training_frame, model_path
from src.utils.config import resolve_path
from src.utils.io import ensure_dir


def main(model_name: str, feature_set: str) -> None:
    path = model_path(model_name, feature_set)
    if not path.exists():
        raise FileNotFoundError(f"No trained pipeline at {path}. Run `python train.py --model {model_name}` first.")
    with open(path, "rb") as f:
        pipe = pickle.load(f)

    _, X_test, _, y_test, meta = load_training_frame(feature_set, return_meta=True)
    y_pred = pipe.predict(X_test)
    m = evaluate_predictions(y_test, y_pred)
    print(f"{model_name} [{feature_set}]: MAE=${m['MAE']:,.0f}  RMSE=${m['RMSE']:,.0f}  R2={m['R2']:.4f}")

    out = resolve_path("metrics_file")
    ensure_dir(out.parent)
    row = pd.DataFrame([{"model": model_name, "feature_set": feature_set, **m}])
    if out.exists():
        old = pd.read_csv(out)
        old = old[~((old["model"] == model_name) & (old["feature_set"] == feature_set))]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(out, index=False)

    pred_out = pd.DataFrame({
        "actual_usd": np.expm1(y_test), "predicted_usd": np.expm1(y_pred),
        "country": meta["country"].values, "city": meta["city"].values,
    })
    pred_out.to_csv(resolve_path("predictions_file"), index=False)
    print(f"Updated {out} and {resolve_path('predictions_file')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a trained pipeline on the held-out test set.")
    ap.add_argument("--model", required=True, choices=list(MODEL_FACTORIES))
    ap.add_argument("--feature-set", default="full", choices=["full", "no_performance", "pre_booking"])
    a = ap.parse_args()
    main(a.model, a.feature_set)
