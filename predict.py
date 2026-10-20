#!/usr/bin/env python
"""Prediction CLI.

    python predict.py input.json
    python predict.py input.json --model hist_gb

Takes one or more *raw* listing records (the schema of listings.csv, minus
ttm_revenue) as JSON and prints the predicted TTM revenue in USD.

The saved pipeline already contains the fitted FeatureEngineer, so a record
is transformed with the *training-time* city means / top countries - there
is no separate feature code path that could drift from training.

    {"guests": 4, "bedrooms": 2, "beds": 2, "baths": 1, "num_reviews": 34,
     "rating_overall": 4.8, "ttm_avg_rate": 145.0, "ttm_occupancy": 0.42,
     "country": "United Kingdom", "city": "London", "room_type": "entire_home"}

Missing fields are imputed with training medians / modes, exactly like a
missing value in the training data.

Output:  {"prediction": 18642.4}
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd

from src.models.pipeline import MODEL_FACTORIES
from src.training_utils import model_path
from src.utils.config import load_training_config


def load_pipeline(model_name: str, feature_set: str = "full"):
    path = model_path(model_name, feature_set)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained pipeline at {path}. Run `python train.py --model {model_name}` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(records: list[dict], model_name: str, feature_set: str = "full") -> list[float]:
    cfg = load_training_config()
    pipe = load_pipeline(model_name, feature_set)
    y_pred = pipe.predict(pd.DataFrame.from_records(records))
    y_pred = np.expm1(y_pred) if cfg["log_transform_target"] else y_pred
    return [round(float(v), 1) for v in y_pred]


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict TTM revenue for one or more listings.")
    ap.add_argument("input", help="Path to a JSON file (single object or list of objects)")
    ap.add_argument("--model", default="xgboost", choices=list(MODEL_FACTORIES))
    ap.add_argument("--feature-set", default="full", choices=["full", "no_performance", "pre_booking"])
    args = ap.parse_args()

    with open(args.input) as f:
        payload = json.load(f)
    records = payload if isinstance(payload, list) else [payload]
    preds = predict(records, args.model, args.feature_set)
    print(json.dumps({"prediction": preds[0]} if len(preds) == 1 else {"predictions": preds}))


if __name__ == "__main__":
    sys.exit(main())
