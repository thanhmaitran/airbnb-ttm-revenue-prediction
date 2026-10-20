#!/usr/bin/env python
"""Benchmark CLI — trains several models on the same split and writes one
comparison table (Table 4 of the technical report).

    python benchmark.py                                   # ridge, random_forest, xgboost, mlp
    python benchmark.py --models ridge random_forest hist_gb mlp
    python benchmark.py --models ridge hist_gb --feature-set pre_booking

Writes reports/tables/benchmark_results[_<feature_set>].csv and saves every
fitted pipeline to models/.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.models.pipeline import MODEL_FACTORIES
from src.training_utils import load_training_frame, save_model_artifacts, train_one_model
from src.utils.config import load_training_config, resolve_path
from src.utils.io import ensure_dir
from src.utils.seed import set_global_seed

DEFAULT_MODELS = ["ridge", "random_forest", "xgboost", "mlp"]
COLUMNS = ["model", "feature_set", "n_features", "MAE", "RMSE", "R2_test",
           "R2_cv", "R2_cv_std", "R2_train", "time_seconds", "best_params"]


def main(models, cv_override, n_jobs, save_models, feature_set) -> None:
    set_global_seed(load_training_config()["seed"])
    X_train, X_test, y_train, y_test = load_training_frame(feature_set)
    print(f"train={len(X_train):,} test={len(X_test):,} features={X_train.shape[1]} [{feature_set}]")

    rows = []
    for name in models:
        print(f"\n{'=' * 60}\n{name} [{feature_set}]\n{'=' * 60}", flush=True)
        r = train_one_model(name, X_train, X_test, y_train, y_test,
                            cv_override=cv_override, n_jobs=n_jobs, feature_set=feature_set)
        print(f"  MAE=${r['MAE']:,.0f}  RMSE=${r['RMSE']:,.0f}  R2_test={r['R2_test']:.4f}  "
              f"R2_cv={r['R2_cv']:.4f}+/-{r['R2_cv_std']:.4f}  time={r['time_seconds']:.1f}s", flush=True)
        if save_models:
            save_model_artifacts(r)
        rows.append({**r, "best_params": str(r["best_params"])})

    table = pd.DataFrame(rows)[COLUMNS]
    suffix = "" if feature_set == "full" else f"_{feature_set}"
    out = ensure_dir(resolve_path("table_dir")) / f"benchmark_results{suffix}.csv"
    if out.exists():  # merge: re-running a subset of models updates only those rows
        old = pd.read_csv(out)
        table = pd.concat([old[~old["model"].isin(table["model"])], table], ignore_index=True)
    table = table.sort_values("R2_test", ascending=False)
    table.to_csv(out, index=False)
    print(f"\nSaved -> {out}\n")
    print(table.drop(columns=["best_params"]).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train and compare models.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODEL_FACTORIES))
    ap.add_argument("--feature-set", default="full", choices=["full", "no_performance", "pre_booking"])
    ap.add_argument("--cv", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--no-save", dest="save_models", action="store_false")
    a = ap.parse_args()
    main(a.models, a.cv, a.n_jobs, a.save_models, a.feature_set)
