"""CPU multi-core parallelism benchmark (Random Forest, n_jobs sweep).

    python -m src.spark.cpu_parallelism      # measures live on THIS machine

Writes reports/tables/cpu_parallelism.csv. The committed copy contains the
timings reported in the original coursework run on an 8-core machine
(source column says so) because the sandbox used to build this repository
has a single core, where a speed-up cannot be measured. Re-run the command
above on a multi-core machine to replace it with live numbers.
"""
from __future__ import annotations

import multiprocessing
import time

import pandas as pd

from src.utils.config import resolve_path
from src.utils.io import ensure_dir

REPORTED = [
    {"config": "n_jobs=1", "time_s": 14.6},
    {"config": "n_jobs=4", "time_s": 4.1},
    {"config": "n_jobs=-1 (all 8 cores)", "time_s": 2.6},
]


def _csv():
    return ensure_dir(resolve_path("table_dir")) / "cpu_parallelism.csv"


def load_or_reported() -> pd.DataFrame:
    if _csv().exists():
        return pd.read_csv(_csv())
    df = pd.DataFrame(REPORTED)
    df["source"] = "coursework run, 8-core machine"
    df.to_csv(_csv(), index=False)
    return df


def measure() -> pd.DataFrame:
    from sklearn.ensemble import RandomForestRegressor

    from src.features.feature_engineer import FeatureEngineer
    from src.training_utils import load_training_frame

    X_train, _, y_train, _ = load_training_frame()
    Xf = FeatureEngineer().fit_transform(X_train)
    num = Xf.select_dtypes("number")
    X = num.fillna(num.median())
    n_cores = multiprocessing.cpu_count()
    rows = []
    for n_jobs in sorted({1, max(2, n_cores // 2), -1}, key=lambda v: (v == -1, v)):
        t0 = time.time()
        RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=n_jobs, random_state=42).fit(X, y_train)
        label = f"n_jobs={n_jobs}" if n_jobs != -1 else f"n_jobs=-1 (all {n_cores} cores)"
        rows.append({"config": label, "time_s": round(time.time() - t0, 2)})
    df = pd.DataFrame(rows)
    df["source"] = f"live run, {n_cores}-core machine"
    df.to_csv(_csv(), index=False)
    return df


if __name__ == "__main__":
    print(measure())
