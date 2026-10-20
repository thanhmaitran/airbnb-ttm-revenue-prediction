"""Small IO helpers shared across scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    """Read .parquet / .csv / .csv.gz / .pkl by file extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python prepare_data.py` or place the raw "
            "files under data/raw/ (see README.md > Data)."
        )
    name = path.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    if name.endswith((".csv", ".csv.gz")):
        # The raw export escapes embedded quotes with a backslash; without
        # escapechar the C parser stops at the first such row.
        return pd.read_csv(path, escapechar="\\", low_memory=False)
    if name.endswith(".pkl"):
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    name = path.name.lower()
    if name.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif name.endswith((".csv", ".csv.gz")):
        df.to_csv(path, index=False)
    elif name.endswith(".pkl"):
        df.to_pickle(path)
    else:
        raise ValueError(f"Unsupported file type: {path.name}")


# Backwards-compatible alias
read_parquet = read_table


def ensure_dir(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
