"""Reproducibility helpers: a single place that sets every random seed."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set the seed for Python, NumPy, and relevant env vars.

    Individual sklearn/XGBoost estimators still take `random_state=seed`
    explicitly (see src/models/), this just covers everything else so
    results are reproducible end to end.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
