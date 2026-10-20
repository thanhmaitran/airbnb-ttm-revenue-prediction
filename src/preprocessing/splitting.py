"""Train/test split and cross-validation fold helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def set_random_seed(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_train_test_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Thin wrapper around sklearn's train_test_split so the split logic
    lives in one place and is reused identically by train.py, evaluate.py
    and the notebooks.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def create_cv(n_splits: int = 5, random_state: int = 42) -> KFold:
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
