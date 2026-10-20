"""Group B — Capacity Ratios.

bed_per_guest / bath_per_guest act as a comfort proxy: under-bedded listings
tend to earn less per unit of guest capacity.
"""
from __future__ import annotations

import pandas as pd


def create_capacity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bed_per_guest"] = df["beds"].fillna(1) / (df["guests"].fillna(2) + 1e-9)
    df["bath_per_guest"] = df["baths"].fillna(1) / (df["guests"].fillna(2) + 1e-9)
    return df
