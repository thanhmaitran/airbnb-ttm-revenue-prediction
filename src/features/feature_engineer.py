"""FeatureEngineer — a single, sklearn-compatible entry point over the
eleven feature groups (Table 2 of the technical report).

Each group lives in its own module (rating_features.py, capacity_features.py,
...) so it can be unit-tested and reasoned about independently. This class
just calls them in the correct order and exposes fit/transform so it can be
dropped into a scikit-learn Pipeline directly if desired.

    fe = FeatureEngineer()
    df_transformed = fe.fit_transform(df)

Note: none of the eleven groups look at ttm_revenue (the target), which is
a deliberate design choice to prevent target leakage (see Section 3.1 of
docs/reproducibility.md and the technical report).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.preprocessing.cleaning import to_binary
from src.features.capacity_features import create_capacity_features
from src.features.interaction_features import (
    create_interaction_features,
    create_price_ratio_features,
)
from src.features.log_features import create_log_features
from src.features.market_features import (
    create_country_grouping,
    create_market_relative_features,
    fit_market_statistics,
    fit_top_countries,
)
from src.features.rating_features import create_rating_features

NUMERIC_RAW = [
    "guests", "bedrooms", "beds", "baths", "min_nights",
    "num_reviews", "rating_overall",
    "rating_accuracy", "rating_checkin", "rating_cleanliness",
    "rating_communication", "rating_location", "rating_value",
    "cleaning_fee", "extra_guest_fee", "photos_count",
    "ttm_avg_rate", "ttm_occupancy",
]

# Every raw column any of the eleven feature groups reads from, beyond
# NUMERIC_RAW. A row missing one of these (e.g. a partial listing sent to
# predict.py) gets it added as NaN rather than raising a KeyError deep
# inside a feature group — the ColumnTransformer's imputer then handles it
# exactly like any other missing value at training time.
CATEGORICAL_RAW = ["country", "city", "room_type"]
BOOLEAN_RAW = ["superhost", "professional_management"]
ALL_RAW_COLUMNS = NUMERIC_RAW + CATEGORICAL_RAW + BOOLEAN_RAW


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Leak-free, *stateful* feature engineering, sklearn Pipeline compatible.

    fit() learns city means and top countries from the training rows only;
    transform() reuses them, so train and serve see identical features.

    Parameters
    ----------
    top_n_countries : int
        Number of countries to keep as their own one-hot category before
        collapsing the rest into 'Other'.
    """

    def __init__(self, top_n_countries: int = 10):
        self.top_n_countries = top_n_countries

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add missing raw columns and coerce types. Shared by fit and transform."""
        df = X.copy()

        # Tolerate partial/raw input (e.g. a hand-written predict.py JSON
        # record): add any missing expected column as NaN so downstream
        # feature groups can rely on it existing instead of raising KeyError.
        for col in ALL_RAW_COLUMNS:
            if col not in df.columns:
                # np.nan (not pd.NA) so sklearn imputers never meet an
                # ambiguous pd.NA boolean in an object-dtype column.
                df[col] = np.nan

        for col in NUMERIC_RAW:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in BOOLEAN_RAW:
            df[col] = to_binary(df[col])
        return df

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        """Learn the only two pieces of state: per-city mean rate/occupancy
        and the top-N countries. Both come from the rows passed to fit -
        inside GridSearchCV that is the training fold only. No target is used.
        """
        df = self._prepare(X)
        self.market_stats_ = fit_market_statistics(df)
        self.top_countries_ = fit_top_countries(df, self.top_n_countries)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["market_stats_", "top_countries_"])
        df = self._prepare(X)

        df = create_rating_features(df)            # Group A
        df = create_capacity_features(df)           # Group B
        df = create_log_features(df)                # Group C
        df = create_price_ratio_features(df)         # Group D
        df = create_interaction_features(df)         # Group E (needs A + C)
        df = create_market_relative_features(df, self.market_stats_)         # Group F
        df = create_country_grouping(df, top_countries=self.top_countries_)   # Group G

        return df.replace([np.inf, -np.inf], np.nan)
