"""Tests for src/features/ — including the no-leakage guarantee that is a
core design constraint of this project (see README.md > Feature Engineering).
"""
from __future__ import annotations

from src.features.feature_engineer import FeatureEngineer

TARGET = "ttm_revenue"


def test_feature_engineer_does_not_touch_target(raw_listings_df):
    """No engineered feature may be derived from ttm_revenue — this is the
    single most important invariant in the whole codebase (Section 3.1 of
    docs/reproducibility.md).
    """
    df = raw_listings_df.copy()
    original_target = df[TARGET].copy()

    fe = FeatureEngineer()
    out = fe.fit_transform(df)

    # The target column itself must be untouched...
    assert (out[TARGET] == original_target).all()
    # ...and no new column name should reference it.
    leaky_columns = [c for c in out.columns if "revenue" in c.lower() and c != TARGET]
    assert leaky_columns == [], f"Possible target leakage in columns: {leaky_columns}"


def test_feature_engineer_produces_expected_groups(raw_listings_df):
    out = FeatureEngineer().fit_transform(raw_listings_df)
    expected = {
        "rating_avg", "rating_std",                      # Group A
        "bed_per_guest", "bath_per_guest",                # Group B
        "log_avg_rate", "log_reviews", "log_cleaning",    # Group C
        "cleaning_fee_ratio",                             # Group D
        "rate_x_reviews", "revpar_proxy", "size_x_rating",  # Group E
        "price_vs_city_mean", "occ_vs_city_mean",         # Group F
        "country_grp",                                    # Group G
    }
    assert expected.issubset(set(out.columns))


def test_feature_engineer_no_missing_after_transform_on_complete_input(raw_listings_df):
    out = FeatureEngineer().fit_transform(raw_listings_df)
    for col in ["rating_avg", "bed_per_guest", "log_avg_rate", "revpar_proxy"]:
        assert out[col].notna().all(), f"Unexpected NaNs in {col}"


def test_country_grouping_collapses_long_tail(raw_listings_df):
    out = FeatureEngineer(top_n_countries=2).fit_transform(raw_listings_df)
    assert out["country_grp"].nunique() <= 3  # top 2 + 'Other'


def test_feature_engineer_output_row_count_matches_input(raw_listings_df):
    out = FeatureEngineer().fit_transform(raw_listings_df)
    assert len(out) == len(raw_listings_df)


def test_feature_engineer_tolerates_partial_input():
    """predict.py sends partial raw records (see tests/test_prediction.py);
    the FeatureEngineer must add missing raw columns as NaN rather than
    raising, so a fresh, sparse listing can still be scored.
    """
    import pandas as pd

    partial = pd.DataFrame([{"guests": 4, "bedrooms": 2, "country": "United Kingdom", "city": "London"}])
    out = FeatureEngineer().fit_transform(partial)
    assert len(out) == 1
    assert "revpar_proxy" in out.columns


def test_market_features_are_fitted_on_training_rows_only(raw_listings_df):
    """City means must come from fit(), not from the frame being transformed:
    a single record scored on its own must be compared with the TRAINING
    city mean (not with itself, which would always give exactly 1.0)."""
    import pandas as pd

    train = raw_listings_df.iloc[:150]
    fe = FeatureEngineer().fit(train)

    one_row = raw_listings_df.iloc[[160]].copy()
    out = fe.transform(one_row)

    city = one_row["city"].iloc[0]
    expected = one_row["ttm_avg_rate"].iloc[0] / (train.loc[train["city"] == city, "ttm_avg_rate"].mean() + 1e-9)
    assert abs(out["price_vs_city_mean"].iloc[0] - expected) < 1e-6
    # Fitting on the single row itself would have produced exactly 1.0:
    self_fit = FeatureEngineer().fit_transform(one_row)
    assert abs(self_fit["price_vs_city_mean"].iloc[0] - 1.0) < 1e-6
    assert isinstance(out, pd.DataFrame)


def test_unseen_city_falls_back_to_global_mean(raw_listings_df):
    train = raw_listings_df.copy()
    fe = FeatureEngineer().fit(train)
    new = train.iloc[[0]].copy()
    new["city"] = "Atlantis"
    out = fe.transform(new)
    expected = new["ttm_avg_rate"].iloc[0] / (train["ttm_avg_rate"].mean() + 1e-9)
    assert abs(out["price_vs_city_mean"].iloc[0] - expected) < 1e-6


def test_top_countries_learned_at_fit(raw_listings_df):
    fe = FeatureEngineer(top_n_countries=2).fit(raw_listings_df)
    assert len(fe.top_countries_) == 2
    new = raw_listings_df.iloc[[0]].copy()
    new["country"] = "Narnia"
    assert FeatureEngineer(top_n_countries=2).fit(raw_listings_df).transform(new)["country_grp"].iloc[0] == "Other"
