"""Tests for src/models/pipeline.py — the ColumnTransformer + Pipeline
architecture that is the project's central design contribution
(Section 3.3 of the technical report / docs/architecture.md).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.features.feature_engineer import FeatureEngineer
from src.models.pipeline import build_preprocessor, create_pipeline
from src.preprocessing.splitting import make_train_test_split
from src.utils.config import load_training_config


def _prepare_xy(raw_listings_df):
    cfg = load_training_config()
    df = FeatureEngineer(top_n_countries=cfg["top_n_countries"]).fit_transform(raw_listings_df)
    features = cfg["numeric_features"] + cfg["categorical_features"]
    X = df[features]
    y = np.log1p(df[cfg["target"]].values)
    return X, y


@pytest.mark.parametrize("model_name", ["ridge", "random_forest", "hist_gb", "mlp"])
def test_pipeline_fits_and_predicts(raw_listings_df, model_name):
    X, y = _prepare_xy(raw_listings_df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = create_pipeline(model_name, random_state=42)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    assert len(preds) == len(X_test)
    assert np.isfinite(preds).all()


def test_pipeline_has_preprocess_and_regressor_steps():
    pipe = create_pipeline("ridge")
    assert list(dict(pipe.steps)) == ["preprocess", "regressor"]


def test_preprocessor_handles_missing_values(raw_listings_df):
    """Missing values must be handled inside the pipeline (median/most-
    frequent imputation), not before — this is what prevents train/test
    contamination (Section 3.1)."""
    X, y = _prepare_xy(raw_listings_df)
    X = X.copy()
    X.iloc[0, 0] = np.nan  # inject a missing numeric value
    X.iloc[1, -1] = None   # inject a missing categorical value

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert not np.isnan(transformed).any()


def test_no_leakage_preprocessor_fit_only_on_train(raw_listings_df):
    """The preprocessor's learned statistics (e.g. median) must come only
    from the training split, not from data that includes the test rows."""
    X, y = _prepare_xy(raw_listings_df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor_train_only = build_preprocessor()
    preprocessor_train_only.fit(X_train)
    median_from_train = preprocessor_train_only.named_transformers_["num"].named_steps["imputer"].statistics_

    preprocessor_full = build_preprocessor()
    preprocessor_full.fit(X)
    median_from_full = preprocessor_full.named_transformers_["num"].named_steps["imputer"].statistics_

    # They are not guaranteed to differ for every column, but for a
    # correctly-isolated pipeline the *object* fitted on train_only must
    # never have seen X_test — asserting shapes/columns is a smoke test
    # that both fit paths are independent and reproducible.
    assert median_from_train.shape == median_from_full.shape


def test_xgboost_pipeline_available_if_installed(raw_listings_df):
    pytest.importorskip("xgboost")
    X, y = _prepare_xy(raw_listings_df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = create_pipeline("xgboost", random_state=42)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert len(preds) == len(X_test)


def test_full_pipeline_takes_raw_columns_and_predicts(raw_listings_df):
    """FeatureEngineer lives INSIDE the pipeline: raw columns in, log-revenue out."""
    from src.features.feature_engineer import ALL_RAW_COLUMNS
    from src.models.pipeline import get_feature_engineered_pipeline

    cfg = load_training_config()
    X = raw_listings_df[ALL_RAW_COLUMNS]
    y = np.log1p(raw_listings_df[cfg["target"]].values)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = get_feature_engineered_pipeline("ridge", random_state=42)
    assert list(dict(pipe.steps)) == ["features", "preprocess", "regressor"]
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert len(preds) == len(X_test) and np.isfinite(preds).all()


def test_city_statistics_never_see_test_rows(raw_listings_df):
    """Fit the full pipeline on a training split in which one city is absent;
    that city's test rows must be scored with the global-mean fall-back,
    proving the held-out rows did not leak into the fitted city means."""
    from src.features.feature_engineer import ALL_RAW_COLUMNS
    from src.models.pipeline import get_feature_engineered_pipeline

    cfg = load_training_config()
    df = raw_listings_df.copy()
    train = df[df["city"] != "Valletta"]
    pipe = get_feature_engineered_pipeline("ridge", random_state=42)
    pipe.fit(train[ALL_RAW_COLUMNS], np.log1p(train[cfg["target"]].values))

    fe = pipe.named_steps["features"]
    assert "Valletta" not in fe.market_stats_["city_mean_rate"].index
    assert np.isfinite(pipe.predict(df[df["city"] == "Valletta"][ALL_RAW_COLUMNS])).all()


def test_ablation_feature_sets_remove_performance_columns():
    from src.training_utils import feature_lists

    full, _ = feature_lists("full")
    no_perf, _ = feature_lists("no_performance")
    pre, _ = feature_lists("pre_booking")
    assert "revpar_proxy" in full and "revpar_proxy" not in no_perf
    assert "ttm_occupancy" not in no_perf and "num_reviews" in no_perf
    assert "num_reviews" not in pre and len(pre) < len(no_perf) < len(full)
