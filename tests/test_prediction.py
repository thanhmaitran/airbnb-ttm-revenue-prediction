"""Tests for predict.py - the saved pipeline scores raw JSON-like records
through the *fitted* FeatureEngineer (train/serve symmetry)."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

import predict as predict_module
import src.training_utils as training_utils
from src.features.feature_engineer import ALL_RAW_COLUMNS
from src.models.pipeline import get_feature_engineered_pipeline
from src.utils.config import load_training_config


@pytest.fixture
def saved_model_dir(raw_listings_df, tmp_path, monkeypatch):
    cfg = load_training_config()
    pipe = get_feature_engineered_pipeline("ridge", random_state=42)
    pipe.fit(raw_listings_df[ALL_RAW_COLUMNS], np.log1p(raw_listings_df[cfg["target"]].values))

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    with open(model_dir / "ridge_pipeline.pkl", "wb") as f:
        pickle.dump(pipe, f)
    monkeypatch.setattr(training_utils, "resolve_path",
                        lambda key: model_dir if key == "model_dir" else None)
    return model_dir


def test_predict_full_records(raw_listings_df, saved_model_dir):
    records = raw_listings_df.drop(columns=["ttm_revenue"]).head(3).to_dict(orient="records")
    preds = predict_module.predict(records, "ridge")
    assert len(preds) == 3 and all(p > 0 and np.isfinite(p) for p in preds)


def test_predict_partial_record(saved_model_dir):
    """A hand-written, sparse listing still scores (missing fields imputed)."""
    preds = predict_module.predict([{"guests": 4, "bedrooms": 2, "city": "London",
                                     "country": "United Kingdom", "ttm_avg_rate": 120.0,
                                     "ttm_occupancy": 0.4}], "ridge")
    assert len(preds) == 1 and np.isfinite(preds[0])


def test_single_record_matches_batch_prediction(raw_listings_df, saved_model_dir):
    """Scoring a row alone must equal scoring it inside a batch - this is the
    train/serve-skew regression test (city means come from fit, not the frame)."""
    records = raw_listings_df.drop(columns=["ttm_revenue"]).head(20).to_dict(orient="records")
    batch = predict_module.predict(records, "ridge")
    singles = [predict_module.predict([r], "ridge")[0] for r in records[:5]]
    assert np.allclose(batch[:5], singles, rtol=1e-6)


def test_predict_missing_model_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(training_utils, "resolve_path", lambda key: tmp_path if key == "model_dir" else None)
    with pytest.raises(FileNotFoundError):
        predict_module.load_pipeline("xgboost")
