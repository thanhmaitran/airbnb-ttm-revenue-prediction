"""Tests for src/utils/io.py and path overrides in src/utils/config.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src.utils.config import PROJECT_ROOT, resolve_path
from src.utils.io import read_table, write_table


def test_csv_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    write_table(df, tmp_path / "t.csv")
    pd.testing.assert_frame_equal(read_table(tmp_path / "t.csv"), df)


def test_read_missing_file_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data.py"):
        read_table(tmp_path / "nope.csv")


def test_env_var_overrides_path(monkeypatch):
    monkeypatch.setenv("AIRBNB_PROCESSED_DATASET", "data/sample/sample_listings.csv")
    assert resolve_path("processed_dataset") == PROJECT_ROOT / "data/sample/sample_listings.csv"


def test_unknown_path_key_raises():
    with pytest.raises(KeyError):
        resolve_path("does_not_exist")
