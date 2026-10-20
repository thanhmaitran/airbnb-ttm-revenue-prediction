"""Config loading helpers.

All paths and hyperparameters live in YAML files under configs/. Nothing in
this codebase should hardcode a filesystem path or a hyperparameter value —
load it from here instead.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _load_yaml(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / name
    with open(path, "r") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=None)
def load_paths() -> Dict[str, Any]:
    return _load_yaml("paths.yaml")


@functools.lru_cache(maxsize=None)
def load_training_config() -> Dict[str, Any]:
    return _load_yaml("training_config.yaml")


@functools.lru_cache(maxsize=None)
def load_model_config() -> Dict[str, Any]:
    return _load_yaml("model_config.yaml")


def resolve_path(key: str) -> Path:
    """Resolve a path key from configs/paths.yaml to an absolute Path.

    Any key can be overridden without editing YAML via an environment
    variable named AIRBNB_<KEY>, e.g.

        AIRBNB_PROCESSED_DATASET=data/sample/sample_listings.csv python train.py --model ridge
    """
    paths = load_paths()
    if key not in paths:
        raise KeyError(f"Unknown path key '{key}'. Available: {list(paths)}")
    override = os.environ.get(f"AIRBNB_{key.upper()}")
    value = override if override else paths[key]
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
