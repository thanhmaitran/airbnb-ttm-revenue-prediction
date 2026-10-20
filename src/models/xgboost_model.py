"""XGBoost Regressor with L1/L2 regularisation and optional GPU support
(Section 3.4.3). Named xgboost_model.py to avoid shadowing the `xgboost`
package on import.

`xgboost` is imported lazily inside each function rather than at module
level, so `import src.models.pipeline` (and therefore `ridge`/
`random_forest`/`mlp` training) still works in environments where xgboost
isn't installed — only calling get_xgb_model() itself requires it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.config import load_model_config

if TYPE_CHECKING:
    import xgboost as xgb


def detect_device() -> str:
    """Probe for CUDA support; fall back to CPU if unavailable. Mirrors the
    try/except GPU probe used in the original coursework script, but keeps
    it out of the training loop so it only ever runs once.
    """
    import numpy as np
    import xgboost as xgb

    try:
        _probe = xgb.XGBRegressor(device="cuda", n_estimators=2, verbosity=0)
        _probe.fit(np.zeros((4, 2)), np.zeros(4))
        return "cuda"
    except Exception:
        return "cpu"


def get_xgb_model(random_state: int = 42, device: str | None = None, **kwargs) -> "xgb.XGBRegressor":
    import xgboost as xgb

    cfg = load_model_config()["xgboost"]
    configured_device = device or cfg.get("device", "auto")
    if configured_device == "auto":
        configured_device = detect_device()
    return xgb.XGBRegressor(
        device=configured_device,
        random_state=random_state,
        verbosity=0,
        **kwargs,
    )


def get_param_grid() -> dict:
    return load_model_config()["xgboost"]["param_grid"]


def get_cv_folds() -> int:
    return load_model_config()["xgboost"].get("cv", 3)
