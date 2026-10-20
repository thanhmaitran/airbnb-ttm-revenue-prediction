"""create_pipeline(model_name) — the production-grade ColumnTransformer +
Pipeline architecture described in Section 3.3 of the technical report.

This module only assembles pipelines; it never trains anything (that's
train.py's job). Keeping assembly and training separate means the exact
same preprocessing definition is reused by train.py, evaluate.py,
predict.py and every notebook, so there is no risk of train/serve skew.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_engineer import FeatureEngineer
from src.models.hist_gb import get_hist_gb_model
from src.models.mlp import get_mlp_model
from src.models.random_forest import get_rf_model
from src.models.ridge import get_ridge_model
from src.models.xgboost_model import get_xgb_model
from src.utils.config import load_training_config

MODEL_FACTORIES = {
    "ridge": get_ridge_model,
    "random_forest": get_rf_model,
    "xgboost": get_xgb_model,
    "hist_gb": get_hist_gb_model,
    "mlp": get_mlp_model,
}


def build_preprocessor(numeric_features=None, categorical_features=None) -> ColumnTransformer:
    """The ColumnTransformer: median-impute + scale numeric columns,
    mode-impute + one-hot encode categorical columns. Fitted exclusively
    on training folds by GridSearchCV/cross_val_score, so there is no
    leakage from test into train (Section 3.1).
    """
    cfg = load_training_config()
    numeric_features = numeric_features or cfg["numeric_features"]
    categorical_features = categorical_features or cfg["categorical_features"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def create_pipeline(
    model_name: str,
    random_state: int = 42,
    numeric_features=None,
    categorical_features=None,
    **model_kwargs,
) -> Pipeline:
    """Build the full ('preprocess', 'regressor') pipeline for one of
    {'ridge', 'random_forest', 'xgboost', 'hist_gb', 'mlp'}.

    numeric_features / categorical_features default to the lists in
    configs/training_config.yaml; pass a subset for ablation studies
    (e.g. the supply-side-only feature set).
    """
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_FACTORIES)}")

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    regressor = MODEL_FACTORIES[model_name](random_state=random_state, **model_kwargs)

    return Pipeline([
        ("preprocess", preprocessor),
        ("regressor", regressor),
    ])


def get_feature_engineered_pipeline(
    model_name: str,
    random_state: int = 42,
    numeric_features=None,
    categorical_features=None,
    top_n_countries: int | None = None,
    **model_kwargs,
) -> Pipeline:
    """The full pipeline used for training and serving:

        raw listing columns -> FeatureEngineer -> ColumnTransformer -> regressor

    Because FeatureEngineer sits *inside* the Pipeline, GridSearchCV /
    cross_val_score re-fit it on every training fold (city means and top
    countries never see held-out rows), and the saved pickle can score raw
    JSON records directly (predict.py).
    """
    top_n = top_n_countries if top_n_countries is not None else load_training_config()["top_n_countries"]
    pipe = create_pipeline(model_name, random_state=random_state,
                           numeric_features=numeric_features,
                           categorical_features=categorical_features, **model_kwargs)
    return Pipeline([("features", FeatureEngineer(top_n_countries=top_n)), *pipe.steps])
