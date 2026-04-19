"""Model-factory helpers owned by the models domain.

@meta
name: factory
type: module
domain: models
responsibility:
  - Provide models behavior for `src/models/factory.py`.
inputs: []
outputs: []
tags:
  - models
lifecycle:
  status: active
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from .logistic_regression import get_logistic_regression
from .random_forest import get_random_forest
from .xgboost_model import get_xgboost


JSONDict = dict[str, Any]
MODEL_FACTORY = {
    "logreg": get_logistic_regression,
    "rf": get_random_forest,
    "xgboost": get_xgboost,
}


def get_model(
    model_name: str,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
) -> Any:
    """Return a configured estimator instance for the requested model family."""
    if model_name not in MODEL_FACTORY:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: {list(MODEL_FACTORY.keys())}"
        )
    return MODEL_FACTORY[model_name](
        class_weight=class_weight,
        random_state=random_state,
    )


def apply_hyperparameters(model: Any, hyperparams: Optional[JSONDict]) -> tuple[Any, JSONDict]:
    """Apply valid hyperparameters to a model and report the accepted subset."""
    if not hyperparams or not hasattr(model, "get_params"):
        return model, {}

    valid_params = model.get_params(deep=True)
    applied = {key: value for key, value in hyperparams.items() if key in valid_params}
    min_samples_split = applied.get("min_samples_split")
    if isinstance(min_samples_split, (int, float)) and min_samples_split < 2:
        applied["min_samples_split"] = 2

    if not applied:
        return model, {}

    try:
        model.set_params(**applied)
    except Exception:
        return model, {}

    return model, applied


def apply_class_weight_adjustments(
    model_name: str,
    model: Any,
    y_train: pd.Series,
    class_weight: Optional[str],
    tuned_params: JSONDict,
) -> JSONDict:
    """Adjust XGBoost weighting when the balanced mode is active."""
    if model_name != "xgboost" or class_weight != "balanced":
        return {}
    if "scale_pos_weight" in tuned_params:
        return {}
    if not hasattr(model, "get_params"):
        return {}

    params = model.get_params()
    if "scale_pos_weight" not in params:
        return {}

    positives = (y_train == 1).sum()
    negatives = (y_train == 0).sum()
    if positives == 0 or negatives == 0:
        return {}

    current = params.get("scale_pos_weight")
    if current not in (None, 0, 1):
        return {}

    scale_pos_weight = float(negatives / positives)
    try:
        model.set_params(scale_pos_weight=scale_pos_weight)
    except Exception:
        return {}

    return {"scale_pos_weight": scale_pos_weight}
