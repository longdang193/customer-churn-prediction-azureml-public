"""Hyperparameter loading, parsing, and management utilities.

@meta
name: hyperparams
type: module
domain: training
responsibility:
  - Provide training behavior for `src/training/hyperparams.py`.
inputs: []
outputs: []
tags:
  - training
lifecycle:
  status: active
"""

import ast
from typing import Any, Dict, Optional

from utils.config_loader import get_config_value

JSONDict = Dict[str, Any]
DEFAULT_MODEL_SEQUENCE = ("logreg", "rf", "xgboost")


def parse_override_value(value: str) -> Any:
    """Parse scalar override values from CLI arguments.
    
    Args:
        value: String value to parse
        
    Returns:
        Parsed value (int, float, bool, None, or string)
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered == "none":
            return None
        return value


def apply_param_overrides(overrides: list[str], hyperparams: Dict[str, JSONDict]) -> Dict[str, JSONDict]:
    """Apply CLI overrides to hyperparameter dictionary.
    
    Args:
        overrides: List of override strings in format "model.param=value"
        hyperparams: Base hyperparameter dictionary
        
    Returns:
        Updated hyperparameter dictionary
        
    Raises:
        ValueError: If override format is invalid
    """
    if not overrides:
        return hyperparams or {}
    
    updated: Dict[str, JSONDict] = {model: dict(params) for model, params in (hyperparams or {}).items()}
    for assignment in overrides:
        try:
            model_part, value_part = assignment.split("=", 1)
            model_name, param_name = model_part.split(".", 1)
            value = parse_override_value(value_part)
            updated.setdefault(model_name, {})[param_name] = value
        except ValueError as exc:
            raise ValueError(f"Invalid override format '{assignment}'. Use model.param=value") from exc
    
    return updated


def is_hpo_mode(model_type: Optional[str] = None) -> bool:
    """Check if running in HPO mode.
    
    Args:
        model_type: Optional model type (indicates HPO if provided)
        
    Returns:
        True if in HPO mode, False otherwise
    """
    return model_type is not None


def prepare_regular_hyperparams(training_config: Dict[str, Any], param_overrides: list[str]) -> Dict[str, JSONDict]:
    """Prepare hyperparameters for regular mode (from config + CLI overrides)."""
    hyperparams_by_model = get_config_value(training_config, "hyperparameters", {})
    if param_overrides:
        hyperparams_by_model = apply_param_overrides(param_overrides, hyperparams_by_model)
    return hyperparams_by_model


def determine_models_to_train(
    is_hpo: bool,
    model_type: Optional[str],
    training_config: Dict[str, Any],
) -> list[str]:
    """Determine which models to train based on mode."""
    if is_hpo and model_type:
        return [model_type]
    return get_config_value(training_config, "models", list(DEFAULT_MODEL_SEQUENCE))

