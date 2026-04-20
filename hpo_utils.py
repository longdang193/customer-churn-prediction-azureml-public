"""
@meta
name: hpo_utils
type: utility
domain: hpo
responsibility:
  - Load HPO config files from the project config surface.
  - Normalize the HPO parameter space before sweep submission.
inputs:
  - HPO YAML config path
outputs:
  - Loaded HPO config mapping
  - Null-filtered parameter space
tags:
  - hpo
  - config
  - utility
capabilities:
  - hpo.keep-hpo-utils-py-supporting-shared-utility-code
lifecycle:
  status: active
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml  # type: ignore[import-untyped]  # PyYAML stub package is not installed in this repo.

# Resolve relative to project root so the module works from any CWD (e.g., notebooks/)
PROJECT_ROOT = Path(__file__).resolve().parents[0]
CONFIG_PATH = PROJECT_ROOT / "configs" / "hpo.yaml"


def load_hpo_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load HPO configuration from the selected config path."""
    resolved_config_path = Path(config_path) if config_path else CONFIG_PATH
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"HPO config not found at {resolved_config_path}")
    with resolved_config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return config


def _resolve_model_types(search_space: Dict[str, Any]) -> List[str]:
    configured_types = search_space.get("model_types")
    if configured_types:
        return list(configured_types)
    inferred: List[str] = []
    for candidate in ("logreg", "rf", "xgboost"):
        if candidate in search_space:
            inferred.append(candidate)
    return inferred


def _filter_nulls(obj: Any) -> Any:
    """Recursively filter out None/null values from lists in the search space.
    
    Azure ML sweep Choice does not accept None/null values.
    """
    if isinstance(obj, dict):
        return {key: _filter_nulls(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Filter out None values from lists
        filtered = [item for item in obj if item is not None]
        return [_filter_nulls(item) for item in filtered]
    else:
        return obj


def build_parameter_space(search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Build the sweep search space from YAML config, filtering out null values.
    
    Azure ML sweep Choice does not accept None/null values, so they are removed.
    """
    return _filter_nulls(search_space)
