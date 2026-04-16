"""Configuration loader for YAML config files."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file and ensure it is a mapping."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, Mapping):
        raise ValueError(
            f"Configuration at {config_path} must be a mapping, "
            f"got {type(config).__name__}."
        )
    
    return dict(config)


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get nested configuration value using dot notation."""
    if not isinstance(config, Mapping):
        return default

    keys = key_path.split(".")
    value: Any = config
    
    for key in keys:
        if isinstance(value, Mapping) and key in value:
            value = value[key]
        else:
            return default
    
    return value

