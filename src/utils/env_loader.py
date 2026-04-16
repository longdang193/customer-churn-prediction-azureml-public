"""Environment variable loading utilities."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .path_utils import get_config_env_path


def load_env_file(config_path: Optional[str] = None, override: bool = True) -> None:
    """Load environment variables from config.env file.
    
    Args:
        config_path: Path to config.env file (default: "config.env" in project root)
        override: If True, override existing environment variables (default: True)
    """
    config_file = get_config_env_path(config_path)
    load_dotenv(str(config_file), override=override)


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get environment variable with optional default and required validation.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        required: If True, raise ValueError when variable is missing
        
    Returns:
        Environment variable value or default
        
    Raises:
        ValueError: If required is True and variable is missing
    """
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' is not set")
    return value

