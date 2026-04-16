"""Path resolution utilities."""

from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the project root directory.
    
    Assumes this file is in src/utils/, so project root is 2 levels up.
    
    Returns:
        Path to project root directory
    """
    return Path(__file__).parents[2]


def get_config_env_path(config_path: Optional[str] = None) -> Path:
    """Get the path to config.env file.
    
    Args:
        config_path: Path to config.env file (default: "config.env" in project root)
        
    Returns:
        Path to config.env file
    """
    if config_path is None:
        return get_project_root() / "config.env"
    return Path(config_path)

