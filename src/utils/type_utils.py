"""Type conversion and parsing utilities."""

from typing import Any


def parse_bool(value: Any, *, default: bool) -> bool:
    """Parse loose truthy/falsey values without relying on distutils.
    
    Args:
        value: Value to parse as boolean
        default: Default value if value is None
        
    Returns:
        Boolean value
        
    Raises:
        ValueError: If value cannot be interpreted as boolean
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False

    raise ValueError(f"Cannot interpret value '{value}' as boolean.")

