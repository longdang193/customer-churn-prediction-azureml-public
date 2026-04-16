"""Data-domain configuration helpers for preprocessing and validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils.config_loader import get_config_value, load_config
from utils.path_utils import get_project_root
from utils.type_utils import parse_bool


DEFAULT_DATA_CONFIG = get_project_root() / "configs" / "data.yaml"
DEFAULT_COLUMNS_TO_REMOVE = ("RowNumber", "CustomerId", "Surname")
DEFAULT_CATEGORICAL_COLUMNS = ("Geography", "Gender")


def get_data_prep_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load data-prep config from file and merge with CLI arguments."""
    config_path = Path(args.config or DEFAULT_DATA_CONFIG)
    config = load_config(str(config_path)) if config_path.exists() else {}
    data_config = get_config_value(config, "data", {}) or {}
    stratify = parse_bool(data_config.get("stratify", True), default=True)

    return {
        "input_path": Path(args.input or get_config_value(data_config, "input_path", "data/churn.csv")),
        "output_dir": Path(args.output or get_config_value(data_config, "output_dir", "data/processed")),
        "test_size": float(args.test_size or get_config_value(data_config, "test_size", 0.2)),
        "random_state": int(args.random_state or get_config_value(data_config, "random_state", 42)),
        "target_col": args.target or get_config_value(data_config, "target_column", "Exited"),
        "columns_to_remove": get_config_value(
            data_config,
            "columns_to_remove",
            DEFAULT_COLUMNS_TO_REMOVE,
        ),
        "categorical_cols": get_config_value(
            data_config,
            "categorical_columns",
            DEFAULT_CATEGORICAL_COLUMNS,
        ),
        "stratify": stratify,
    }
