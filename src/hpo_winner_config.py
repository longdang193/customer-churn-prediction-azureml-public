"""Build fixed training configs from HPO winner manifests.

@meta
name: hpo_winner_config
type: module
domain: hpo
responsibility:
  - Provide hpo behavior for `src/hpo_winner_config.py`.
inputs: []
outputs: []
tags:
  - hpo
lifecycle:
  status: active
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


TRAIN_CONFIG_FILENAME = "train_config.yaml"
TRAINING_PARAM_KEYS = ("class_weight", "random_state", "use_smote")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config object from disk."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def coerce_scalar(value: Any) -> Any:
    """Coerce common YAML-like scalar strings from Azure ML command inputs."""
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def extract_winner_hyperparameters(
    winner_family: str,
    hpo_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Extract unprefixed hyperparameters for the selected HPO family."""
    raw_params = hpo_manifest.get("params", {}).get("hyperparameters", {}) or {}
    prefix = f"{winner_family}_"
    normalized: dict[str, Any] = {}
    for key, value in raw_params.items():
        key_str = str(key)
        if key_str.startswith(prefix):
            normalized[key_str[len(prefix) :]] = coerce_scalar(value)
    if not normalized:
        raise RuntimeError(
            f"No winner hyperparameters found in HPO manifest for family '{winner_family}'."
        )
    return normalized


def build_fixed_train_config(
    *,
    base_config: dict[str, Any],
    winner_family: str,
    hpo_manifest: dict[str, Any],
    train_manifest: dict[str, Any],
    experiment_name: str | None = None,
    display_name: str | None = None,
    canonical_train_config: str | None = None,
) -> dict[str, Any]:
    """Build a standard train config for the selected HPO winner."""
    config = dict(base_config)
    training = dict(config.get("training", {}) or {})
    promotion = dict(config.get("promotion", {}) or {})
    lineage = dict(config.get("lineage", {}) or {})
    manifest_params = train_manifest.get("params", {}) or {}

    training["models"] = [winner_family]
    for key in TRAINING_PARAM_KEYS:
        if key in manifest_params:
            training[key] = coerce_scalar(manifest_params[key])
    if experiment_name:
        training["experiment_name"] = experiment_name
    if display_name:
        training["display_name"] = display_name
    training["hyperparameters"] = {
        winner_family: extract_winner_hyperparameters(winner_family, hpo_manifest)
    }
    if canonical_train_config:
        lineage["canonical_train_config"] = canonical_train_config

    config["training"] = training
    config["promotion"] = promotion
    if lineage:
        config["lineage"] = lineage
    return config


def write_fixed_train_config(
    *,
    base_config: dict[str, Any],
    winner_family: str,
    hpo_manifest: dict[str, Any],
    train_manifest: dict[str, Any],
    output_dir: Path,
    experiment_name: str | None = None,
    display_name: str | None = None,
    canonical_train_config: str | None = None,
) -> Path:
    """Write a fixed train config into the canonical output folder."""
    output_path = output_dir / TRAIN_CONFIG_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_train_config = build_fixed_train_config(
        base_config=base_config,
        winner_family=winner_family,
        hpo_manifest=hpo_manifest,
        train_manifest=train_manifest,
        experiment_name=experiment_name,
        display_name=display_name,
        canonical_train_config=canonical_train_config,
    )
    output_path.write_text(
        yaml.safe_dump(fixed_train_config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return output_path
