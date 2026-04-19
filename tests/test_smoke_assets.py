"""
@meta
type: test
scope: unit
domain: smoke-assets
covers:
  - Repo-backed smoke data schema and class balance
  - Negative-path smoke fixture intent
  - Smoke config and deployment payload shape
excludes:
  - Real Azure ML job submission
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_COLUMNS = [
    "RowNumber",
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
]
MODEL_INPUT_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


def test_positive_smoke_data_matches_raw_schema_and_supports_stratified_split() -> None:
    """
    @proves data-prep.provide-positive-negative-smoke-fixtures-local-data-prep
    """
    smoke_path = PROJECT_ROOT / "data" / "smoke" / "positive" / "churn_smoke.csv"

    frame = pd.read_csv(smoke_path)

    assert list(frame.columns) == RAW_COLUMNS
    assert len(frame) >= 20
    assert set(frame["Exited"].unique()) == {0, 1}
    assert frame["Exited"].value_counts().min() >= 5


def test_negative_smoke_data_has_documented_validation_gate_difference() -> None:
    """
    @proves data-prep.provide-positive-negative-smoke-fixtures-local-data-prep
    """
    edge_path = PROJECT_ROOT / "data" / "smoke" / "negative" / "churn_validation_edge.csv"

    frame = pd.read_csv(edge_path)

    assert list(frame.columns) == RAW_COLUMNS
    assert frame["CreditScore"].isna().mean() > 0.2


def test_data_smoke_config_points_at_smoke_paths() -> None:
    """
    @proves data-prep.accept-selected-data-config-files-through-aml-validation
    @proves data-prep.provide-positive-negative-smoke-fixtures-local-data-prep
    """
    config = load_config(str(PROJECT_ROOT / "configs" / "data_smoke.yaml"))

    assert config["data"]["input_path"] == "data/smoke/positive/churn_smoke.csv"
    assert config["data"]["output_dir"] == "data/processed_smoke"
    assert config["data"]["target_column"] == "Exited"
    assert config["data"]["stratify"] is True
    assert config["validation"]["fail_on_missing_fraction"] is True


def test_positive_smoke_asset_directory_contains_only_positive_fixture() -> None:
    smoke_asset_dir = PROJECT_ROOT / "data" / "smoke" / "positive"

    assert smoke_asset_dir.is_dir()
    csv_files = sorted(path.name for path in smoke_asset_dir.glob("*.csv"))

    assert csv_files == ["churn_smoke.csv"]


def test_negative_smoke_asset_directory_contains_only_validation_edge_fixture() -> None:
    smoke_asset_dir = PROJECT_ROOT / "data" / "smoke" / "negative"

    assert smoke_asset_dir.is_dir()
    csv_files = sorted(path.name for path in smoke_asset_dir.glob("*.csv"))

    assert csv_files == ["churn_validation_edge.csv"]


def test_train_smoke_config_uses_smoke_scoped_training_and_promotion_policy() -> None:
    config = load_config(str(PROJECT_ROOT / "configs" / "train_smoke.yaml"))

    assert config["training"]["experiment_name"] == "train-smoke"
    assert config["training"]["models"] == ["logreg"]
    assert config["training"]["use_smote"] == "false"
    assert config["promotion"]["minimum_candidate_score"] == 0.0


def test_sample_data_payload_matches_aml_endpoint_request_shape() -> None:
    payload = json.loads((PROJECT_ROOT / "sample-data.json").read_text(encoding="utf-8"))

    assert sorted(payload.keys()) == ["input_data"]
    assert isinstance(payload["input_data"], list)
    assert len(payload["input_data"]) == 1
    assert len(payload["input_data"][0]) == len(MODEL_INPUT_COLUMNS)
    assert all(isinstance(value, (int, float)) for value in payload["input_data"][0])
