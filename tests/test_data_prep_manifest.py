"""
@meta
name: test_data_prep_manifest
type: test
scope: unit
domain: tests
covers:
  - tests/test_data_prep_manifest.py
tags:
  - ci-safe
lifecycle:
  status: active
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import pandas as pd


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_prepare_data_writes_step_manifest() -> None:
    """
    @proves data-prep.encode-categoricals-scale-numerics-split-dataset-train-test
    @proves data-prep.emit-processed-csv-artifacts-plus-metadata-scaler-encoder
    @proves data-prep.emit-structured-step-manifest-json-artifact-validation-data
    """
    from src.data_prep import prepare_data

    temp_dir = _make_temp_dir()
    try:
        input_path = temp_dir / "churn.csv"
        output_dir = temp_dir / "processed"
        manifest_output_path = temp_dir / "data_prep_manifest.json"
        pd.DataFrame(
            [
                {
                    "RowNumber": 1,
                    "CustomerId": 1001,
                    "Surname": "A",
                    "CreditScore": 600,
                    "Geography": "France",
                    "Gender": "Female",
                    "Age": 40,
                    "Tenure": 3,
                    "Balance": 0.0,
                    "NumOfProducts": 1,
                    "HasCrCard": 1,
                    "IsActiveMember": 1,
                    "EstimatedSalary": 50000.0,
                    "Exited": 0,
                },
                {
                    "RowNumber": 2,
                    "CustomerId": 1002,
                    "Surname": "B",
                    "CreditScore": 720,
                    "Geography": "Spain",
                    "Gender": "Male",
                    "Age": 50,
                    "Tenure": 5,
                    "Balance": 1000.0,
                    "NumOfProducts": 2,
                    "HasCrCard": 1,
                    "IsActiveMember": 0,
                    "EstimatedSalary": 70000.0,
                    "Exited": 1,
                },
                {
                    "RowNumber": 3,
                    "CustomerId": 1003,
                    "Surname": "C",
                    "CreditScore": 650,
                    "Geography": "France",
                    "Gender": "Female",
                    "Age": 35,
                    "Tenure": 7,
                    "Balance": 1500.0,
                    "NumOfProducts": 2,
                    "HasCrCard": 0,
                    "IsActiveMember": 1,
                    "EstimatedSalary": 80000.0,
                    "Exited": 0,
                },
                {
                    "RowNumber": 4,
                    "CustomerId": 1004,
                    "Surname": "D",
                    "CreditScore": 710,
                    "Geography": "Spain",
                    "Gender": "Male",
                    "Age": 45,
                    "Tenure": 6,
                    "Balance": 2200.0,
                    "NumOfProducts": 1,
                    "HasCrCard": 1,
                    "IsActiveMember": 1,
                    "EstimatedSalary": 90000.0,
                    "Exited": 1,
                },
            ]
        ).to_csv(input_path, index=False)

        summary = prepare_data(
            input_path=input_path,
            output_dir=output_dir,
            test_size=0.5,
            random_state=42,
            target_col="Exited",
            columns_to_remove=["RowNumber", "CustomerId", "Surname"],
            categorical_cols=["Geography", "Gender"],
            stratify=True,
            config_path=temp_dir / "data_smoke.yaml",
            execution_mode="smoke_preflight",
            manifest_output_path=manifest_output_path,
        )

        assert summary["train_row_count"] == 2
        manifest = json.loads((output_dir / "step_manifest.json").read_text(encoding="utf-8"))
        mirrored_manifest = json.loads(manifest_output_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "success"
        assert manifest["run_context"]["execution_mode"] == "smoke_preflight"
        assert manifest["metrics"]["feature_count"] >= 1
        assert manifest["metrics"]["raw_input_column_count"] == 14
        assert manifest["metrics"]["post_drop_column_count"] == 11
        assert manifest["step_specific"]["dropped_columns"] == ["RowNumber", "CustomerId", "Surname"]
        assert manifest == mirrored_manifest
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_prepare_data_loads_multiple_csv_files_from_directory() -> None:
    """
    @proves data-prep.load-one-more-csv-files-configured-raw-data
    """
    from src.data_prep import prepare_data

    temp_dir = _make_temp_dir()
    try:
        input_dir = temp_dir / "raw"
        output_dir = temp_dir / "processed"
        input_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "RowNumber": 1,
                "CustomerId": 1001,
                "Surname": "A",
                "CreditScore": 600,
                "Geography": "France",
                "Gender": "Female",
                "Age": 40,
                "Tenure": 3,
                "Balance": 0.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000.0,
                "Exited": 0,
            },
            {
                "RowNumber": 2,
                "CustomerId": 1002,
                "Surname": "B",
                "CreditScore": 720,
                "Geography": "Spain",
                "Gender": "Male",
                "Age": 50,
                "Tenure": 5,
                "Balance": 1000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 0,
                "EstimatedSalary": 70000.0,
                "Exited": 1,
            },
        ]
        pd.DataFrame(rows).to_csv(input_dir / "part_a.csv", index=False)
        pd.DataFrame(rows).to_csv(input_dir / "part_b.csv", index=False)

        summary = prepare_data(
            input_path=input_dir,
            output_dir=output_dir,
            test_size=0.5,
            random_state=42,
            target_col="Exited",
            columns_to_remove=["RowNumber", "CustomerId", "Surname"],
            categorical_cols=["Geography", "Gender"],
            stratify=True,
        )

        assert summary["input_row_count"] == 4
        assert summary["source_files"] == ["part_a.csv", "part_b.csv"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
