#!/usr/bin/env python3
"""Data preparation CLI for the churn dataset.

@meta
name: data_prep
type: module
domain: data-prep
responsibility:
  - Provide data-prep behavior for `src/data_prep.py`.
inputs: []
outputs: []
tags:
  - data-prep
features:
  - churn-data-preparation
capabilities:
  - data-prep.accept-selected-data-config-files-through-aml-validation
  - data-prep.encode-categoricals-scale-numerics-split-dataset-train-test
  - data-prep.emit-processed-csv-artifacts-plus-metadata-scaler-encoder
  - data-prep.emit-structured-step-manifest-json-artifact-validation-data
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from data.config import DEFAULT_DATA_CONFIG, get_data_prep_config
from data import encode_categoricals, load_data, remove_columns, save_artifacts, save_preprocessed_data, scale_features
from utils.mlflow_utils import is_azure_ml
from utils.step_manifest import (
    add_warning,
    build_step_manifest,
    finalize_manifest,
    manifest_path_for_dir,
    merge_config,
    merge_section,
    set_failure,
)


def prepare_data(
    *,
    input_path: Path,
    output_dir: Path,
    test_size: float,
    random_state: int,
    target_col: str,
    columns_to_remove: Iterable[str],
    categorical_cols: Iterable[str],
    stratify: bool,
    config_path: Path | None = None,
    execution_mode: str | None = None,
    validation_summary_path: Path | None = None,
    manifest_output_path: Path | None = None,
) -> dict[str, object]:
    """Execute the end-to-end preprocessing pipeline.

    @capability data-prep.encode-categoricals-scale-numerics-split-dataset-train-test
    @capability data-prep.emit-processed-csv-artifacts-plus-metadata-scaler-encoder
    @capability data-prep.emit-structured-step-manifest-json-artifact-validation-data
    
    Args:
        input_path: Path to input CSV file or directory containing CSV file(s)
        output_dir: Directory to save preprocessed data and artifacts
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        target_col: Name of the target column
        columns_to_remove: Iterable of column names to remove
        categorical_cols: Iterable of categorical column names to encode
        stratify: Whether to stratify the train-test split
        
    Raises:
        ValueError: If target column is not present in data
    """
    print(f"{'=' * 70}\nDATA PREPARATION PIPELINE\n{'=' * 70}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_step_manifest(step_name="data_prep", stage_name="data_prep")
    manifest_path = manifest_path_for_dir(output_dir)
    resolved_execution_mode = execution_mode or ("azureml" if is_azure_ml() else "local")
    is_smoke = "smoke" in str(config_path or "").lower()
    merge_section(
        manifest,
        "run_context",
        {
            "execution_mode": resolved_execution_mode,
            "is_smoke": is_smoke,
        },
    )
    merge_config(
        manifest,
        config_paths={"data_config": config_path} if config_path else None,
        resolved={
            "test_size": test_size,
            "random_state": random_state,
            "target_column": target_col,
            "stratify": stratify,
        },
    )
    merge_section(
        manifest,
        "inputs",
        {
            "input_path": input_path,
            "columns_to_remove": list(columns_to_remove),
            "categorical_columns": list(categorical_cols),
            "validation_summary_path": validation_summary_path,
        },
    )
    merge_section(
        manifest,
        "outputs",
        {
            "output_dir": output_dir,
            "manifest_output_path": manifest_output_path,
        },
    )
    merge_section(
        manifest,
        "artifacts",
        {
            "x_train": output_dir / "X_train.csv",
            "x_test": output_dir / "X_test.csv",
            "y_train": output_dir / "y_train.csv",
            "y_test": output_dir / "y_test.csv",
            "encoders": output_dir / "encoders.pkl",
            "scaler": output_dir / "scaler.pkl",
            "metadata": output_dir / "metadata.json",
            "step_manifest": manifest_path,
            "declared_step_manifest": manifest_output_path,
        },
    )

    try:
        df = load_data(input_path)
        raw_column_count = len(df.columns)
        discovered_csv_files = (
            sorted(path.name for path in input_path.glob("*.csv"))
            if input_path.is_dir()
            else [input_path.name]
        )
        df, columns_removed = remove_columns(df, columns_to_remove)
        post_drop_column_count = len(df.columns)

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not present in data.")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None
        )

        X_train_encoded, encoders = encode_categoricals(X_train, categorical_cols=categorical_cols)
        X_test_encoded, _ = encode_categoricals(
            X_test,
            categorical_cols=categorical_cols,
            encoders=encoders,
        )

        encoded_categorical_cols = [col for col in categorical_cols if col in X_train_encoded.columns]

        X_train_scaled, scaler, scaled_numeric_cols = scale_features(
            X_train_encoded,
            exclude_cols=encoded_categorical_cols,
        )
        X_test_scaled, _, _ = scale_features(
            X_test_encoded,
            scaler=scaler,
            columns=scaled_numeric_cols,
        )

        save_preprocessed_data(
            output_dir,
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
        )

        metadata = {
            "feature_names": list(X_train_scaled.columns),
            "target_name": target_col,
            "n_train": len(X_train_scaled),
            "n_test": len(X_test_scaled),
            "categorical_encoded_columns": encoded_categorical_cols,
            "scaled_numeric_columns": list(scaled_numeric_cols),
            "dropped_columns": columns_removed,
        }
        save_artifacts(output_dir, encoders=encoders, scaler=scaler, metadata=metadata)

        summary = {
            "input_row_count": len(df),
            "input_column_count": len(df.columns),
            "train_row_count": len(X_train_scaled),
            "test_row_count": len(X_test_scaled),
            "feature_count": len(X_train_scaled.columns),
            "train_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean()),
            "dropped_columns": list(columns_removed),
            "categorical_encoded_columns": encoded_categorical_cols,
            "scaled_numeric_columns": list(scaled_numeric_cols),
            "source_files": discovered_csv_files,
        }
        merge_section(
            manifest,
            "metrics",
            {
                "input_row_count": summary["input_row_count"],
                "raw_input_column_count": raw_column_count,
                "post_drop_column_count": post_drop_column_count,
                "source_file_count": len(discovered_csv_files),
                "train_row_count": summary["train_row_count"],
                "test_row_count": summary["test_row_count"],
                "feature_count": summary["feature_count"],
                "train_churn_rate": summary["train_churn_rate"],
                "test_churn_rate": summary["test_churn_rate"],
            },
        )
        merge_section(
            manifest,
            "params",
            {
                "test_size": test_size,
                "random_state": random_state,
                "stratify": stratify,
            },
        )
        merge_section(
            manifest,
            "step_specific",
            {
                "source_files": discovered_csv_files,
                "dropped_columns": list(columns_removed),
                "categorical_encoded_columns": encoded_categorical_cols,
                "scaled_numeric_columns": list(scaled_numeric_cols),
                "feature_names": list(X_train_scaled.columns),
            },
        )
        finalized_manifest_path = finalize_manifest(
            manifest,
            output_path=manifest_path,
            mirror_output_path=manifest_output_path,
            status="success",
        )
        print(f"STEP_MANIFEST_PATH={finalized_manifest_path}")

        print(
            f"\n{'=' * 70}\nOK DATA PREPARATION COMPLETE\n{'=' * 70}\n"
            f"Features: {len(X_train_scaled.columns)} | Train: {len(X_train_scaled)} | Test: {len(X_test_scaled)}\n"
            f"Churn rate: {y_train.mean():.2%} (train) / {y_test.mean():.2%} (test)"
        )
        return summary
    except Exception as exc:
        add_warning(manifest, "Data preparation exited before completing all outputs.")
        set_failure(manifest, phase="prepare_data", exc=exc)
        finalized_manifest_path = finalize_manifest(
            manifest,
            output_path=manifest_path,
            mirror_output_path=manifest_output_path,
            status="failed",
        )
        print(f"STEP_MANIFEST_PATH={finalized_manifest_path}")
        raise


def main() -> None:
    """
    CLI entry-point.

    @capability data-prep.accept-selected-data-config-files-through-aml-validation
    """
    parser = argparse.ArgumentParser(
        description="Prepare churn data for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=str, help="Input CSV file or directory containing CSV file(s)")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--config", type=str, help=f"Config file (default: {DEFAULT_DATA_CONFIG})")
    parser.add_argument(
        "--validation-summary",
        type=str,
        help="Optional upstream validation summary used only to gate AML execution order.",
    )
    parser.add_argument(
        "--manifest-output",
        type=str,
        help="Optional file or directory path to write the data-prep step manifest JSON.",
    )
    parser.add_argument("--test-size", type=float, help="Override test split proportion")
    parser.add_argument("--random-state", type=int, help="Override random seed")
    parser.add_argument("--target", type=str, help="Override target column name")
    args = parser.parse_args()

    config = get_data_prep_config(args)
    prepare_data(
        **config,
        config_path=Path(args.config or DEFAULT_DATA_CONFIG),
        validation_summary_path=Path(args.validation_summary) if args.validation_summary else None,
        manifest_output_path=Path(args.manifest_output) if args.manifest_output else None,
    )


if __name__ == "__main__":
    main()
