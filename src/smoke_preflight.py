"""
@meta
name: smoke_preflight
type: script
domain: smoke-testing
responsibility:
  - Run local smoke data prep, validation, and training before AML submission.
  - Catch smoke-fixture and dependency drifts without requiring a cloud job.
inputs:
  - configs/data_smoke.yaml
  - configs/train_smoke.yaml
outputs:
  - Local smoke-preflight artifacts
  - Summary JSON
tags:
  - smoke
  - preflight
  - local
features:
  - churn-data-preparation
capabilities:
  - data-prep.provide-positive-negative-smoke-fixtures-local-data-prep
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from argparse import Namespace
from pathlib import Path
import sys
from typing import Any

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.data.config import get_data_prep_config
from src.data_prep import prepare_data
import src.training.training as training_module
from src.training import determine_models_to_train, prepare_regular_hyperparams, train_pipeline_stage
from src.utils.config_loader import get_config_value, load_config
import src.utils.mlflow_utils as mlflow_utils
from src.validate_data import load_validation_config, run_validation

DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data_smoke.yaml"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train_smoke.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "smoke_preflight"
AZURE_MLFLOW_MARKERS = (
    "AZUREML_RUN_ID",
    "AZUREML_RUN_TOKEN",
    "AZUREML_RUN_CONFIGURATION",
)


def _build_data_args(data_config_path: Path, output_dir: Path) -> Namespace:
    return Namespace(
        input=None,
        output=str(output_dir),
        config=str(data_config_path),
        test_size=None,
        random_state=None,
        target=None,
    )


def run_smoke_preflight(
    *,
    data_config_path: Path = DEFAULT_DATA_CONFIG,
    train_config_path: Path = DEFAULT_TRAIN_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    clean: bool = False,
) -> dict[str, Any]:
    """
    Run local smoke prep, validation, and training using the smoke configs.

    @capability data-prep.provide-positive-negative-smoke-fixtures-local-data-prep
    """
    if clean and output_root.exists():
        shutil.rmtree(output_root)

    prepared_dir = output_root / "prepared"
    validation_dir = output_root / "validation"
    validation_summary_path = output_root / "validation_summary.json"
    candidate_metrics_path = output_root / "candidate_metrics.json"
    mlflow_model_dir = output_root / "mlflow_model"
    model_artifact_dir = output_root / "model_artifact"
    summary_path = output_root / "smoke_preflight_summary.json"
    tracking_dir = output_root / "mlruns"
    output_root.mkdir(parents=True, exist_ok=True)

    data_config = get_data_prep_config(_build_data_args(data_config_path, prepared_dir))
    prepare_data(
        **data_config,
        config_path=data_config_path,
        execution_mode="smoke_preflight",
    )

    validation_summary = run_validation(
        reference_path=Path(data_config["input_path"]),
        current_path=Path(data_config["input_path"]),
        output_dir=validation_dir,
        summary_path=validation_summary_path,
        config=load_validation_config(str(data_config_path)),
        config_path=data_config_path,
        execution_mode="smoke_preflight",
    )

    train_config = load_config(str(train_config_path)) if train_config_path.exists() else {}
    training_config = get_config_value(train_config, "training", {}) or {}
    models = determine_models_to_train(False, None, training_config)
    hyperparams_by_model = prepare_regular_hyperparams(training_config, [])

    previous_tracking_uri = mlflow.get_tracking_uri()
    previous_tracking_env = os.environ.get("MLFLOW_TRACKING_URI")
    previous_marker_values = {
        key: os.environ.get(key)
        for key in AZURE_MLFLOW_MARKERS
    }
    previous_is_azure_ml = mlflow_utils.is_azure_ml
    previous_training_is_azure_ml = training_module.is_azure_ml
    previous_start_parent_run = training_module.start_parent_run
    previous_get_active_run = training_module.get_active_run
    tracking_uri = tracking_dir.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    for key in AZURE_MLFLOW_MARKERS:
        os.environ.pop(key, None)
    mlflow_utils.is_azure_ml = lambda: False
    training_module.is_azure_ml = lambda: False
    training_module.get_active_run = lambda: mlflow.active_run()
    training_module.start_parent_run = (
        lambda experiment_name: (mlflow.set_experiment(experiment_name), mlflow.start_run(run_name="Churn_Training_Pipeline"))[1]
    )
    try:
        training_results = train_pipeline_stage(
            data_dir=str(prepared_dir),
            models=models,
            class_weight=str(training_config.get("class_weight", "balanced")),
            random_state=int(training_config.get("random_state", 42)),
            experiment_name=str(training_config.get("experiment_name", "train-smoke")),
            use_smote=str(training_config.get("use_smote", "false")).lower() == "true",
            hyperparams_by_model=hyperparams_by_model,
            model_artifact_dir=str(model_artifact_dir),
            candidate_metrics_output=str(candidate_metrics_path),
            mlflow_model_output=str(mlflow_model_dir),
            config_path=str(train_config_path),
            execution_mode="smoke_preflight",
        )
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
        mlflow_utils.is_azure_ml = previous_is_azure_ml
        training_module.is_azure_ml = previous_training_is_azure_ml
        training_module.start_parent_run = previous_start_parent_run
        training_module.get_active_run = previous_get_active_run
        if previous_tracking_env is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = previous_tracking_env
        for key, value in previous_marker_values.items():
            if value is not None:
                os.environ[key] = value

    summary = {
        "status": "passed",
        "data_config": str(data_config_path),
        "train_config": str(train_config_path),
        "prepared_dir": str(prepared_dir),
        "validation_status": validation_summary["status"],
        "trained_models": sorted(training_results.keys()),
        "candidate_metrics_path": str(candidate_metrics_path),
        "mlflow_model_dir": str(mlflow_model_dir),
        "model_artifact_dir": str(model_artifact_dir),
        "step_manifests": {
            "data_prep": str(prepared_dir / "step_manifest.json"),
            "validate_data": str(validation_dir / "step_manifest.json"),
            "train": str(model_artifact_dir / "step_manifest.json"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local smoke preflight for prep, validation, and training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG), help="Smoke data config")
    parser.add_argument("--train-config", default=str(DEFAULT_TRAIN_CONFIG), help="Smoke train config")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output directory")
    parser.add_argument("--clean", action="store_true", help="Delete the previous output directory first")
    args = parser.parse_args()

    summary = run_smoke_preflight(
        data_config_path=Path(args.data_config),
        train_config_path=Path(args.train_config),
        output_root=Path(args.output_root),
        clean=args.clean,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
