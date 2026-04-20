#!/usr/bin/env python3
"""Helper entry-point to invoke train.py with sweep-managed hyperparameters.

@meta
name: run_sweep_trial
type: module
domain: training
responsibility:
  - Provide training behavior for `src/run_sweep_trial.py`.
inputs: []
outputs: []
tags:
  - training
capabilities:
  - hpo.submit-reload-sweep-jobs-azure-ml-run-hpo
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from utils.output_paths import resolve_named_input_file
from utils.step_manifest import (
    build_step_manifest,
    finalize_manifest,
    merge_config,
    merge_section,
    resolve_manifest_output_path,
    set_failure,
)


MODEL_HYPERPARAM_KEYS = [
    "rf_n_estimators",
    "rf_max_depth",
    "rf_min_samples_split",
    "rf_min_samples_leaf",
    "rf_max_features",
    "logreg_C",
    "logreg_solver",
    "xgboost_n_estimators",
    "xgboost_max_depth",
    "xgboost_learning_rate",
    "xgboost_subsample",
    "xgboost_colsample_bytree",
]

TRAINING_PARAM_KEYS = [
    "use_smote",
    "class_weight",
    "random_state",
]

HYPERPARAM_KEYS = MODEL_HYPERPARAM_KEYS + TRAINING_PARAM_KEYS
CANDIDATE_METRICS_FILENAME = "candidate_metrics.json"


def _format_override_key(raw_key: str) -> str:
    """Convert CLI-friendly keys to train.py override format."""
    for model_prefix in ("rf", "logreg", "xgboost"):
        expected_prefix = f"{model_prefix}_"
        if raw_key.startswith(expected_prefix):
            return f"{model_prefix}.{raw_key[len(expected_prefix):]}"
    return raw_key


def _add_hyperparam_arguments(parser: argparse.ArgumentParser) -> None:
    for key in HYPERPARAM_KEYS:
        parser.add_argument(f"--{key}", default=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Invoke train.py with sweep-managed hyperparameters."
    )
    parser.add_argument("--data", required=True, help="Processed data URI/folder.")
    parser.add_argument(
        "--config",
        required=False,
        default=None,
        help="Optional training config path to forward into train.py.",
    )
    parser.add_argument(
        "--hpo-config",
        required=False,
        default=None,
        help="Optional HPO config path for manifest traceability.",
    )
    parser.add_argument(
        "--model-type",
        required=True,
        help="Model type to train (logreg|rf|xgboost).",
    )
    parser.add_argument(
        "--model-output",
        required=False,
        default=None,
        help="Directory/URI where the trained model artifacts should be stored.",
    )
    parser.add_argument(
        "--model-artifact-dir",
        required=False,
        default=None,
        help="Compatibility alias for --model-output.",
    )
    parser.add_argument(
        "--candidate-metrics-output",
        required=False,
        default=None,
        help="Optional path for the best-candidate metrics summary JSON.",
    )
    parser.add_argument(
        "--mlflow-model-output",
        required=False,
        default=None,
        help="Optional directory for the best-model MLflow bundle.",
    )
    parser.add_argument(
        "--manifest-output",
        required=False,
        default=None,
        help="Directory/URI where the training step manifest should be stored.",
    )
    parser.add_argument(
        "--hpo-manifest-output",
        required=False,
        default=None,
        help="Directory/URI where the HPO trial manifest should be stored.",
    )
    _add_hyperparam_arguments(parser)
    args = parser.parse_args()
    resolved_model_output = args.model_output or args.model_artifact_dir
    if not resolved_model_output:
        raise SystemExit("Either --model-output or --model-artifact-dir is required.")

    repo_root = Path(__file__).resolve().parent
    train_script = repo_root / "train.py"
    hpo_manifest_path = (
        resolve_manifest_output_path(Path(args.hpo_manifest_output))
        if args.hpo_manifest_output
        else None
    )
    hpo_manifest = build_step_manifest(step_name="hpo_trial", stage_name="model_sweep")
    merge_config(
        hpo_manifest,
        config_paths={
            "train_config": args.config,
            "hpo_config": args.hpo_config,
        },
        resolved={
            "model_type": args.model_type,
            "processed_data": args.data,
        },
    )
    merge_section(
        hpo_manifest,
        "outputs",
        {
            "model_output": resolved_model_output,
            "candidate_metrics_output": args.candidate_metrics_output,
            "mlflow_model_output": args.mlflow_model_output,
            "train_manifest_output": args.manifest_output,
            "hpo_manifest_output": args.hpo_manifest_output,
        },
    )
    merge_section(
        hpo_manifest,
        "params",
        {
            "model_type": args.model_type,
            "hyperparameters": {
                key: getattr(args, key)
                for key in HYPERPARAM_KEYS
                if getattr(args, key) is not None and str(getattr(args, key)).lower() != "none"
            },
        },
    )

    cli = [
        sys.executable,
        str(train_script),
        "--data",
        args.data,
        "--model-type",
        args.model_type,
        "--model-artifact-dir",
        resolved_model_output,
    ]

    if args.config:
        cli.extend(["--config", args.config])
    if args.hpo_config:
        merge_section(hpo_manifest, "inputs", {"hpo_config": args.hpo_config})
    if args.candidate_metrics_output:
        cli.extend(["--candidate-metrics-output", args.candidate_metrics_output])
    if args.mlflow_model_output:
        cli.extend(["--mlflow-model-output", args.mlflow_model_output])
    if args.manifest_output:
        cli.extend(["--manifest-output", args.manifest_output])

    for key in HYPERPARAM_KEYS:
        value = getattr(args, key)
        if value is None or str(value).lower() == "none":
            continue
        cli.extend(["--set", f"{_format_override_key(key)}={value}"])

    try:
        subprocess.run(cli, check=True)
        if args.candidate_metrics_output:
            candidate_metrics_path = resolve_named_input_file(
                Path(args.candidate_metrics_output),
                CANDIDATE_METRICS_FILENAME,
            )
            if candidate_metrics_path.exists():
                candidate_metrics = json.loads(candidate_metrics_path.read_text(encoding="utf-8"))
                merge_section(
                    hpo_manifest,
                    "metrics",
                    {
                        key: candidate_metrics[key]
                        for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
                        if key in candidate_metrics
                    },
                )
                merge_section(
                    hpo_manifest,
                    "tags",
                    {
                        "candidate_model_name": candidate_metrics.get("model_name"),
                        "candidate_run_id": candidate_metrics.get("run_id"),
                    },
                )
        merge_section(
            hpo_manifest,
            "artifacts",
            {
                "train_manifest": args.manifest_output,
                "candidate_metrics": args.candidate_metrics_output,
                "mlflow_model_output": args.mlflow_model_output,
                "model_output": resolved_model_output,
            },
        )
        if hpo_manifest_path is not None:
            finalize_manifest(hpo_manifest, output_path=hpo_manifest_path, status="success")
            print(f"HPO_MANIFEST_PATH={hpo_manifest_path}")
    except Exception as exc:
        set_failure(hpo_manifest, phase="run_sweep_trial", exc=exc)
        if hpo_manifest_path is not None:
            finalize_manifest(hpo_manifest, output_path=hpo_manifest_path, status="failed")
            print(f"HPO_MANIFEST_PATH={hpo_manifest_path}")
        raise


if __name__ == "__main__":
    main()
