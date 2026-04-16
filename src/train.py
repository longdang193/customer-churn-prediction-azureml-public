#!/usr/bin/env python3
"""Training entry-point for the churn prediction models."""

import argparse
from pathlib import Path

from config.runtime import load_training_runtime_defaults
from training import (
    determine_models_to_train,
    is_hpo_mode,
    prepare_regular_hyperparams,
    train_pipeline_stage,
)
from utils.config_loader import get_config_value, load_config
from utils.type_utils import parse_bool

DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "train.yaml"


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(
        description='Train churn prediction models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", type=str, default=None, help="Directory with preprocessed data")
    parser.add_argument("--config", type=str, default=None, help=f"Config file (default: {DEFAULT_CONFIG})")
    parser.add_argument(
        "--model-type", type=str, default=None, choices=['logreg', 'rf', 'xgboost'],
        help='Single model type for HPO mode'
    )
    parser.add_argument("--class-weight", type=str, default=None, help='Class weight strategy')
    parser.add_argument("--random-state", type=int, default=None, help='Random seed')
    parser.add_argument("--experiment-name", type=str, default=None, help='MLflow experiment name')
    parser.add_argument("--use-smote", action='store_true', help='Apply SMOTE')
    parser.add_argument("--model-artifact-dir", type=str, default=None, help='Directory to save best model')
    parser.add_argument("--parent-run-id-output", type=str, default=None, help='File to write parent run ID')
    parser.add_argument(
        "--candidate-metrics-output",
        type=str,
        default=None,
        help="File to write the best candidate metrics summary",
    )
    parser.add_argument(
        "--mlflow-model-output",
        type=str,
        default=None,
        help="Directory to write the best-model MLflow bundle",
    )
    parser.add_argument(
        "--manifest-output",
        type=str,
        default=None,
        help="Optional file or directory path to write the training step manifest JSON.",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="model.param=value",
        help="Override hyperparameters (can be used multiple times)"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config or DEFAULT_CONFIG)
    config = load_config(str(config_path)) if config_path.exists() else {}
    runtime_defaults = load_training_runtime_defaults(config_path)
    
    training_config = get_config_value(config, 'training', {})
    lineage_config = get_config_value(config, "lineage", {})
    
    # Apply training-level parameter overrides from --set (e.g., use_smote, class_weight, random_state)
    # These are parameters that don't have a model prefix (unlike model.param=value)
    from training.hyperparams import parse_override_value
    training_level_params = ['use_smote', 'class_weight', 'random_state']
    model_param_overrides = []  # Filter out training-level params for model hyperparams
    for override in args.set:
        if '=' in override:
            try:
                key, value_str = override.split('=', 1)
                if key in training_level_params:
                    # Parse the value (handles booleans, ints, strings, etc.)
                    parsed_value = parse_override_value(value_str)
                    training_config[key] = parsed_value
                else:
                    # Keep model-specific overrides (model.param=value format)
                    model_param_overrides.append(override)
            except (ValueError, AttributeError):
                # Keep invalid overrides - they'll be handled by apply_param_overrides
                model_param_overrides.append(override)
        else:
            # Keep overrides without '=' (shouldn't happen, but be safe)
            model_param_overrides.append(override)
    
    hpo_mode = is_hpo_mode(args.model_type)
    hyperparams_by_model = prepare_regular_hyperparams(training_config, model_param_overrides)
    
    models_to_train = determine_models_to_train(hpo_mode, args.model_type, training_config)
    resolved_class_weight = (
        args.class_weight
        if args.class_weight is not None
        else str(training_config.get("class_weight", runtime_defaults.class_weight))
    )
    resolved_random_state = (
        args.random_state
        if args.random_state is not None
        else int(training_config.get("random_state", runtime_defaults.random_state))
    )
    resolved_use_smote = (
        True
        if args.use_smote
        else parse_bool(
            training_config.get("use_smote", runtime_defaults.use_smote),
            default=runtime_defaults.use_smote,
        )
    )
    
    train_pipeline_stage(
        data_dir=args.data or 'data/processed',
        models=models_to_train,
        class_weight=resolved_class_weight,
        random_state=resolved_random_state,
        experiment_name=args.experiment_name or runtime_defaults.experiment_name,
        use_smote=resolved_use_smote,
        hyperparams_by_model=hyperparams_by_model,
        model_artifact_dir=args.model_artifact_dir,
        parent_run_id_output=args.parent_run_id_output,
        candidate_metrics_output=args.candidate_metrics_output,
        mlflow_model_output=args.mlflow_model_output,
        manifest_output_path=args.manifest_output,
        config_path=str(config_path),
        canonical_train_config=(
            str(lineage_config.get("canonical_train_config"))
            if lineage_config.get("canonical_train_config")
            else None
        ),
    )


if __name__ == '__main__':
    main()
