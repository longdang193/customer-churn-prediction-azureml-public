"""Core training logic for model training and evaluation.

@meta
name: training
type: module
domain: training
responsibility:
  - Provide training behavior for `src/training/training.py`.
inputs: []
outputs: []
tags:
  - training
capabilities:
  - fixed-train.emit-release-artifacts
  - fixed-train.emit-declared-manifest-folders
  - fixed-train.share-training-artifact-vocabulary
lifecycle:
  status: active
"""

# Import azureml.mlflow before mlflow to register Azure ML tracking store
import azureml.mlflow  # noqa: F401

import os
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
 
import joblib
import mlflow
import mlflow.sklearn

from data.data_utils import apply_smote, load_prepared_data
from models.factory import (
    apply_class_weight_adjustments,
    apply_hyperparameters,
    get_model,
)
from utils.mlflow_conda import normalize_mlflow_conda_for_azure_serving
from utils.metrics import calculate_metrics
from utils.mlflow_utils import (
    get_active_run,
    get_run_id,
    is_azure_ml,
    start_nested_run,
    start_parent_run,
)
from utils.output_paths import resolve_named_output_file
from utils.step_manifest import (
    add_warning,
    build_step_manifest,
    finalize_manifest,
    manifest_path_for_dir,
    merge_config,
    merge_section,
    set_failure,
)

JSONDict = Dict[str, Any]
CANDIDATE_METRICS_FILENAME = "candidate_metrics.json"
PARENT_RUN_ID_FILENAME = "parent_run_id.txt"


def _model_artifact_path(model_name: str) -> Path:
    """Return the intermediate artifact path for a trained model."""
    base_dir = Path(
        os.getenv("AZUREML_ARTIFACTS_DIRECTORY")
        or os.getenv("AZUREML_OUTPUT_DIRECTORY")
        or "outputs"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{model_name}_model.pkl"


def write_candidate_metrics_summary(
    results: Dict[str, JSONDict],
    output_path: Path,
) -> JSONDict:
    """Persist the best candidate metrics for downstream promotion."""
    output_path = resolve_named_output_file(output_path, CANDIDATE_METRICS_FILENAME)
    best_model_name = max(results, key=lambda model_name: results[model_name]["test_metrics"]["f1"])
    best_result = results[best_model_name]
    payload: JSONDict = {
        "model_name": best_model_name,
        "run_id": best_result["run_id"],
        **best_result["test_metrics"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _normalize_mlflow_conda_python_spec(conda_path: Path) -> None:
    """Use a conda-resolvable Python minor version for Azure ML image builds."""
    normalize_mlflow_conda_for_azure_serving(conda_path)


def save_mlflow_model_bundle(model: Any, output_dir: Path) -> None:
    """Persist a deployable MLflow model bundle for the selected best model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(sk_model=model, path=str(output_dir))
    _normalize_mlflow_conda_python_spec(output_dir / "conda.yaml")



def train_model(
    model_name: str,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    model_hyperparams: Optional[JSONDict] = None,
) -> JSONDict:
    """Train and evaluate a single model, logging to MLflow.
    
    Args:
        model_name: Model identifier ('logreg', 'rf', or 'xgboost')
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        class_weight: Class weight strategy
        random_state: Random seed
        model_hyperparams: Optional hyperparameters for the model
        
    Returns:
        Dictionary with test_metrics, run_id, and artifact_path
    """
    nested_run, run_id = start_nested_run(model_name)
    
    try:
        model = get_model(model_name, class_weight=class_weight, random_state=random_state)
        model, tuned_params = apply_hyperparameters(model, model_hyperparams)
        
        extra_params = apply_class_weight_adjustments(
            model_name, model, y_train, class_weight, tuned_params
        )
        
        # Log parameters
        params_to_log = {**tuned_params, **extra_params}
        if params_to_log:
            mlflow.log_params(params_to_log)
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)
        
        artifact_path = f"model_{model_name}"
        model_path = _model_artifact_path(model_name)
        joblib.dump(model, model_path)
        mlflow.set_tag(f"{model_name}_model_file", str(model_path))
        
        return {
            'test_metrics': test_metrics,
            'run_id': run_id,
            'artifact_path': artifact_path,
            'model': model,
        }
    finally:
        if not is_azure_ml():
            mlflow.end_run()


def train_pipeline_stage(
    data_dir: str,
    models: Iterable[str],
    class_weight: Optional[str],
    random_state: int,
    experiment_name: str,
    use_smote: bool,
    hyperparams_by_model: Optional[Dict[str, JSONDict]],
    model_artifact_dir: Optional[str] = None,
    parent_run_id_output: Optional[str] = None,
    candidate_metrics_output: Optional[str] = None,
    mlflow_model_output: Optional[str] = None,
    manifest_output_path: Optional[str] = None,
    config_path: Optional[str] = None,
    canonical_train_config: Optional[str] = None,
    execution_mode: Optional[str] = None,
) -> Dict[str, JSONDict]:
    """Orchestrate the training stage of the pipeline.

    This function prepares the dataset once, optionally applies SMOTE,
    spins up the parent MLflow run, loops through each requested model
    (delegating to ``train_model``), and captures summary artifacts.
    
    Args:
        data_dir: Directory containing preprocessed data.
        models: Iterable of model names to train.
        class_weight: Class weight strategy.
        random_state: Random seed for reproducibility.
        experiment_name: MLflow experiment name for the parent run.
        use_smote: Whether to apply SMOTE before training.
        hyperparams_by_model: Optional mapping of model name to overrides.
        model_artifact_dir: Optional directory to copy the best model into.
        parent_run_id_output: Optional file path to write the parent run ID.
        
    Returns:
        Dictionary mapping model names to their training results.
        
    Raises:
        RuntimeError: If no models are successfully trained.
    """
    is_azure = is_azure_ml()
    manifest_dir = Path(
        model_artifact_dir
        or mlflow_model_output
        or (
            os.getenv("AZUREML_ARTIFACTS_DIRECTORY")
            or os.getenv("AZUREML_OUTPUT_DIRECTORY")
            or "outputs"
        )
    )
    manifest = build_step_manifest(step_name="train", stage_name="fixed_train")
    manifest_path = manifest_path_for_dir(manifest_dir)
    resolved_execution_mode = execution_mode or ("azureml" if is_azure else "local")
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
        config_paths={
            key: value
            for key, value in {
                "train_config": config_path,
                "canonical_train_config": canonical_train_config,
            }.items()
            if value
        }
        or None,
        resolved={
            "data_dir": data_dir,
            "experiment_name": experiment_name,
        },
    )
    merge_section(manifest, "inputs", {"processed_data_dir": data_dir})
    merge_section(
        manifest,
        "outputs",
        {
            "model_artifact_dir": model_artifact_dir,
            "parent_run_id_output": parent_run_id_output,
            "candidate_metrics_output": candidate_metrics_output,
            "mlflow_model_output": mlflow_model_output,
            "manifest_output_path": manifest_output_path,
        },
    )
    merge_section(
        manifest,
        "artifacts",
        {
            "step_manifest": manifest_path,
            "declared_step_manifest": manifest_output_path,
            "candidate_metrics": candidate_metrics_output,
            "mlflow_model_dir": mlflow_model_output,
        },
    )
    merge_section(
        manifest,
        "params",
        {
            "models": list(models),
            "class_weight": class_weight,
            "random_state": random_state,
            "use_smote": use_smote,
        },
    )
    started_run = False
    parent_run = None
    
    if not is_azure:
        parent_run = start_parent_run(experiment_name)
        started_run = True
    else:
        parent_run = get_active_run()
    
    try:
        merge_section(
            manifest,
            "run_context",
            {
                "run_id": get_run_id(parent_run) if parent_run else os.getenv("MLFLOW_RUN_ID"),
            },
        )
        mlflow.log_params({"use_smote": use_smote, "class_weight": class_weight, "random_state": random_state})
        
        X_train, X_test, y_train, y_test = load_prepared_data(data_dir)
        merge_section(
            manifest,
            "metrics",
            {
                "train_row_count_before_smote": len(X_train),
                "test_row_count": len(X_test),
                "feature_count": len(getattr(X_train, "columns", [])),
            },
        )
        if use_smote:
            X_train, y_train = apply_smote(X_train, y_train, random_state)
            class_weight = None
            add_warning(manifest, "SMOTE applied; class_weight was reset to None.")
            merge_section(
                manifest,
                "metrics",
                {
                    "train_row_count_after_smote": len(X_train),
                },
            )
        
        results = {}
        for model_name in models:
            try:
                model_hps = (hyperparams_by_model or {}).get(model_name, {})
                result = train_model(model_name, X_train, X_test, y_train, y_test, class_weight, random_state, model_hps)
                results[model_name] = result
            except Exception as e:
                raise RuntimeError(f"Error training {model_name}: {e}") from e
        
        if not results:
            raise RuntimeError("No models were successfully trained. Check earlier errors/logs.")
        
        # Log metrics based on mode
        best_model_name = list(results.keys())[0] if len(results) == 1 else max(results, key=lambda m: results[m]['test_metrics']['f1'])
        best_result = results[best_model_name]
        best_metrics = best_result['test_metrics']
        best_run_id = best_result['run_id']
        merge_section(
            manifest,
            "metrics",
            {
                "best_model_f1": best_metrics["f1"],
                "best_model_roc_auc": best_metrics["roc_auc"],
            },
        )
        merge_section(
            manifest,
            "tags",
            {
                "best_model": best_model_name,
                "best_model_run_id": best_run_id,
            },
        )
        merge_section(
            manifest,
            "step_specific",
            {
                "candidate_models": sorted(results.keys()),
                "results_by_model": {
                    model_name: {
                        "run_id": result["run_id"],
                        "test_metrics": result["test_metrics"],
                    }
                    for model_name, result in results.items()
                },
            },
        )
        if len(X_test) < 20:
            add_warning(
                manifest,
                f"Very small test split ({len(X_test)} rows); evaluation metrics are not stable.",
            )
        primary_metric_values = (
            best_metrics.get("accuracy"),
            best_metrics.get("precision"),
            best_metrics.get("recall"),
            best_metrics.get("f1"),
            best_metrics.get("roc_auc"),
        )
        if len(X_test) < 20 and all(value == 1.0 for value in primary_metric_values):
            add_warning(
                manifest,
                "All primary metrics are 1.0 on a tiny test split; treat this as wiring confirmation, not model-quality evidence.",
            )
        
        if len(results) == 1:
            # HPO mode: single model
            mlflow.log_metric(f"{best_model_name}_f1", best_metrics['f1'])
            mlflow.log_metric(f"{best_model_name}_roc_auc", best_metrics['roc_auc'])
            mlflow.log_metric("f1", best_metrics['f1'])
            mlflow.log_metric("roc_auc", best_metrics['roc_auc'])
            mlflow.set_tag("model_type", best_model_name)
        else:
            # Regular mode: multiple models
            mlflow.log_metric("best_model_f1", best_metrics['f1'])
            mlflow.log_metric("best_model_roc_auc", best_metrics['roc_auc'])
            mlflow.set_tag("best_model", best_model_name)
            mlflow.set_tag("best_model_run_id", best_run_id)
        
        # Write parent run ID if requested
        if parent_run_id_output:
            run_id_path = resolve_named_output_file(
                Path(parent_run_id_output),
                PARENT_RUN_ID_FILENAME,
            )
            run_id_path.parent.mkdir(parents=True, exist_ok=True)
            run_id = get_run_id(parent_run) if parent_run else os.getenv("MLFLOW_RUN_ID", "unknown")
            run_id_path.write_text(run_id)

        if candidate_metrics_output:
            write_candidate_metrics_summary(results, Path(candidate_metrics_output))

        if mlflow_model_output:
            best_model = best_result.get("model")
            if best_model is None:
                raise RuntimeError("Best trained model instance missing for MLflow export.")
            save_mlflow_model_bundle(best_model, Path(mlflow_model_output))
        
        # Save model artifact if requested
        if model_artifact_dir:
            output_dir = Path(model_artifact_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            source_file = _model_artifact_path(best_model_name)
            dest_file = output_dir / f"{best_model_name}_model.pkl"
            if not source_file.exists():
                    raise FileNotFoundError(f"Model file not found: {source_file}")
            dest_file.write_bytes(source_file.read_bytes())
            merge_section(
                manifest,
                "artifacts",
                {
                    "best_model_file": dest_file,
                },
            )
        finalized_manifest_path = finalize_manifest(
            manifest,
            output_path=manifest_path,
            mirror_output_path=Path(manifest_output_path) if manifest_output_path else None,
            status="success",
        )
        print(f"STEP_MANIFEST_PATH={finalized_manifest_path}")
        
        return results
    except Exception as exc:
        set_failure(manifest, phase="train_pipeline_stage", exc=exc)
        finalized_manifest_path = finalize_manifest(
            manifest,
            output_path=manifest_path,
            mirror_output_path=Path(manifest_output_path) if manifest_output_path else None,
            status="failed",
        )
        print(f"STEP_MANIFEST_PATH={finalized_manifest_path}")
        raise
    finally:
        if started_run:
            mlflow.end_run()

