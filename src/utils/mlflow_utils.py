"""MLflow utilities for run context management and Azure ML detection."""

import os
from typing import Any

import mlflow


def is_azure_ml() -> bool:
    """Check if running in Azure ML environment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if tracking_uri and "azureml" in tracking_uri.lower():
        return True
    # Fallback to explicit Azure ML env markers
    return any(
        os.getenv(var)
        for var in ("AZUREML_RUN_ID", "AZUREML_RUN_TOKEN", "AZUREML_RUN_CONFIGURATION")
    )


def get_run_id(run_obj: Any) -> str:
    """Extract run ID from MLflow run object."""
    if hasattr(run_obj, 'info'):
        return run_obj.info.run_id
    elif hasattr(run_obj, 'run_id'):
        return run_obj.run_id
    else:
        return os.getenv("MLFLOW_RUN_ID", "unknown")


def get_active_run():
    """Get the active MLflow run, creating one if needed in local mode."""
    is_azure = is_azure_ml()
    
    if is_azure:
        active_run = mlflow.active_run()
        if active_run is None:
            run_id = os.getenv("MLFLOW_RUN_ID")
            if run_id:
                active_run = mlflow.tracking.MlflowClient().get_run(run_id)
            else:
                raise RuntimeError("Could not determine MLflow run context in Azure ML")
        return active_run
    else:
        return mlflow.active_run()


def start_parent_run(experiment_name: str, run_name: str = "Churn_Training_Pipeline"):
    """Start a parent MLflow run for local execution."""
    if is_azure_ml():
        return get_active_run()
    
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def start_nested_run(run_name: str):
    """Start a nested MLflow run."""
    if is_azure_ml():
        active_run = get_active_run()
        run_id = get_run_id(active_run)
        mlflow.set_tag("model_name", run_name)
        return active_run, run_id
    else:
        nested_run = mlflow.start_run(run_name=run_name, nested=True)
        return nested_run, nested_run.info.run_id

