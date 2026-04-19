"""
@meta
name: azureml_registry
type: utility
domain: azure-ml
responsibility:
  - Bridge Azure ML model registry operations with pure release helper logic.
  - Resolve approved baselines and register promoted MLflow model outputs.
inputs:
  - MLClient model operations
  - Candidate metrics and promotion decision payloads
outputs:
  - Baseline metrics files
  - Registered Azure ML model assets
tags:
  - azure-ml
  - registry
  - release
features:
  - online-endpoint-deployment
capabilities:
  - online-deploy.register-selected-promoted-model-bundle-azure-ml
  - online-deploy.stamp-approved-registered-models-lightweight-data-config-component
lifecycle:
  status: active
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from azure.ai.ml.constants import AssetTypes  # type: ignore[import-not-found]
from azure.ai.ml.entities import Model  # type: ignore[import-not-found]
from azure.core.exceptions import ResourceNotFoundError  # type: ignore[import-not-found]

from src.release import (
    build_baseline_metrics_payload,
    build_job_output_uri,
    build_registered_model_tags,
    select_latest_approved_model_baseline,
    select_reusable_registered_model,
)


def write_registry_backed_baseline_file(
    ml_client: object,
    *,
    model_name: str,
    output_path: Path,
) -> dict[str, object]:
    """Resolve the latest approved registry model into a baseline metrics file."""
    model_operations = getattr(ml_client, "models")
    try:
        registered_models = list(model_operations.list(name=model_name))
    except ResourceNotFoundError:
        registered_models = []

    approved_baseline = select_latest_approved_model_baseline(registered_models)
    payload = build_baseline_metrics_payload(approved_baseline)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def register_promoted_model(
    ml_client: object,
    *,
    job_name: str,
    model_name: str,
    candidate_metrics: Mapping[str, object],
    promotion_decision: Mapping[str, object],
    lineage_tags: Mapping[str, object] | None = None,
) -> object:
    """Register the promoted MLflow model output in Azure ML."""
    model_asset = Model(
        name=model_name,
        path=build_job_output_uri(job_name, "mlflow_model"),
        type=AssetTypes.MLFLOW_MODEL,
        description=f"Promoted churn model from Azure ML job {job_name}",
        tags=build_registered_model_tags(
            candidate_metrics=candidate_metrics,
            promotion_decision=promotion_decision,
            job_name=job_name,
            lineage_tags=lineage_tags,
        ),
    )
    model_operations = getattr(ml_client, "models")
    return model_operations.create_or_update(model_asset)


def find_reusable_registered_model(
    ml_client: object,
    *,
    model_name: str,
    job_name: str,
    effective_lineage: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
) -> object | None:
    """Find an existing approved model version that already represents this release."""
    model_operations = getattr(ml_client, "models")
    try:
        registered_models = list(model_operations.list(name=model_name))
    except ResourceNotFoundError:
        return None

    return select_reusable_registered_model(
        registered_models,
        source_job_name=job_name,
        effective_lineage=effective_lineage,
        candidate_metrics=candidate_metrics,
    )
