"""
@meta
name: azureml_client
type: utility
domain: azure-ml
responsibility:
  - Build a shared Azure ML client from env-backed runtime config.
  - Keep AML credential and workspace wiring out of entrypoint scripts.
inputs:
  - config.env path
outputs:
  - Configured MLClient instance
tags:
  - azure-ml
  - client
  - orchestration
features:
  - model-training-pipeline
  - notebook-hpo
  - online-endpoint-deployment
capabilities:
  - fixed-train.use-azureml-adapters
  - hpo.reuse-shared-src-azureml-client-input-adapters-instead
  - online-deploy.reuse-shared-src-azureml-registry-deployment-adapters-release
lifecycle:
  status: active
"""

from __future__ import annotations

from azure.ai.ml import MLClient  # type: ignore[import-not-found]
from azure.identity import DefaultAzureCredential  # type: ignore[import-not-found]

from src.config.runtime import load_azure_config


def get_ml_client(config_path: str | None = None) -> MLClient:
    """Build an MLClient from the configured workspace coordinates."""
    azure_config = load_azure_config(config_path)
    return MLClient(
        DefaultAzureCredential(),
        subscription_id=azure_config["subscription_id"],
        resource_group_name=azure_config["resource_group"],
        workspace_name=azure_config["workspace_name"],
    )
