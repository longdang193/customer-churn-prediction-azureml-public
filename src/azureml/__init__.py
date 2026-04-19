"""
@meta
name: azureml_adapters
type: utility
domain: azure-ml
responsibility:
  - Provide shared Azure ML adapter helpers for orchestration entrypoints.
  - Centralize AML client, input, registry, and deployment integration points.
inputs:
  - Runtime config from env-backed settings
outputs:
  - Reusable adapter functions for pipeline, HPO, and release scripts
tags:
  - azure-ml
  - adapters
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

from importlib import import_module
from pkgutil import extend_path
from typing import Any

__path__ = extend_path(__path__, __name__)

_EXPORTS = {
    "build_asset_input": (".inputs", "build_asset_input"),
    "build_local_file_input": (".inputs", "build_local_file_input"),
    "build_local_or_uri_folder_input": (".inputs", "build_local_or_uri_folder_input"),
    "build_uri_folder_input": (".inputs", "build_uri_folder_input"),
    "deploy_registered_model": (".deployment", "deploy_registered_model"),
    "get_ml_client": (".client", "get_ml_client"),
    "install_azure_console_noise_filters": (
        ".submission",
        "install_azure_console_noise_filters",
    ),
    "register_promoted_model": (".registry", "register_promoted_model"),
    "submit_job_quietly": (".submission", "submit_job_quietly"),
    "write_registry_backed_baseline_file": (
        ".registry",
        "write_registry_backed_baseline_file",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "build_asset_input",
    "build_local_file_input",
    "build_local_or_uri_folder_input",
    "build_uri_folder_input",
    "deploy_registered_model",
    "get_ml_client",
    "install_azure_console_noise_filters",
    "register_promoted_model",
    "submit_job_quietly",
    "write_registry_backed_baseline_file",
]
