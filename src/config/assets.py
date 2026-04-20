"""Lightweight asset manifest and lineage tag helpers.

@meta
name: assets
type: module
domain: config
responsibility:
  - Provide config behavior for `src/config/assets.py`.
inputs: []
outputs: []
tags:
  - config
capabilities:
  - fixed-train.attach-lineage-tags
  - online-deploy.invoke-endpoint-deployment-smoke-payload-path-configs-assets
  - online-deploy.stamp-approved-registered-models-lightweight-data-config-component
lifecycle:
  status: active
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import os
from pathlib import Path
import subprocess
from typing import Any

from utils.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_EVIDENCE_ONLY_MODE = "release_evidence_only"
WORKSPACE_BLOBSTORE_JSONL_CAPTURE_MODE = "workspaceblobstore_jsonl"
DEFAULT_ASSET_MANIFEST = PROJECT_ROOT / "configs" / "assets.yaml"
DEFAULT_MANIFEST: dict[str, object] = {
    "data_assets": {
        "production": {"name": "churn-data", "version": "1"},
        "reference": {"name": "churn-data", "version": "1"},
        "smoke": {"name": "churn-data-smoke", "version": "1"},
    },
    "components": {
        "validate_data": {"name": "validate_data", "version": "1"},
        "data_prep": {"name": "data_prep", "version": "1"},
        "train": {"name": "train_model", "version": "1"},
        "promote": {"name": "promote_model", "version": "1"},
    },
    "environment": {
        "name": "bank-churn-env",
        "version": "1",
        "image_repository": "bank-churn",
        "image_tag": "1",
    },
    "model": {
        "name": "churn-prediction-model",
        "experiment_name": "churn-prediction-experiment",
    },
    "deployment": {
        "endpoint_name": "churn-endpoint",
        "deployment_name": "churn-deployment",
        "instance_type": "Standard_D2as_v4",
        "instance_count": "1",
        "smoke_payload": "sample-data.json",
        "online_base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
        "inference_capture": {
            "enabled": False,
            "mode": RELEASE_EVIDENCE_ONLY_MODE,
            "sample_rate": 1.0,
            "max_rows_per_request": 5,
            "capture_inputs": True,
            "capture_outputs": True,
            "redact_inputs": False,
            "output_path": "/tmp/repo-owned-inference-capture",
            "storage_connection_string_env": "INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING",
            "storage_container_env": "INFERENCE_CAPTURE_STORAGE_CONTAINER",
        },
    },
}


@dataclass(frozen=True)
class InferenceCaptureSettings:
    enabled: bool
    mode: str
    sample_rate: float
    max_rows_per_request: int
    capture_inputs: bool
    capture_outputs: bool
    redact_inputs: bool
    output_path: str
    storage_connection_string_env: str | None
    storage_container_env: str | None

    def as_environment_variables(
        self,
        *,
        storage_connection_string: str | None = None,
        storage_container: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, str]:
        env_vars = {
            "INFERENCE_CAPTURE_ENABLED": str(self.enabled).lower(),
            "INFERENCE_CAPTURE_MODE": self.mode,
            "INFERENCE_CAPTURE_SAMPLE_RATE": str(self.sample_rate),
            "INFERENCE_CAPTURE_MAX_ROWS": str(self.max_rows_per_request),
            "INFERENCE_CAPTURE_INPUTS": str(self.capture_inputs).lower(),
            "INFERENCE_CAPTURE_OUTPUTS": str(self.capture_outputs).lower(),
            "INFERENCE_CAPTURE_REDACT_INPUTS": str(self.redact_inputs).lower(),
            "INFERENCE_CAPTURE_OUTPUT_PATH": self.output_path,
        }
        if storage_connection_string:
            env_vars["INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING"] = (
                storage_connection_string
            )
        if storage_container:
            env_vars["INFERENCE_CAPTURE_STORAGE_CONTAINER"] = storage_container
        if session_id:
            env_vars["INFERENCE_CAPTURE_SESSION_ID"] = session_id
        return env_vars


def load_asset_manifest(config_path: str | Path | None = None) -> dict[str, object]:
    """Load the lightweight asset manifest, falling back to known defaults."""
    manifest_path = Path(config_path) if config_path else DEFAULT_ASSET_MANIFEST
    if not manifest_path.exists():
        return dict(DEFAULT_MANIFEST)
    return load_config(str(manifest_path))


def _mapping_value(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    if isinstance(value, Mapping):
        return value
    return {}


def _bool_value(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _float_value(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    return float(value)


def _int_value(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    return int(value)


def deployment_online_base_image(manifest: Mapping[str, object]) -> str:
    """Return the base image used for repo-owned online deployment serving."""
    deployment_defaults = _mapping_value(DEFAULT_MANIFEST, "deployment")
    deployment = _mapping_value(manifest, "deployment")
    return str(
        deployment.get("online_base_image")
        or deployment_defaults.get("online_base_image")
        or "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"
    )


def deployment_inference_capture_settings(
    manifest: Mapping[str, object],
) -> InferenceCaptureSettings:
    """Return repo-owned inference-capture settings for managed online deployment."""
    deployment_defaults = _mapping_value(DEFAULT_MANIFEST, "deployment")
    capture_defaults = _mapping_value(deployment_defaults, "inference_capture")
    deployment = _mapping_value(manifest, "deployment")
    capture = _mapping_value(deployment, "inference_capture")
    capture_inputs = capture.get(
        "capture_inputs",
        capture.get("collect_inputs", capture_defaults.get("capture_inputs", capture_defaults.get("collect_inputs"))),
    )
    capture_outputs = capture.get(
        "capture_outputs",
        capture.get("collect_outputs", capture_defaults.get("capture_outputs", capture_defaults.get("collect_outputs"))),
    )
    return InferenceCaptureSettings(
        enabled=_bool_value(
            capture.get("enabled", capture_defaults.get("enabled")),
            default=False,
        ),
        mode=str(capture.get("mode") or capture_defaults.get("mode") or RELEASE_EVIDENCE_ONLY_MODE),
        sample_rate=_float_value(
            capture.get("sample_rate") or capture_defaults.get("sample_rate"),
            default=1.0,
        ),
        max_rows_per_request=_int_value(
            capture.get("max_rows_per_request") or capture_defaults.get("max_rows_per_request"),
            default=5,
        ),
        capture_inputs=_bool_value(
            capture_inputs,
            default=True,
        ),
        capture_outputs=_bool_value(
            capture_outputs,
            default=True,
        ),
        redact_inputs=_bool_value(
            capture.get("redact_inputs", capture_defaults.get("redact_inputs")),
            default=False,
        ),
        output_path=str(
            capture.get("output_path")
            or capture_defaults.get("output_path")
            or "/tmp/repo-owned-inference-capture"
        ),
        storage_connection_string_env=str(
            capture.get("storage_connection_string_env")
            or capture_defaults.get("storage_connection_string_env")
            or "INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING"
        ),
        storage_container_env=str(
            capture.get("storage_container_env")
            or capture_defaults.get("storage_container_env")
            or "INFERENCE_CAPTURE_STORAGE_CONTAINER"
        ),
    )


def repo_owned_online_inference_capture_settings(
    manifest: Mapping[str, object],
) -> InferenceCaptureSettings:
    """Return deployment-time capture settings after repo-owned serving normalization."""
    settings = deployment_inference_capture_settings(manifest)
    if settings.mode == "azureml_data_collector":
        return InferenceCaptureSettings(
            enabled=False,
            mode=RELEASE_EVIDENCE_ONLY_MODE,
            sample_rate=settings.sample_rate,
            max_rows_per_request=settings.max_rows_per_request,
            capture_inputs=settings.capture_inputs,
            capture_outputs=settings.capture_outputs,
            redact_inputs=settings.redact_inputs,
            output_path=settings.output_path,
            storage_connection_string_env=settings.storage_connection_string_env,
            storage_container_env=settings.storage_container_env,
        )
    return settings


def component_identity(manifest: Mapping[str, object], key: str) -> str:
    """Return a component identity string such as `data_prep:1`."""
    defaults = _mapping_value(_mapping_value(DEFAULT_MANIFEST, "components"), key)
    components = _mapping_value(manifest, "components")
    component = _mapping_value(components, key)
    name = str(component.get("name") or defaults.get("name") or key)
    version = str(component.get("version") or defaults.get("version") or "1")
    return f"{name}:{version}"


def environment_identity(manifest: Mapping[str, object]) -> str:
    """Return an environment identity string such as `bank-churn-env:1`."""
    defaults = _mapping_value(DEFAULT_MANIFEST, "environment")
    environment = _mapping_value(manifest, "environment")
    name = str(environment.get("name") or defaults.get("name") or "bank-churn-env")
    version = str(environment.get("version") or defaults.get("version") or "1")
    return f"{name}:{version}"


def environment_image_defaults(manifest: Mapping[str, object]) -> dict[str, str]:
    """Return the environment image repository/tag defaults from the manifest."""
    defaults = _mapping_value(DEFAULT_MANIFEST, "environment")
    environment = _mapping_value(manifest, "environment")
    image_repository = str(
        environment.get("image_repository")
        or defaults.get("image_repository")
        or "bank-churn"
    )
    image_tag = str(environment.get("image_tag") or defaults.get("image_tag") or "1")
    return {
        "image_repository": image_repository,
        "image_tag": image_tag,
    }


def build_component_lineage_tags(manifest: Mapping[str, object]) -> dict[str, str]:
    """Build component and environment lineage tags from the asset manifest."""
    return {
        "validate_component": component_identity(manifest, "validate_data"),
        "data_prep_component": component_identity(manifest, "data_prep"),
        "train_component": component_identity(manifest, "train"),
        "promote_component": component_identity(manifest, "promote"),
        "environment": environment_identity(manifest),
    }


def get_git_commit(repo_root: str | Path | None = None, *, default: str = "unknown") -> str:
    """Resolve the current Git commit best-effort without blocking workflows."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root or PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return default
    commit = result.stdout.strip()
    return commit or default


def _path_tag(path: str | Path) -> str:
    return Path(path).as_posix()


def build_asset_lineage_tags(
    *,
    current_data_asset_name: str,
    current_data_asset_version: str,
    reference_data_asset_name: str,
    reference_data_asset_version: str,
    data_config_path: str | Path,
    train_config_path: str | Path,
    manifest: Mapping[str, object] | None = None,
    git_commit: str | None = None,
) -> dict[str, str]:
    """Build a compact lineage tag set for Azure ML jobs and models."""
    selected_manifest = manifest or load_asset_manifest()
    tags = {
        "data_asset": current_data_asset_name,
        "data_version": current_data_asset_version,
        "reference_data_asset": reference_data_asset_name,
        "reference_data_version": reference_data_asset_version,
        "data_config": _path_tag(data_config_path),
        "train_config": _path_tag(train_config_path),
        **build_component_lineage_tags(selected_manifest),
        "git_commit": git_commit if git_commit is not None else get_git_commit(),
    }
    return {key: str(value) for key, value in tags.items()}
