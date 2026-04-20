"""Structured step-manifest helpers for local and AML-backed runs.

@meta
name: step_manifest
type: module
domain: utils
responsibility:
  - Provide utils behavior for `src/utils/step_manifest.py`.
inputs: []
outputs: []
tags:
  - utils
capabilities:
  - data-prep.emit-structured-step-manifest-json-artifact-validation-data
lifecycle:
  status: active
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any


STEP_MANIFEST_FILENAME = "step_manifest.json"
STEP_MANIFEST_ARTIFACT_PATH = "step_manifests"
SCHEMA_VERSION = 1
_SECRET_MARKERS = ("secret", "token", "password", "key", "connection_string")


def _looks_secret(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SECRET_MARKERS)


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if _looks_secret(key_str):
                continue
            normalized[key_str] = _normalize(item)
        return normalized
    if isinstance(value, (list, tuple, set)):
        return [_normalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_step_manifest(*, step_name: str, stage_name: str) -> dict[str, Any]:
    """Create a new manifest with the stable top-level schema."""
    return {
        "schema_version": SCHEMA_VERSION,
        "step_name": step_name,
        "stage_name": stage_name,
        "status": "partial",
        "run_context": {},
        "config": {
            "config_paths": {},
            "resolved": {},
            "overrides": {},
        },
        "inputs": {},
        "outputs": {},
        "tags": {},
        "params": {},
        "metrics": {},
        "artifacts": {},
        "warnings": [],
        "step_specific": {},
    }


def manifest_path_for_dir(output_dir: Path) -> Path:
    """Return the default manifest path for an output directory."""
    return Path(output_dir) / STEP_MANIFEST_FILENAME


def resolve_manifest_output_path(output_path: Path) -> Path:
    """Resolve a manifest output argument to the JSON file path to write.

    Azure ML uri_folder outputs are passed as directories, while local callers
    can still pass an explicit *.json file path.
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() == ".json":
        return output_path
    return manifest_path_for_dir(output_path)


def merge_section(manifest: dict[str, Any], section: str, values: dict[str, Any]) -> None:
    """Merge normalized values into a manifest section."""
    if section not in manifest or not isinstance(manifest[section], dict):
        manifest[section] = {}
    manifest[section].update(_normalize(values))


def merge_config(
    manifest: dict[str, Any],
    *,
    config_paths: dict[str, Any] | None = None,
    resolved: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Merge config details into the manifest."""
    config_section = manifest.setdefault(
        "config",
        {"config_paths": {}, "resolved": {}, "overrides": {}},
    )
    if config_paths:
        config_section["config_paths"].update(_normalize(config_paths))
    if resolved:
        config_section["resolved"].update(_normalize(resolved))
    if overrides:
        config_section["overrides"].update(_normalize(overrides))


def add_warning(manifest: dict[str, Any], message: str) -> None:
    """Append a warning message."""
    warnings = manifest.setdefault("warnings", [])
    warnings.append(str(message))


def set_failure(manifest: dict[str, Any], *, phase: str, exc: BaseException) -> None:
    """Record failure details in a stable step-specific section."""
    step_specific = manifest.setdefault("step_specific", {})
    step_specific["failure"] = {
        "phase": phase,
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _log_manifest_artifact(output_path: Path) -> None:
    """Best-effort MLflow artifact logging for UI-friendly manifest access."""
    try:
        import mlflow

        if mlflow.active_run() is not None:
            _log_artifact_with_fallback(mlflow, output_path)
        elif os.getenv("MLFLOW_RUN_ID"):
            with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
                _log_artifact_with_fallback(mlflow, output_path)
        else:
            return
        print(f"STEP_MANIFEST_ARTIFACT={STEP_MANIFEST_ARTIFACT_PATH}/{output_path.name}")
    except Exception as exc:  # pragma: no cover - defensive against MLflow backend issues.
        print(
            "STEP_MANIFEST_ARTIFACT_WARNING="
            f"failed to log manifest artifact: {type(exc).__name__}: {exc}"
        )


def _log_artifact_with_fallback(mlflow_module: Any, output_path: Path) -> None:
    """Log under a folder when supported; fall back to root for Azure ML backends."""
    try:
        mlflow_module.log_artifact(
            str(output_path),
            artifact_path=STEP_MANIFEST_ARTIFACT_PATH,
        )
    except TypeError as exc:
        if "azureml_artifacts_builder" not in str(exc):
            raise
        try:
            mlflow_module.log_artifact(str(output_path))
        except TypeError as fallback_exc:
            if "azureml_artifacts_builder" not in str(fallback_exc):
                raise
            print(
                "STEP_MANIFEST_ARTIFACT_SKIPPED="
                "Azure ML MLflow artifact backend does not support step manifest logging; "
                f"use STEP_MANIFEST_PATH={output_path}"
            )


def finalize_manifest(
    manifest: dict[str, Any],
    *,
    output_path: Path,
    status: str,
    mirror_output_path: Path | None = None,
) -> Path:
    """Finalize and persist a manifest JSON artifact."""
    manifest["status"] = status
    run_context = manifest.setdefault("run_context", {})
    run_context.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
    output_path = Path(output_path)
    canonical_path = output_path
    normalized_manifest = json.dumps(_normalize(manifest), indent=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(normalized_manifest, encoding="utf-8")
    if mirror_output_path is not None:
        mirror_output_path = resolve_manifest_output_path(Path(mirror_output_path))
        if mirror_output_path != output_path:
            mirror_output_path.parent.mkdir(parents=True, exist_ok=True)
            mirror_output_path.write_text(normalized_manifest, encoding="utf-8")
            canonical_path = mirror_output_path
    _log_manifest_artifact(canonical_path)
    return canonical_path
