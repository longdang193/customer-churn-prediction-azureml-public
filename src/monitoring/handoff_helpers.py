"""
@meta
name: handoff_helpers
type: utility
domain: release-monitoring-orchestration
responsibility:
  - Provide shared helper functions for release-monitoring wrapper scripts.
  - Keep orchestration wrappers thin while preserving exact capture retrieval semantics.
inputs:
  - Release record mappings
  - Capture manifest paths
  - Probe payload JSON files
outputs:
  - Resolved release targets
  - Downloaded exact capture records
  - Normalized JSON helper behavior
tags:
  - monitoring
  - release-handoff
  - orchestration
  - utility
lifecycle:
  status: active
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=False)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def coerce_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def python_command(script_name: str) -> list[str]:
    return [sys.executable, str(PROJECT_ROOT / script_name)]


def collect_capture_paths(manifest_paths: list[Path]) -> list[str]:
    capture_paths: list[str] = []
    for manifest_path in manifest_paths:
        manifest = load_json(manifest_path)
        capture_path = manifest.get("capture_path")
        if capture_path:
            capture_paths.append(str(capture_path))
    return capture_paths


def resolve_release_targets(release_record: dict[str, object]) -> tuple[str, str, str, str]:
    deployment = coerce_dict(release_record.get("deployment"))
    registered_model = coerce_dict(release_record.get("registered_model"))
    endpoint_name = str(deployment.get("endpoint_name") or "")
    deployment_name = str(deployment.get("deployment_name") or "")
    model_name = str(registered_model.get("name") or "unknown-model")
    model_version = str(registered_model.get("version") or "unknown-version")
    if not endpoint_name or not deployment_name:
        raise SystemExit("Release record is missing endpoint or deployment coordinates.")
    return endpoint_name, deployment_name, model_name, model_version


def download_exact_capture(
    *,
    capture_uri: str,
    destination_dir: Path,
    connection_string_env: str = "INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING",
    container_env: str = "INFERENCE_CAPTURE_STORAGE_CONTAINER",
    runner: Any = None,
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = Path(capture_uri)
    if candidate_path.exists():
        destination_path = destination_dir / candidate_path.name
        shutil.copy2(candidate_path, destination_path)
        return destination_path

    destination_path = destination_dir / Path(capture_uri).name
    command = python_command("src/monitoring/download_capture_blob.py")
    command.extend(
        [
            "--capture-uri",
            capture_uri,
            "--output-file",
            str(destination_path),
            "--connection-string-env",
            connection_string_env,
            "--container-env",
            container_env,
        ]
    )
    effective_runner = run_command if runner is None else runner
    effective_runner(command)
    return destination_path


def invoke_capture(
    *,
    endpoint_name: str,
    deployment_name: str,
    model_name: str,
    model_version: str,
    request_file: Path,
    capture_config: str,
    azure_config: str,
    manifest_path: Path,
    runner: Any = None,
) -> Path:
    command = python_command("run_inference_capture.py")
    command.extend(
        [
            "--endpoint-name",
            endpoint_name,
            "--deployment-name",
            deployment_name,
            "--request-file",
            str(request_file),
            "--config",
            capture_config,
            "--azure-config",
            azure_config,
            "--output-manifest",
            str(manifest_path),
            "--model-name",
            model_name,
            "--model-version",
            model_version,
        ]
    )
    effective_runner = run_command if runner is None else runner
    effective_runner(command)
    return manifest_path


def invoke_monitor(
    *,
    release_record_path: Path,
    monitor_config: str,
    capture_dir: Path,
    monitor_dir: Path,
    runner: Any = None,
) -> Path:
    command = python_command("run_monitor.py")
    command.extend(
        [
            "--release-record",
            str(release_record_path),
            "--config",
            monitor_config,
            "--capture-path",
            str(capture_dir),
            "--output-dir",
            str(monitor_dir),
        ]
    )
    effective_runner = run_command if runner is None else runner
    effective_runner(command)
    return monitor_dir / "monitor_summary.json"
