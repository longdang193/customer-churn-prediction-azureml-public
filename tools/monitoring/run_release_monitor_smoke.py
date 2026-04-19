"""
@meta
name: run_release_monitor_smoke
type: script
domain: release-monitoring-orchestration
responsibility:
  - Compose release, caller-side capture, exact capture retrieval, and monitor evaluation.
  - Emit one bounded release-plus-monitor automation summary.
inputs:
  - Either a fixed-train job name or an existing release_record.json
  - Probe payload JSON files
  - Caller capture config
  - Monitor config
outputs:
  - Release artifacts
  - Capture manifests
  - Downloaded exact capture records
  - Monitor artifacts
  - release_monitor_summary.json
tags:
  - release
  - monitoring
  - orchestration
  - azure-ml
features:
  - online-endpoint-deployment
  - release-monitoring-evaluator
capabilities:
  - online-deploy.compose-release-follow-up-automation-through-thin-wrapper
  - online-deploy.hand-off-deployed-artifacts-release-metadata-smoke-test
  - monitor.support-one-thin-release-plus-monitor-automation-resolves
  - monitor.treat-blob-backed-caller-capture-exact-path-evidence
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.config.runtime import get_release_config
from src.inference.client_capture import load_caller_capture_settings
from src.monitoring.handoff_helpers import (
    coerce_dict as _coerce_dict,
    collect_capture_paths,
    download_exact_capture,
    invoke_capture as _invoke_capture,
    invoke_monitor as _invoke_monitor,
    load_json as _load_json,
    resolve_release_targets as _resolve_release_targets,
    run_command as _run,
    write_json as _write_json,
)

def _python_command(script_name: str) -> list[str]:
    return [sys.executable, str(PROJECT_ROOT / script_name)]


def _resolve_release_record_path(*, release_dir: Path, job_name: str) -> Path:
    return release_dir / job_name / "release_record.json"


def _default_probe_requests(config_path: str) -> list[Path]:
    release_config = get_release_config(config_path)
    smoke_payload = Path(release_config["smoke_payload"])
    if not smoke_payload.is_absolute():
        smoke_payload = (PROJECT_ROOT / smoke_payload).resolve()
    return [smoke_payload]


def _invoke_release(args: argparse.Namespace, release_dir: Path) -> Path:
    assert args.job_name is not None
    command = _python_command("run_release.py")
    command.extend(
        [
            "--job-name",
            args.job_name,
            "--config",
            args.config,
            "--download-dir",
            str(release_dir),
            "--data-config",
            args.data_config,
            "--train-config",
            args.train_config,
        ]
    )
    if args.deploy:
        command.append("--deploy")
    if args.allow_lineage_mismatch:
        command.append("--allow-lineage-mismatch")
    if args.force_reregister:
        command.append("--force-reregister")
    _run(command)
    return _resolve_release_record_path(release_dir=release_dir, job_name=args.job_name)


def _resolve_release_record(args: argparse.Namespace, release_dir: Path) -> Path:
    if args.release_record:
        return Path(args.release_record)
    return _invoke_release(args, release_dir)


def _resolve_probe_requests(args: argparse.Namespace) -> list[Path]:
    if args.probe_request:
        return [Path(path).resolve() for path in args.probe_request]
    return _default_probe_requests(args.config)


def _build_summary(
    *,
    status: str,
    release_record_path: Path | None,
    release_record: dict[str, object] | None,
    capture_manifest_paths: list[Path],
    capture_downloads: list[Path],
    monitor_summary_path: Path | None,
    monitor_summary: dict[str, object] | None,
    failure: dict[str, object] | None = None,
) -> dict[str, object]:
    release_status = None
    release_capture_summary: dict[str, object] = {
        "status": "unknown",
        "mode": None,
        "enabled": False,
        "warnings": [],
        "output_path": None,
        "evidence_plane": "deployment_owned",
    }
    if release_record is not None:
        release_status = str(release_record.get("status"))
        release_capture_summary.update(_coerce_dict(release_record.get("deployment_capture")))
    capture_status = "not_run"
    if capture_manifest_paths:
        capture_status = "retrieved" if capture_downloads else "captured"
    if failure is not None and str(failure.get("stage")) == "capture":
        capture_status = "failed"
    monitor_status = None
    monitor_evidence_level = None
    monitor_handoff_status = None
    if monitor_summary is not None:
        monitor_status = str(monitor_summary.get("monitor_status"))
        monitor_evidence_level = str(monitor_summary.get("evidence_level"))
        monitor_handoff_status = str(monitor_summary.get("monitoring_handoff_status"))
    caller_capture_record_count = len(capture_downloads)
    if monitor_summary is not None:
        caller_capture_record_count = int(monitor_summary.get("capture_record_count", caller_capture_record_count))

    handoff_status = "blocked"
    if monitor_status == "capture_backed":
        handoff_status = "capture_backed_monitoring_ready"
    elif monitor_status == "limited_but_healthy":
        handoff_status = "release_evidence_only_ready"
    elif monitor_status == "degraded":
        handoff_status = "capture_degraded_after_release"
    elif status == "failed":
        handoff_status = "blocked"

    return {
        "status": status,
        "release": {
            "status": release_status,
            "record_path": str(release_record_path) if release_record_path else None,
        },
        "deployment_capture": release_capture_summary,
        "caller_capture": {
            "status": capture_status,
            "record_count": caller_capture_record_count,
            "manifest_paths": [str(path) for path in capture_manifest_paths],
            "downloaded_files": [str(path) for path in capture_downloads],
            "evidence_plane": "caller_side",
        },
        "capture": {
            "status": capture_status,
            "manifest_paths": [str(path) for path in capture_manifest_paths],
            "downloaded_files": [str(path) for path in capture_downloads],
        },
        "monitor": {
            "status": monitor_status,
            "evidence_level": monitor_evidence_level,
            "handoff_status": monitor_handoff_status,
            "summary_path": str(monitor_summary_path) if monitor_summary_path else None,
        },
        "handoff": {
            "status": handoff_status,
        },
        "failure": failure,
    }


def main() -> None:
    """
    @capability monitor.support-one-thin-release-plus-monitor-automation-resolves
    @capability monitor.treat-blob-backed-caller-capture-exact-path-evidence
    """
    parser = argparse.ArgumentParser(
        description="Compose release, caller capture, retrieval, and monitor evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    release_group = parser.add_mutually_exclusive_group(required=True)
    release_group.add_argument("--job-name", help="Completed fixed-train job name for release automation")
    release_group.add_argument("--release-record", help="Existing release_record.json to reuse")
    parser.add_argument("--config", default="config.env", help="Azure runtime config.env path")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Release data config for fresh release mode")
    parser.add_argument("--train-config", default="configs/train.yaml", help="Release train config for fresh release mode")
    parser.add_argument("--deploy", action="store_true", help="Forward deploy mode to run_release.py")
    parser.add_argument(
        "--allow-lineage-mismatch",
        action="store_true",
        help="Forward lineage mismatch override to run_release.py",
    )
    parser.add_argument(
        "--force-reregister",
        action="store_true",
        help="Force new registration instead of release-model reuse",
    )
    parser.add_argument("--capture-config", default="configs/inference_capture_blob.yaml", help="Caller capture config")
    parser.add_argument("--monitor-config", default="configs/monitor.yaml", help="Monitor config")
    parser.add_argument(
        "--probe-request",
        action="append",
        help="Probe payload JSON. Repeat for multiple payloads. Defaults to release smoke payload.",
    )
    parser.add_argument("--output-dir", required=True, help="Top-level automation output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    release_dir = output_dir / "release"
    capture_manifest_dir = output_dir / "capture_manifests"
    downloaded_capture_dir = output_dir / "downloaded_capture"
    monitor_dir = output_dir / "monitor"
    summary_path = output_dir / "release_monitor_summary.json"

    release_record_path: Path | None = None
    release_record: dict[str, object] | None = None
    capture_manifest_paths: list[Path] = []
    capture_downloads: list[Path] = []
    monitor_summary_path: Path | None = None
    monitor_summary: dict[str, object] | None = None

    try:
        release_record_path = _resolve_release_record(args, release_dir)
        release_record = _load_json(release_record_path)
        endpoint_name, deployment_name, model_name, model_version = _resolve_release_targets(release_record)
        probe_requests = _resolve_probe_requests(args)

        for index, probe_request in enumerate(probe_requests):
            manifest_path = capture_manifest_dir / f"capture_manifest_{index}.json"
            capture_manifest_paths.append(
                _invoke_capture(
                    endpoint_name=endpoint_name,
                    deployment_name=deployment_name,
                    model_name=model_name,
                    model_version=model_version,
                    request_file=probe_request,
                    capture_config=args.capture_config,
                    azure_config=args.config,
                    manifest_path=manifest_path,
                )
            )

        caller_capture_settings = load_caller_capture_settings(Path(args.capture_config))
        capture_uris = collect_capture_paths(capture_manifest_paths)
        for capture_uri in capture_uris:
            capture_downloads.append(
                download_exact_capture(
                    capture_uri=capture_uri,
                    destination_dir=downloaded_capture_dir,
                    connection_string_env=str(
                        caller_capture_settings.storage_connection_string_env
                        or "INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING"
                    ),
                    container_env=str(
                        caller_capture_settings.storage_container_env
                        or "INFERENCE_CAPTURE_STORAGE_CONTAINER"
                    ),
                )
            )

        monitor_summary_path = _invoke_monitor(
            release_record_path=release_record_path,
            monitor_config=args.monitor_config,
            capture_dir=downloaded_capture_dir,
            monitor_dir=monitor_dir,
        )
        monitor_summary = _load_json(monitor_summary_path)
        summary = _build_summary(
            status="succeeded",
            release_record_path=release_record_path,
            release_record=release_record,
            capture_manifest_paths=capture_manifest_paths,
            capture_downloads=capture_downloads,
            monitor_summary_path=monitor_summary_path,
            monitor_summary=monitor_summary,
        )
        _write_json(summary_path, summary)
    except Exception as error:
        failure_stage = "release"
        if release_record is not None and not capture_manifest_paths:
            failure_stage = "capture"
        elif capture_manifest_paths and monitor_summary is None:
            failure_stage = "monitor"
        summary = _build_summary(
            status="failed",
            release_record_path=release_record_path,
            release_record=release_record,
            capture_manifest_paths=capture_manifest_paths,
            capture_downloads=capture_downloads,
            monitor_summary_path=monitor_summary_path,
            monitor_summary=monitor_summary,
            failure={
                "stage": failure_stage,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        _write_json(summary_path, summary)
        raise SystemExit(
            "Release-plus-monitor automation failed. "
            f"See {summary_path}"
        ) from error

    print(f"Release-plus-monitor summary: {summary_path}")


if __name__ == "__main__":
    main()
