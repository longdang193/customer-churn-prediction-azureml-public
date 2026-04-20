"""
@meta
name: run_monitor_handoff
type: script
domain: release-monitoring-orchestration
responsibility:
  - Re-run bounded caller-side capture and monitor evaluation from a saved release record.
  - Emit one stable monitoring handoff summary without redeploying or mutating release truth.
inputs:
  - Saved release_record.json
  - Probe payload JSON files
  - Caller capture config
  - Monitor config
outputs:
  - Capture manifests
  - Downloaded exact capture records
  - Monitor artifacts
  - handoff_summary.json
tags:
  - monitoring
  - release-handoff
  - orchestration
  - azure-ml
features:
  - online-endpoint-deployment
  - release-monitoring-evaluator
capabilities:
  - online-deploy.hand-off-saved-release-truth-repeatable-monitoring-first
  - online-deploy.support-exact-caller-side-blob-capture-retrieval-repeatable
  - online-deploy.provide-release-evidence-monitor-stage-retraining-policy-can
  - online-deploy.provide-enough-release-monitor-provenance-later-retraining-candidate
  - online-deploy.provide-enough-release-monitor-provenance-later-post-validation
  - online-deploy.accept-optional-post-release-monitoring-handoff-continuation-retraining
  - monitor.support-one-thin-monitoring-first-automation-consumes-saved
  - monitor.treat-blob-backed-caller-capture-exact-path-evidence
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.monitoring.handoff_helpers import (
    coerce_dict as _coerce_dict,
    collect_capture_paths,
    download_exact_capture,
    invoke_capture as _invoke_capture,
    invoke_monitor as _invoke_monitor,
    load_json as _load_json,
    resolve_release_targets as _resolve_release_targets,
    write_json as _write_json,
)
from src.inference.client_capture import load_caller_capture_settings

DEFAULT_CONNECTION_STRING_ENV = "INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING"
DEFAULT_CONTAINER_ENV = "INFERENCE_CAPTURE_STORAGE_CONTAINER"


def _resolve_probe_requests(probe_request_paths: list[str]) -> list[Path]:
    return [Path(path).resolve() for path in probe_request_paths]


def _resolve_runtime_contract(
    release_record: dict[str, object],
    monitor_summary: dict[str, object] | None,
) -> str | None:
    deployment = _coerce_dict(release_record.get("deployment"))
    runtime_contract = deployment.get("repo_owned_scoring_status")
    if runtime_contract:
        return str(runtime_contract)
    if monitor_summary is not None and monitor_summary.get("runtime_contract"):
        return str(monitor_summary.get("runtime_contract"))
    return None


def _build_retraining_recommendation(
    monitor_summary: dict[str, object] | None,
) -> dict[str, object] | None:
    if monitor_summary is None:
        return None
    retraining_policy = _coerce_dict(monitor_summary.get("retraining_policy"))
    if not retraining_policy:
        return None
    return {
        "trigger": retraining_policy.get("trigger"),
        "recommended_training_path": retraining_policy.get("recommended_training_path"),
        "path_recommendation": retraining_policy.get("path_recommendation"),
        "policy_confidence": retraining_policy.get("policy_confidence"),
        "requires_dataset_freeze": retraining_policy.get("requires_dataset_freeze"),
        "requires_data_validation": retraining_policy.get("requires_data_validation"),
        "requires_human_review": retraining_policy.get("requires_human_review"),
        "reason_codes": retraining_policy.get("reason_codes", []),
        "path_recommendation_reason_codes": retraining_policy.get(
            "path_recommendation_reason_codes", []
        ),
        "next_step": retraining_policy.get("next_step"),
        "recommended_action": monitor_summary.get("recommended_action"),
    }


def _build_handoff_summary(
    *,
    status: str,
    release_record_path: Path,
    release_record: dict[str, object],
    capture_manifest_paths: list[Path],
    capture_downloads: list[Path],
    monitor_summary_path: Path | None,
    monitor_summary: dict[str, object] | None,
    failure: dict[str, object] | None = None,
) -> dict[str, object]:
    deployment_capture = _coerce_dict(release_record.get("deployment_capture"))
    if monitor_summary is not None:
        monitor_deployment_capture = _coerce_dict(monitor_summary.get("deployment_capture"))
        if monitor_deployment_capture:
            deployment_capture = monitor_deployment_capture

    release_status = str(release_record.get("status") or "unknown")
    capture_status = "not_run"
    if capture_manifest_paths:
        capture_status = "retrieved" if capture_downloads else "captured"
    if failure is not None and str(failure.get("stage")) == "capture":
        capture_status = "failed"

    monitor_status = None
    evidence_level = None
    monitor_caller_capture: dict[str, object] = {}
    if monitor_summary is not None:
        monitor_status = str(monitor_summary.get("monitor_status"))
        evidence_level = str(monitor_summary.get("evidence_level"))
        monitor_caller_capture = _coerce_dict(monitor_summary.get("caller_capture"))
        if monitor_caller_capture.get("status"):
            capture_status = str(monitor_caller_capture.get("status"))

    caller_capture_record_count = len(capture_downloads)
    if monitor_caller_capture.get("record_count") is not None:
        caller_capture_record_count = int(monitor_caller_capture.get("record_count", caller_capture_record_count))
    elif monitor_summary is not None:
        caller_capture_record_count = int(monitor_summary.get("capture_record_count", caller_capture_record_count))

    handoff_status = "blocked"
    if monitor_status == "capture_backed":
        handoff_status = "capture_backed_monitoring_ready"
    elif monitor_status == "limited_but_healthy":
        handoff_status = "release_evidence_only_ready"
    elif monitor_status == "degraded":
        handoff_status = "needs_attention"

    return {
        "status": status,
        "release": {
            "status": release_status,
            "record_path": str(release_record_path),
            "runtime_contract": _resolve_runtime_contract(release_record, monitor_summary),
        },
        "deployment_capture": deployment_capture,
        "caller_capture": {
            "status": capture_status,
            "record_count": caller_capture_record_count,
            "manifest_paths": [str(path) for path in capture_manifest_paths],
            "downloaded_files": [str(path) for path in capture_downloads],
            "evidence_plane": str(monitor_caller_capture.get("evidence_plane") or "caller_side"),
        },
        "monitor": {
            "status": monitor_status,
            "evidence_level": evidence_level,
            "summary_path": str(monitor_summary_path) if monitor_summary_path else None,
        },
        "retraining_recommendation": _build_retraining_recommendation(monitor_summary),
        "handoff": {
            "status": handoff_status,
        },
        "failure": failure,
    }


def main() -> None:
    """
    @capability monitor.support-one-thin-monitoring-first-automation-consumes-saved
    @capability monitor.treat-blob-backed-caller-capture-exact-path-evidence
    """
    parser = argparse.ArgumentParser(
        description="Re-run bounded monitoring handoff from a saved release record.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--release-record", required=True, help="Saved release_record.json to monitor from")
    parser.add_argument("--azure-config", default="config.env", help="Azure runtime config.env path")
    parser.add_argument("--capture-config", default="configs/inference_capture_blob.yaml", help="Caller capture config")
    parser.add_argument("--monitor-config", default="configs/monitor.yaml", help="Monitor config")
    parser.add_argument(
        "--probe-request",
        action="append",
        required=True,
        help="Probe payload JSON. Repeat for multiple payloads.",
    )
    parser.add_argument("--output-dir", required=True, help="Monitoring handoff output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_manifest_dir = output_dir / "capture"
    downloaded_capture_dir = output_dir / "downloaded_capture"
    monitor_dir = output_dir / "monitor"
    summary_path = output_dir / "handoff_summary.json"

    release_record_path = Path(args.release_record)
    release_record = _load_json(release_record_path)
    capture_manifest_paths: list[Path] = []
    capture_downloads: list[Path] = []
    monitor_summary_path: Path | None = None
    monitor_summary: dict[str, object] | None = None
    current_stage = "capture"

    try:
        endpoint_name, deployment_name, model_name, model_version = _resolve_release_targets(release_record)
        probe_requests = _resolve_probe_requests(args.probe_request)

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
                    azure_config=args.azure_config,
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
                        or DEFAULT_CONNECTION_STRING_ENV
                    ),
                    container_env=str(
                        caller_capture_settings.storage_container_env
                        or DEFAULT_CONTAINER_ENV
                    ),
                )
            )

        current_stage = "monitor"
        monitor_summary_path = _invoke_monitor(
            release_record_path=release_record_path,
            monitor_config=args.monitor_config,
            capture_dir=downloaded_capture_dir,
            monitor_dir=monitor_dir,
        )
        monitor_summary = _load_json(monitor_summary_path)
        _write_json(
            summary_path,
            _build_handoff_summary(
                status="succeeded",
                release_record_path=release_record_path,
                release_record=release_record,
                capture_manifest_paths=capture_manifest_paths,
                capture_downloads=capture_downloads,
                monitor_summary_path=monitor_summary_path,
                monitor_summary=monitor_summary,
            ),
        )
    except Exception as error:
        _write_json(
            summary_path,
            _build_handoff_summary(
                status="failed",
                release_record_path=release_record_path,
                release_record=release_record,
                capture_manifest_paths=capture_manifest_paths,
                capture_downloads=capture_downloads,
                monitor_summary_path=monitor_summary_path,
                monitor_summary=monitor_summary,
                failure={
                    "stage": current_stage,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            ),
        )
        raise SystemExit(
            "Monitoring handoff automation failed. "
            f"See {summary_path}"
        ) from error

    print(f"Monitoring handoff summary: {summary_path}")


if __name__ == "__main__":
    main()
