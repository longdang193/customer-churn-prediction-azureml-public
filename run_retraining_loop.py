"""
@meta
name: run_retraining_loop
type: script
domain: retraining-loop
responsibility:
  - Orchestrate the bounded phase-1 retraining loop from monitor decision through candidate freeze, validation, path selection, and optional selected-path submission.
  - Preserve existing stage boundaries by delegating to the authoritative candidate, selection, fixed-train, and HPO bridges.
inputs:
  - release_record.json
  - monitor_summary.json or retraining_decision.json
  - Current and reference dataset identifiers
  - configs/data.yaml or smoke-scoped equivalents
outputs:
  - retraining_loop_summary.json
  - retraining_loop_report.md
  - retraining_loop_manifest/step_manifest.json
tags:
  - monitoring
  - retraining
  - orchestration
  - cli
features:
  - model-training-pipeline
  - online-endpoint-deployment
  - release-monitoring-evaluator
capabilities:
  - fixed-train.accept-phase-one-loop
  - fixed-train.continue-loop-after-promotion-evidence
  - online-deploy.accept-opt-retraining-loop-continuation-only-after-promotion
  - online-deploy.accept-optional-post-release-monitoring-handoff-continuation-retraining
  - monitor.support-thin-phase-1-loop-orchestrator-composes-existing
  - monitor.allow-thin-retraining-loop-orchestrator-continue-run-release
  - monitor.allow-same-opt-retraining-loop-continuation-compose-run
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal, Mapping, Optional, TypedDict, cast

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from run_retraining_fixed_train_smoke import (  # noqa: E402
    DEFAULT_DATA_CONFIG_PATH as DEFAULT_DATA_CONFIG_PATH,
)
from run_retraining_fixed_train_smoke import (  # noqa: E402
    DEFAULT_TRAIN_CONFIG_PATH,
)
from run_retraining_hpo_smoke import DEFAULT_HPO_CONFIG_PATH  # noqa: E402
from src.utils.step_manifest import build_step_manifest, finalize_manifest, merge_config, merge_section


MODE_FREEZE_ONLY = "freeze_only"
MODE_FREEZE_AND_VALIDATE = "freeze_and_validate"
MODE_VALIDATE_AND_SELECT_PATH = "validate_and_select_path"
MODE_SUBMIT_SELECTED_PATH = "submit_selected_path"
RELEASE_MODE_DISABLED = "disabled"
RELEASE_MODE_AFTER_PROMOTION = "after_promotion"
RELEASE_MODE_AFTER_RELEASE_MONITOR_HANDOFF = "after_release_monitor_handoff"
SUPPORTED_MODES = (
    MODE_FREEZE_ONLY,
    MODE_FREEZE_AND_VALIDATE,
    MODE_VALIDATE_AND_SELECT_PATH,
    MODE_SUBMIT_SELECTED_PATH,
)
SUPPORTED_RELEASE_MODES = (
    RELEASE_MODE_DISABLED,
    RELEASE_MODE_AFTER_PROMOTION,
    RELEASE_MODE_AFTER_RELEASE_MONITOR_HANDOFF,
)
SUCCESSFUL_MONITOR_HANDOFF_STATUSES = {
    "capture_backed_monitoring_ready",
    "release_evidence_only_ready",
}

FINAL_STAGE_MONITOR_GATE = "monitor_gate"
FINAL_STAGE_CANDIDATE = "candidate"
FINAL_STAGE_VALIDATION = "validation"
FINAL_STAGE_PATH_SELECTION = "path_selection"
FINAL_STAGE_SELECTED_BRIDGE = "selected_bridge"
FINAL_STAGE_PROMOTION_GATE = "promotion_gate"
FINAL_STAGE_RELEASE = "release"
FINAL_STAGE_MONITOR_HANDOFF = "monitor_handoff"

STATUS_SUCCEEDED = "succeeded"
STATUS_BLOCKED = "blocked"
STATUS_FAILED = "failed"

BRIDGE_FIXED_TRAIN = "run_retraining_fixed_train_smoke.py"
BRIDGE_HPO = "run_retraining_hpo_smoke.py"
BRIDGE_HPO_TO_FIXED_TRAIN = "run_retraining_hpo_to_fixed_train.py"

TRIGGER_RETRAINING_CANDIDATE = "retraining_candidate"
TERMINAL_AZURE_JOB_STATUSES = {"completed", "failed", "canceled", "notresponding", "paused"}
RELEASE_CANDIDATE_WAIT_TIMEOUT_SECONDS = 1800
RELEASE_CANDIDATE_POLL_INTERVAL_SECONDS = 15


class DecisionPayload(TypedDict, total=False):
    trigger: str
    reason_codes: list[str]
    policy_version: int
    next_step: str
    recommended_training_path: str | None


@dataclass(frozen=True)
class DecisionContext:
    source_path: Path
    source_kind: Literal["monitor_summary", "retraining_decision"]
    payload: DecisionPayload


@dataclass(frozen=True)
class ReleaseCandidate:
    eligible: bool
    promotion_status: str
    job_name: str
    release_train_config_path: str
    promotion_decision_path: Path
    candidate_metrics_path: Path


def _wait_for_release_candidate_job(
    *,
    ml_client: object,
    job_name: str,
    timeout_seconds: int = RELEASE_CANDIDATE_WAIT_TIMEOUT_SECONDS,
    poll_interval_seconds: int = RELEASE_CANDIDATE_POLL_INTERVAL_SECONDS,
) -> str:
    """Wait for the downstream AML job to reach a terminal state before release gating."""
    deadline = time.monotonic() + timeout_seconds
    last_status = "unknown"

    while True:
        job = ml_client.jobs.get(job_name)
        raw_status = getattr(job, "status", None)
        last_status = str(raw_status or "unknown").strip().lower()
        if last_status in TERMINAL_AZURE_JOB_STATUSES:
            return last_status
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"Timed out waiting for AML job {job_name!r} before release gating; last status={last_status!r}"
            )
        time.sleep(poll_interval_seconds)


def _release_candidate_value(
    release_candidate: ReleaseCandidate | Mapping[str, object],
    field: str,
) -> object:
    if isinstance(release_candidate, Mapping):
        return release_candidate.get(field)
    return getattr(release_candidate, field)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _load_resumed_model_sweep_continuation(
    continuation_summary_path: Path,
) -> dict[str, str | Path]:
    payload = _load_json(continuation_summary_path)
    status = payload.get("status")
    selected_path = payload.get("selected_path")
    submitted_job_name = payload.get("submitted_job_name")
    exported_train_config_path = payload.get("exported_train_config_path")
    hpo_smoke_summary_path = payload.get("hpo_smoke_summary_path")

    if status != "submitted":
        raise ValueError(
            "Resumed continuation summary must have status='submitted' to continue release"
        )
    if selected_path != "model_sweep":
        raise ValueError("Resumed continuation summary must come from selected_path='model_sweep'")
    if not isinstance(submitted_job_name, str) or not submitted_job_name.strip():
        raise ValueError("Resumed continuation summary is missing submitted_job_name")
    if not isinstance(exported_train_config_path, str) or not exported_train_config_path.strip():
        raise ValueError("Resumed continuation summary is missing exported_train_config_path")

    result: dict[str, str | Path] = {
        "summary_path": continuation_summary_path,
        "job_name": submitted_job_name,
        "train_config_path": exported_train_config_path,
    }
    if isinstance(hpo_smoke_summary_path, str) and hpo_smoke_summary_path.strip():
        result["selected_bridge_summary_path"] = Path(hpo_smoke_summary_path)
    return result


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_decision(
    *,
    monitor_summary_path: Path | None,
    retraining_decision_path: Path | None,
) -> DecisionContext:
    if retraining_decision_path is not None:
        return DecisionContext(
            source_path=retraining_decision_path,
            source_kind="retraining_decision",
            payload=cast(DecisionPayload, _load_json(retraining_decision_path)),
        )

    assert monitor_summary_path is not None
    monitor_summary = _load_json(monitor_summary_path)
    retraining_policy = monitor_summary.get("retraining_policy", {})
    if not isinstance(retraining_policy, Mapping):
        raise ValueError("monitor_summary.json did not contain a valid retraining_policy block")
    return DecisionContext(
        source_path=monitor_summary_path,
        source_kind="monitor_summary",
        payload=cast(DecisionPayload, dict(retraining_policy)),
    )


def _build_candidate_command(
    *,
    release_record_path: Path,
    decision: DecisionContext,
    current_data: str,
    reference_data: str,
    data_config_path: str,
    output_dir: Path,
    run_validation_now: bool,
) -> list[str]:
    command = [
        sys.executable,
        "run_retraining_candidate.py",
        "--release-record",
        str(release_record_path),
        "--current-data",
        current_data,
        "--reference-data",
        reference_data,
        "--data-config",
        data_config_path,
        "--output-dir",
        str(output_dir),
    ]
    if decision.source_kind == "monitor_summary":
        command.extend(["--monitor-summary", str(decision.source_path)])
    else:
        command.extend(["--retraining-decision", str(decision.source_path)])
    if run_validation_now:
        command.append("--run-validation")
    return command


def _build_path_selection_command(
    *,
    release_record_path: Path,
    decision: DecisionContext,
    candidate_manifest_path: Path,
    validation_summary_path: Path,
    data_config_path: str,
    train_config_path: str,
    hpo_config_path: str,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "run_retraining_path_selection.py",
        "--release-record",
        str(release_record_path),
        "--candidate-manifest",
        str(candidate_manifest_path),
        "--validation-summary",
        str(validation_summary_path),
        "--data-config",
        data_config_path,
        "--train-config",
        train_config_path,
        "--hpo-config",
        hpo_config_path,
        "--output-dir",
        str(output_dir),
    ]
    if decision.source_kind == "monitor_summary":
        command.extend(["--monitor-summary", str(decision.source_path)])
    else:
        command.extend(["--retraining-decision", str(decision.source_path)])
    return command


def _invoke_candidate_bridge(
    *,
    release_record_path: Path,
    decision: DecisionContext,
    current_data: str,
    reference_data: str,
    data_config_path: str,
    output_dir: Path,
    run_validation_now: bool,
) -> dict[str, Path]:
    completed = subprocess.run(
        _build_candidate_command(
            release_record_path=release_record_path,
            decision=decision,
            current_data=current_data,
            reference_data=reference_data,
            data_config_path=data_config_path,
            output_dir=output_dir,
            run_validation_now=run_validation_now,
        ),
        check=True,
        text=True,
        capture_output=True,
    )
    del completed
    return {
        "summary_path": output_dir / "candidate_summary.json",
        "candidate_manifest_path": output_dir / "retraining_candidate_manifest.json",
        "validation_summary_path": output_dir / "validation" / "validation_summary.json",
    }


def _invoke_path_selection_bridge(
    *,
    release_record_path: Path,
    decision: DecisionContext,
    candidate_manifest_path: Path,
    validation_summary_path: Path,
    data_config_path: str,
    train_config_path: str,
    hpo_config_path: str,
    output_dir: Path,
) -> dict[str, Path | str]:
    completed = subprocess.run(
        _build_path_selection_command(
            release_record_path=release_record_path,
            decision=decision,
            candidate_manifest_path=candidate_manifest_path,
            validation_summary_path=validation_summary_path,
            data_config_path=data_config_path,
            train_config_path=train_config_path,
            hpo_config_path=hpo_config_path,
            output_dir=output_dir,
        ),
        check=True,
        text=True,
        capture_output=True,
    )
    del completed
    selection_path = output_dir / "retraining_path_selection.json"
    selection_payload = _load_json(selection_path)
    selected_path = selection_payload.get("selected_path")
    return {
        "selection_path": selection_path,
        "summary_path": output_dir / "retraining_path_selection_summary.json",
        "selected_path": str(selected_path) if isinstance(selected_path, str) else "unknown",
    }


def _build_fixed_train_command(
    *,
    release_record_path: Path,
    candidate_manifest_path: Path,
    validation_summary_path: Path,
    data_config_path: str,
    train_config_path: str,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "run_retraining_fixed_train_smoke.py",
        "--release-record",
        str(release_record_path),
        "--candidate-manifest",
        str(candidate_manifest_path),
        "--validation-summary",
        str(validation_summary_path),
        "--data-config",
        data_config_path,
        "--train-config",
        train_config_path,
        "--output-dir",
        str(output_dir),
        "--submit",
    ]


def _invoke_fixed_train_bridge(
    *,
    release_record_path: Path,
    candidate_manifest_path: Path,
    validation_summary_path: Path,
    data_config_path: str,
    train_config_path: str,
    output_dir: Path,
) -> dict[str, Path | str | None]:
    completed = subprocess.run(
        _build_fixed_train_command(
            release_record_path=release_record_path,
            candidate_manifest_path=candidate_manifest_path,
            validation_summary_path=validation_summary_path,
            data_config_path=data_config_path,
            train_config_path=train_config_path,
            output_dir=output_dir,
        ),
        check=True,
        text=True,
        capture_output=True,
    )
    del completed
    summary_path = output_dir / "retraining_fixed_train_summary.json"
    summary_payload = _load_json(summary_path)
    submission = summary_payload.get("submission", {})
    job_name = submission.get("job_name") if isinstance(submission, Mapping) else None
    return {
        "summary_path": summary_path,
        "status": str(summary_payload.get("status") or "unknown"),
        "job_name": str(job_name) if isinstance(job_name, str) else None,
    }


def _build_hpo_command(
    *,
    release_record_path: Path,
    selection_path: Path,
    candidate_manifest_path: Path,
    validation_summary_path: Path,
    data_config_path: str,
    hpo_config_path: str,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "run_retraining_hpo_smoke.py",
        "--release-record",
        str(release_record_path),
        "--selection",
        str(selection_path),
        "--candidate-manifest",
        str(candidate_manifest_path),
        "--validation-summary",
        str(validation_summary_path),
        "--data-config",
        data_config_path,
        "--hpo-config",
        hpo_config_path,
        "--output-dir",
        str(output_dir),
        "--submit",
    ]


def _invoke_hpo_bridge(
    *,
    release_record_path: Path,
    selection_path: Path,
    candidate_manifest_path: Path,
    validation_summary_path: Path,
    data_config_path: str,
    hpo_config_path: str,
    output_dir: Path,
) -> dict[str, Path | str | None]:
    completed = subprocess.run(
        _build_hpo_command(
            release_record_path=release_record_path,
            selection_path=selection_path,
            candidate_manifest_path=candidate_manifest_path,
            validation_summary_path=validation_summary_path,
            data_config_path=data_config_path,
            hpo_config_path=hpo_config_path,
            output_dir=output_dir,
        ),
        check=True,
        text=True,
        capture_output=True,
    )
    del completed
    summary_path = output_dir / "retraining_hpo_smoke_summary.json"
    summary_payload = _load_json(summary_path)
    submission = summary_payload.get("submission", {})
    job_name = submission.get("job_name") if isinstance(submission, Mapping) else None
    return {
        "summary_path": summary_path,
        "status": str(summary_payload.get("status") or "unknown"),
        "job_name": str(job_name) if isinstance(job_name, str) else None,
    }


def _build_hpo_to_fixed_train_command(
    *,
    release_record_path: Path,
    selection_path: Path,
    candidate_manifest_path: Path,
    validation_summary_path: Path,
    hpo_smoke_summary_path: Path,
    base_train_config_path: str,
    resource_group: str,
    workspace_name: str,
    hpo_job_name: str,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "run_retraining_hpo_to_fixed_train.py",
        "--release-record",
        str(release_record_path),
        "--selection",
        str(selection_path),
        "--candidate-manifest",
        str(candidate_manifest_path),
        "--validation-summary",
        str(validation_summary_path),
        "--hpo-smoke-summary",
        str(hpo_smoke_summary_path),
        "--hpo-job-name",
        hpo_job_name,
        "--resource-group",
        resource_group,
        "--workspace-name",
        workspace_name,
        "--base-train-config",
        base_train_config_path,
        "--output-dir",
        str(output_dir),
        "--submit",
    ]


def _invoke_hpo_to_fixed_train_bridge(
    *,
    release_record_path: Path,
    selection_path: Path,
    candidate_manifest_path: Path,
    validation_summary_path: Path,
    hpo_smoke_summary_path: Path,
    base_train_config_path: str,
    release_config_path: str,
    hpo_job_name: str,
    output_dir: Path,
) -> dict[str, Path | str | None]:
    from src.config.runtime import load_azure_config

    azure_config = load_azure_config(release_config_path)
    completed = subprocess.run(
        _build_hpo_to_fixed_train_command(
            release_record_path=release_record_path,
            selection_path=selection_path,
            candidate_manifest_path=candidate_manifest_path,
            validation_summary_path=validation_summary_path,
            hpo_smoke_summary_path=hpo_smoke_summary_path,
            base_train_config_path=base_train_config_path,
            resource_group=azure_config["resource_group"],
            workspace_name=azure_config["workspace_name"],
            hpo_job_name=hpo_job_name,
            output_dir=output_dir,
        ),
        check=True,
        text=True,
        capture_output=True,
    )
    del completed
    summary_path = output_dir / "retraining_hpo_to_fixed_train_summary.json"
    summary_payload = _load_json(summary_path)
    job_name = summary_payload.get("submitted_job_name")
    exported_train_config_path = summary_payload.get("exported_train_config_path")
    return {
        "summary_path": summary_path,
        "status": str(summary_payload.get("status") or "unknown"),
        "job_name": str(job_name) if isinstance(job_name, str) else None,
        "train_config_path": (
            str(exported_train_config_path)
            if isinstance(exported_train_config_path, str)
            else None
        ),
    }


def _resolve_release_candidate(
    *,
    selected_path: str,
    submitted_job_name: str | None,
    release_train_config_path: str,
    release_config_path: str,
    data_config_path: str,
    output_dir: Path,
) -> ReleaseCandidate:
    if submitted_job_name is None:
        raise SystemExit("Selected bridge did not expose a submitted AML job name")

    import run_release

    ml_client = run_release.get_ml_client(release_config_path)
    job_status = _wait_for_release_candidate_job(
        ml_client=ml_client,
        job_name=submitted_job_name,
    )
    if job_status != "completed":
        raise SystemExit(
            f"Selected AML job {submitted_job_name!r} did not complete successfully for release continuation; status={job_status!r}"
        )
    gate_dir = output_dir / "promotion-gate" / submitted_job_name
    promotion_decision = run_release._download_json_output(
        ml_client,
        submitted_job_name,
        "promotion_decision",
        gate_dir,
    )
    candidate_metrics = run_release._download_json_output(
        ml_client,
        submitted_job_name,
        "candidate_metrics",
        gate_dir,
    )
    del candidate_metrics

    promotion_status = str(promotion_decision.get("status") or "unknown")
    eligible = False
    try:
        run_release.ensure_promotable_decision(promotion_decision)
        eligible = True
    except Exception:
        eligible = False

    return ReleaseCandidate(
        eligible=eligible,
        promotion_status=promotion_status,
        job_name=submitted_job_name,
        release_train_config_path=release_train_config_path,
        promotion_decision_path=gate_dir / "promotion_decision" / "promotion_decision.json",
        candidate_metrics_path=gate_dir / "candidate_metrics" / "candidate_metrics.json",
    )


def _build_release_command(
    *,
    job_name: str,
    release_config_path: str,
    data_config_path: str,
    train_config_path: str,
    download_dir: Path,
    deploy: bool,
    allow_lineage_mismatch: bool,
    force_reregister: bool,
) -> list[str]:
    command = [
        sys.executable,
        "run_release.py",
        "--job-name",
        job_name,
        "--config",
        release_config_path,
        "--data-config",
        data_config_path,
        "--train-config",
        train_config_path,
        "--download-dir",
        str(download_dir),
    ]
    if deploy:
        command.append("--deploy")
    if allow_lineage_mismatch:
        command.append("--allow-lineage-mismatch")
    if force_reregister:
        command.append("--force-reregister")
    return command


def _invoke_release_bridge(
    *,
    job_name: str,
    release_config_path: str,
    data_config_path: str,
    train_config_path: str,
    output_dir: Path,
    deploy: bool,
    allow_lineage_mismatch: bool,
    force_reregister: bool,
) -> dict[str, Path | str | None]:
    release_download_dir = output_dir / "release"
    completed = subprocess.run(
        _build_release_command(
            job_name=job_name,
            release_config_path=release_config_path,
            data_config_path=data_config_path,
            train_config_path=train_config_path,
            download_dir=release_download_dir,
            deploy=deploy,
            allow_lineage_mismatch=allow_lineage_mismatch,
            force_reregister=force_reregister,
        ),
        check=True,
        text=True,
        capture_output=True,
    )
    del completed
    release_record_path = release_download_dir / job_name / "release_record.json"
    release_record = _load_json(release_record_path)
    return {
        "status": str(release_record.get("status") or "unknown"),
        "release_record_path": release_record_path,
    }


def _build_monitor_handoff_command(
    *,
    release_record_path: Path,
    release_config_path: str,
    capture_config_path: str,
    monitor_config_path: str,
    probe_requests: list[str],
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "run_monitor_handoff.py",
        "--release-record",
        str(release_record_path),
        "--azure-config",
        release_config_path,
        "--capture-config",
        capture_config_path,
        "--monitor-config",
        monitor_config_path,
        "--output-dir",
        str(output_dir),
    ]
    for probe_request in probe_requests:
        command.extend(["--probe-request", probe_request])
    return command


def _invoke_monitor_handoff_bridge(
    *,
    release_record_path: Path,
    release_config_path: str,
    capture_config_path: str,
    monitor_config_path: str,
    probe_requests: list[str],
    output_dir: Path,
) -> dict[str, Path | str | None]:
    if not probe_requests:
        raise SystemExit(
            "Release monitor handoff requires at least one --probe-request payload"
        )
    handoff_output_dir = output_dir / "monitor-handoff"
    completed = subprocess.run(
        _build_monitor_handoff_command(
            release_record_path=release_record_path,
            release_config_path=release_config_path,
            capture_config_path=capture_config_path,
            monitor_config_path=monitor_config_path,
            probe_requests=probe_requests,
            output_dir=handoff_output_dir,
        ),
        check=True,
        text=True,
        capture_output=True,
    )
    del completed
    summary_path = handoff_output_dir / "handoff_summary.json"
    summary_payload = _load_json(summary_path)
    handoff = summary_payload.get("handoff", {})
    handoff_status = handoff.get("status") if isinstance(handoff, Mapping) else None
    return {
        "status": str(handoff_status or summary_payload.get("status") or "unknown"),
        "summary_path": summary_path,
    }


def _build_summary(
    *,
    status: str,
    mode: str,
    release_mode: str,
    final_stage: str,
    release_record_path: Path,
    decision: DecisionContext,
    candidate: Mapping[str, object] | None,
    path_selection: Mapping[str, object] | None,
    selected_bridge: str | None,
    selected_bridge_status: str | None,
    selected_bridge_summary_path: Path | None,
    submitted_job_name: str | None,
    continuation_bridge: str | None = None,
    continuation_bridge_status: str | None = None,
    continuation_bridge_summary_path: Path | None = None,
    release_candidate_job_name: str | None = None,
    promotion_status: str | None = None,
    release_attempted: bool = False,
    release_status: str | None = None,
    continued_release_record_path: Path | None = None,
    monitor_handoff_attempted: bool = False,
    monitor_handoff_status: str | None = None,
    monitor_handoff_summary_path: Path | None = None,
    resumed_from_continuation_summary_path: Path | None = None,
) -> dict[str, object]:
    return {
        "created_at_utc": _utc_timestamp(),
        "status": status,
        "mode": mode,
        "release_mode": release_mode,
        "final_stage": final_stage,
        "release_record_path": str(release_record_path),
        "decision_source": {
            "path": str(decision.source_path),
            "kind": decision.source_kind,
        },
        "trigger": decision.payload.get("trigger"),
        "reason_codes": decision.payload.get("reason_codes", []),
        "candidate": candidate,
        "path_selection": path_selection,
        "selected_bridge": selected_bridge,
        "selected_bridge_status": selected_bridge_status,
        "selected_bridge_summary_path": (
            str(selected_bridge_summary_path) if selected_bridge_summary_path else None
        ),
        "submitted_job_name": submitted_job_name,
        "continuation_bridge": continuation_bridge,
        "continuation_bridge_status": continuation_bridge_status,
        "continuation_bridge_summary_path": (
            str(continuation_bridge_summary_path) if continuation_bridge_summary_path else None
        ),
        "release_candidate_job_name": release_candidate_job_name,
        "promotion_status": promotion_status,
        "release_attempted": release_attempted,
        "release_status": release_status,
        "continued_release_record_path": (
            str(continued_release_record_path) if continued_release_record_path else None
        ),
        "monitor_handoff_attempted": monitor_handoff_attempted,
        "monitor_handoff_status": monitor_handoff_status,
        "monitor_handoff_summary_path": (
            str(monitor_handoff_summary_path) if monitor_handoff_summary_path else None
        ),
        "resumed_from_continuation_summary_path": (
            str(resumed_from_continuation_summary_path)
            if resumed_from_continuation_summary_path
            else None
        ),
    }


def _build_report(summary: Mapping[str, object]) -> str:
    lines = [
        "# Retraining Loop Report",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Mode: `{summary.get('mode')}`",
        f"- Release mode: `{summary.get('release_mode')}`",
        f"- Final stage: `{summary.get('final_stage')}`",
        f"- Trigger: `{summary.get('trigger')}`",
        f"- Selected bridge: `{summary.get('selected_bridge')}`",
        f"- Selected bridge status: `{summary.get('selected_bridge_status')}`",
        f"- Submitted job name: `{summary.get('submitted_job_name')}`",
        f"- Release candidate job name: `{summary.get('release_candidate_job_name')}`",
        f"- Promotion status: `{summary.get('promotion_status')}`",
        f"- Release attempted: `{summary.get('release_attempted')}`",
        f"- Release status: `{summary.get('release_status')}`",
        f"- Monitor handoff attempted: `{summary.get('monitor_handoff_attempted')}`",
        f"- Monitor handoff status: `{summary.get('monitor_handoff_status')}`",
        f"- Resumed continuation summary: `{summary.get('resumed_from_continuation_summary_path')}`",
    ]
    candidate = summary.get("candidate")
    if isinstance(candidate, Mapping):
        lines.extend(
            [
                "",
                "## Candidate",
                f"- Summary path: `{candidate.get('summary_path')}`",
                f"- Manifest path: `{candidate.get('candidate_manifest_path')}`",
                f"- Validation summary path: `{candidate.get('validation_summary_path')}`",
            ]
        )
    path_selection = summary.get("path_selection")
    if isinstance(path_selection, Mapping):
        lines.extend(
            [
                "",
                "## Path Selection",
                f"- Summary path: `{path_selection.get('summary_path')}`",
                f"- Selection path: `{path_selection.get('selection_path')}`",
                f"- Selected path: `{path_selection.get('selected_path')}`",
            ]
        )
    if summary.get("continuation_bridge"):
        lines.extend(
            [
                "",
                "## Continuation",
                f"- Bridge: `{summary.get('continuation_bridge')}`",
                f"- Status: `{summary.get('continuation_bridge_status')}`",
                f"- Summary path: `{summary.get('continuation_bridge_summary_path')}`",
            ]
        )
    if summary.get("continued_release_record_path") or summary.get("monitor_handoff_summary_path"):
        lines.extend(
            [
                "",
                "## Release Continuation",
                f"- Release record path: `{summary.get('continued_release_record_path')}`",
                f"- Monitor handoff summary path: `{summary.get('monitor_handoff_summary_path')}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _write_loop_outputs(
    *,
    summary: Mapping[str, object],
    summary_path: Path,
    report_path: Path,
    manifest_path: Path,
    release_record_path: Path,
    decision: DecisionContext,
    current_data: str,
    reference_data: str,
    data_config_path: str,
    train_config_path: str,
    hpo_config_path: str,
) -> None:
    _write_json(summary_path, summary)
    _write_text(report_path, _build_report(summary))

    manifest = build_step_manifest(step_name="run_retraining_loop", stage_name="monitor")
    merge_config(
        manifest,
        config_paths={
            "release_record": release_record_path,
            "decision_source": decision.source_path,
            "data_config": data_config_path,
            "train_config": train_config_path,
            "hpo_config": hpo_config_path,
        },
        overrides={
            "mode": summary.get("mode"),
            "release_mode": summary.get("release_mode"),
        },
    )
    merge_section(
        manifest,
        "inputs",
        {
            "release_record": {"path": release_record_path},
            "decision_source": {"path": decision.source_path, "kind": decision.source_kind},
            "current_data": {"value": current_data},
            "reference_data": {"value": reference_data},
        },
    )
    merge_section(
        manifest,
        "outputs",
        {
            "retraining_loop_summary": {"path": summary_path},
            "retraining_loop_report": {"path": report_path},
            "retraining_loop_manifest": {"path": manifest_path},
            "continued_release_record": {"path": Path(str(summary["continued_release_record_path"]))}
            if summary.get("continued_release_record_path")
            else None,
            "monitor_handoff_summary": {"path": Path(str(summary["monitor_handoff_summary_path"]))}
            if summary.get("monitor_handoff_summary_path")
            else None,
        },
    )
    for key in ("continued_release_record", "monitor_handoff_summary"):
        if key in manifest["outputs"] and manifest["outputs"][key] is None:
            del manifest["outputs"][key]
    merge_section(
        manifest,
        "step_specific",
        {
            "retraining_loop": {
                "final_stage": summary["final_stage"],
                "trigger": summary["trigger"],
                "selected_bridge": summary["selected_bridge"],
                "selected_bridge_status": summary["selected_bridge_status"],
                "submitted_job_name": summary["submitted_job_name"],
                "continuation_bridge": summary["continuation_bridge"],
                "continuation_bridge_status": summary["continuation_bridge_status"],
                "release_candidate_job_name": summary["release_candidate_job_name"],
                "promotion_status": summary["promotion_status"],
                "release_status": summary["release_status"],
                "monitor_handoff_status": summary["monitor_handoff_status"],
                "resumed_from_continuation_summary_path": summary[
                    "resumed_from_continuation_summary_path"
                ],
            }
        },
    )
    finalize_manifest(manifest, output_path=manifest_path, status=str(summary["status"]))


def main() -> None:
    """
    @capability monitor.support-thin-phase-1-loop-orchestrator-composes-existing
    @capability monitor.allow-thin-retraining-loop-orchestrator-continue-run-release
    @capability monitor.allow-same-opt-retraining-loop-continuation-compose-run
    """
    parser = argparse.ArgumentParser(
        description="Run the bounded phase-1 retraining loop from monitor decision through train-path submission.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--release-record", required=True, help="Path to saved release_record.json")
    decision_group = parser.add_mutually_exclusive_group(required=True)
    decision_group.add_argument("--monitor-summary", help="Path to monitor_summary.json")
    decision_group.add_argument("--retraining-decision", help="Path to retraining_decision.json")
    parser.add_argument("--current-data", required=True, help="Current dataset identifier or local path")
    parser.add_argument("--reference-data", required=True, help="Reference dataset identifier or local path")
    parser.add_argument("--data-config", required=True, help="Data config used for candidate freeze and validation")
    parser.add_argument(
        "--train-config",
        default=DEFAULT_TRAIN_CONFIG_PATH,
        help="Train config used when the selected path is fixed_train",
    )
    parser.add_argument(
        "--hpo-config",
        default=DEFAULT_HPO_CONFIG_PATH,
        help="HPO config used when the selected path is model_sweep",
    )
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default=MODE_VALIDATE_AND_SELECT_PATH,
        help="Where the phase-1 loop should stop",
    )
    parser.add_argument(
        "--release-mode",
        choices=SUPPORTED_RELEASE_MODES,
        default=RELEASE_MODE_DISABLED,
        help="Optional continuation layer after train-path submission",
    )
    parser.add_argument(
        "--release-config",
        default="config.env",
        help="Azure runtime config.env path used for release and optional HPO continuation resolution",
    )
    parser.add_argument(
        "--release-download-dir",
        default=".release-artifacts",
        help="Download root recorded for release continuation audit artifacts",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Pass through deployment to run_release.py when release continuation is enabled",
    )
    parser.add_argument(
        "--allow-lineage-mismatch",
        action="store_true",
        help="Pass through lineage mismatch allowance to run_release.py",
    )
    parser.add_argument(
        "--force-reregister",
        action="store_true",
        help="Pass through forced reregistration to run_release.py",
    )
    parser.add_argument(
        "--capture-config",
        default="configs/inference_capture_blob.yaml",
        help="Capture config used when release_mode continues into monitor handoff",
    )
    parser.add_argument(
        "--monitor-config",
        default="configs/monitor.yaml",
        help="Monitor config used when release_mode continues into monitor handoff",
    )
    parser.add_argument(
        "--probe-request",
        action="append",
        default=[],
        help="Probe payload JSON used by optional post-release monitor handoff. Repeat for multiple payloads.",
    )
    parser.add_argument(
        "--resume-continuation-summary",
        help=(
            "Optional model_sweep continuation summary to resume from already-promotable "
            "HPO-continuation evidence instead of rerunning the stochastic branch"
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for loop artifacts")
    args = parser.parse_args()

    release_record_path = Path(args.release_record).resolve()
    decision = _load_decision(
        monitor_summary_path=Path(args.monitor_summary).resolve() if args.monitor_summary else None,
        retraining_decision_path=(
            Path(args.retraining_decision).resolve() if args.retraining_decision else None
        ),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "retraining_loop_summary.json"
    report_path = output_dir / "retraining_loop_report.md"
    manifest_path = output_dir / "retraining_loop_manifest" / "step_manifest.json"

    candidate_output_dir = output_dir / "candidate"
    selection_output_dir = output_dir / "path-selection"
    selected_bridge_output_dir = output_dir / "selected-bridge"
    hpo_continuation_output_dir = output_dir / "hpo-to-fixed-train"

    trigger = str(decision.payload.get("trigger", "unknown"))
    if trigger != TRIGGER_RETRAINING_CANDIDATE:
        summary = _build_summary(
            status=STATUS_BLOCKED,
            mode=args.mode,
            release_mode=args.release_mode,
            final_stage=FINAL_STAGE_MONITOR_GATE,
            release_record_path=release_record_path,
            decision=decision,
            candidate=None,
            path_selection=None,
            selected_bridge=None,
            selected_bridge_status=None,
            selected_bridge_summary_path=None,
            submitted_job_name=None,
        )
        _write_loop_outputs(
            summary=summary,
            summary_path=summary_path,
            report_path=report_path,
            manifest_path=manifest_path,
            release_record_path=release_record_path,
            decision=decision,
            current_data=args.current_data,
            reference_data=args.reference_data,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            hpo_config_path=args.hpo_config,
        )
        raise SystemExit("Monitor decision did not open a retraining candidate")

    run_validation_now = args.mode != MODE_FREEZE_ONLY
    resumed_continuation_summary_path = (
        Path(args.resume_continuation_summary).resolve()
        if args.resume_continuation_summary
        else None
    )
    candidate_result = _invoke_candidate_bridge(
        release_record_path=release_record_path,
        decision=decision,
        current_data=args.current_data,
        reference_data=args.reference_data,
        data_config_path=args.data_config,
        output_dir=candidate_output_dir,
        run_validation_now=run_validation_now,
    )
    candidate_summary = {
        "summary_path": str(candidate_result["summary_path"]),
        "candidate_manifest_path": str(candidate_result["candidate_manifest_path"]),
        "validation_summary_path": str(candidate_result["validation_summary_path"]),
    }

    if args.mode == MODE_FREEZE_ONLY:
        summary = _build_summary(
            status=STATUS_SUCCEEDED,
            mode=args.mode,
            release_mode=args.release_mode,
            final_stage=FINAL_STAGE_CANDIDATE,
            release_record_path=release_record_path,
            decision=decision,
            candidate=candidate_summary,
            path_selection=None,
            selected_bridge=None,
            selected_bridge_status=None,
            selected_bridge_summary_path=None,
            submitted_job_name=None,
        )
    else:
        validation_summary_payload = _load_json(Path(candidate_result["validation_summary_path"]))
        validation_status = validation_summary_payload.get("status")
        if validation_status != "passed":
            summary = _build_summary(
                status=STATUS_BLOCKED,
                mode=args.mode,
                release_mode=args.release_mode,
                final_stage=FINAL_STAGE_VALIDATION,
                release_record_path=release_record_path,
                decision=decision,
                candidate=candidate_summary,
                path_selection=None,
                selected_bridge=None,
                selected_bridge_status=None,
                selected_bridge_summary_path=None,
                submitted_job_name=None,
            )
            _write_loop_outputs(
                summary=summary,
                summary_path=summary_path,
                report_path=report_path,
                manifest_path=manifest_path,
                release_record_path=release_record_path,
                decision=decision,
                current_data=args.current_data,
                reference_data=args.reference_data,
                data_config_path=args.data_config,
                train_config_path=args.train_config,
                hpo_config_path=args.hpo_config,
            )
            raise SystemExit("Candidate validation did not pass")

        if args.mode == MODE_FREEZE_AND_VALIDATE:
            summary = _build_summary(
                status=STATUS_SUCCEEDED,
                mode=args.mode,
                release_mode=args.release_mode,
                final_stage=FINAL_STAGE_VALIDATION,
                release_record_path=release_record_path,
                decision=decision,
                candidate=candidate_summary,
                path_selection=None,
                selected_bridge=None,
                selected_bridge_status=None,
                selected_bridge_summary_path=None,
                submitted_job_name=None,
            )
        else:
            path_selection_result = _invoke_path_selection_bridge(
                release_record_path=release_record_path,
                decision=decision,
                candidate_manifest_path=Path(candidate_result["candidate_manifest_path"]),
                validation_summary_path=Path(candidate_result["validation_summary_path"]),
                data_config_path=args.data_config,
                train_config_path=args.train_config,
                hpo_config_path=args.hpo_config,
                output_dir=selection_output_dir,
            )
            path_selection_summary = {
                "summary_path": str(path_selection_result["summary_path"]),
                "selection_path": str(path_selection_result["selection_path"]),
                "selected_path": path_selection_result["selected_path"],
            }

            if args.mode == MODE_VALIDATE_AND_SELECT_PATH:
                summary = _build_summary(
                    status=STATUS_SUCCEEDED,
                    mode=args.mode,
                    release_mode=args.release_mode,
                    final_stage=FINAL_STAGE_PATH_SELECTION,
                    release_record_path=release_record_path,
                    decision=decision,
                    candidate=candidate_summary,
                    path_selection=path_selection_summary,
                    selected_bridge=None,
                    selected_bridge_status=None,
                    selected_bridge_summary_path=None,
                    submitted_job_name=None,
                )
            else:
                selected_path = str(path_selection_result["selected_path"])
                selected_bridge = None
                selected_bridge_status: str | None = None
                selected_bridge_summary_path: Path | None = None
                submitted_job_name: str | None = None
                continuation_bridge: str | None = None
                continuation_bridge_status: str | None = None
                continuation_bridge_summary_path: Path | None = None
                release_candidate_job_name: str | None = None
                release_train_config_path: str | None = None
                promotion_status: str | None = None
                continued_release_record_path: Path | None = None
                release_status: str | None = None
                monitor_handoff_status: str | None = None
                monitor_handoff_summary_path: Path | None = None

                if selected_path == "fixed_train":
                    bridge_result = _invoke_fixed_train_bridge(
                        release_record_path=release_record_path,
                        candidate_manifest_path=Path(candidate_result["candidate_manifest_path"]),
                        validation_summary_path=Path(candidate_result["validation_summary_path"]),
                        data_config_path=args.data_config,
                        train_config_path=args.train_config,
                        output_dir=selected_bridge_output_dir,
                    )
                    selected_bridge = BRIDGE_FIXED_TRAIN
                    selected_bridge_status = str(bridge_result.get("status") or "unknown")
                    selected_bridge_summary_path = cast(Path, bridge_result["summary_path"])
                    submitted_job_name = cast(Optional[str], bridge_result["job_name"])
                    release_candidate_job_name = submitted_job_name
                    release_train_config_path = args.train_config
                elif selected_path == "model_sweep":
                    selected_bridge = BRIDGE_HPO
                    if resumed_continuation_summary_path is not None:
                        try:
                            resumed_continuation = _load_resumed_model_sweep_continuation(
                                resumed_continuation_summary_path
                            )
                        except ValueError as exc:
                            summary = _build_summary(
                                status=STATUS_BLOCKED,
                                mode=args.mode,
                                release_mode=args.release_mode,
                                final_stage=FINAL_STAGE_SELECTED_BRIDGE,
                                release_record_path=release_record_path,
                                decision=decision,
                                candidate=candidate_summary,
                                path_selection=path_selection_summary,
                                selected_bridge=selected_bridge,
                                selected_bridge_status="blocked_by_invalid_resume",
                                selected_bridge_summary_path=None,
                                submitted_job_name=None,
                                continuation_bridge=BRIDGE_HPO_TO_FIXED_TRAIN,
                                continuation_bridge_status="blocked_by_invalid_resume",
                                continuation_bridge_summary_path=resumed_continuation_summary_path,
                                resumed_from_continuation_summary_path=resumed_continuation_summary_path,
                            )
                            _write_loop_outputs(
                                summary=summary,
                                summary_path=summary_path,
                                report_path=report_path,
                                manifest_path=manifest_path,
                                release_record_path=release_record_path,
                                decision=decision,
                                current_data=args.current_data,
                                reference_data=args.reference_data,
                                data_config_path=args.data_config,
                                train_config_path=args.train_config,
                                hpo_config_path=args.hpo_config,
                            )
                            raise SystemExit(str(exc)) from exc
                        selected_bridge_status = "resumed"
                        selected_bridge_summary_path = cast(
                            Optional[Path],
                            resumed_continuation.get("selected_bridge_summary_path"),
                        )
                        continuation_bridge = BRIDGE_HPO_TO_FIXED_TRAIN
                        continuation_bridge_status = "resumed"
                        continuation_bridge_summary_path = resumed_continuation_summary_path
                        release_candidate_job_name = cast(
                            str,
                            resumed_continuation["job_name"],
                        )
                        release_train_config_path = cast(
                            str,
                            resumed_continuation["train_config_path"],
                        )
                    else:
                        bridge_result = _invoke_hpo_bridge(
                            release_record_path=release_record_path,
                            selection_path=Path(path_selection_result["selection_path"]),
                            candidate_manifest_path=Path(candidate_result["candidate_manifest_path"]),
                            validation_summary_path=Path(candidate_result["validation_summary_path"]),
                            data_config_path=args.data_config,
                            hpo_config_path=args.hpo_config,
                            output_dir=selected_bridge_output_dir,
                        )
                        selected_bridge_status = str(bridge_result.get("status") or "unknown")
                        selected_bridge_summary_path = cast(Path, bridge_result["summary_path"])
                        submitted_job_name = cast(Optional[str], bridge_result["job_name"])
                else:
                    raise SystemExit(f"Unsupported selected path {selected_path!r}")

                if args.release_mode == RELEASE_MODE_DISABLED:
                    summary = _build_summary(
                        status=STATUS_SUCCEEDED,
                        mode=args.mode,
                        release_mode=args.release_mode,
                        final_stage=FINAL_STAGE_SELECTED_BRIDGE,
                        release_record_path=release_record_path,
                        decision=decision,
                        candidate=candidate_summary,
                        path_selection=path_selection_summary,
                        selected_bridge=selected_bridge,
                        selected_bridge_status=selected_bridge_status,
                        selected_bridge_summary_path=selected_bridge_summary_path,
                        submitted_job_name=submitted_job_name,
                    )
                else:
                    if selected_path == "model_sweep":
                        if resumed_continuation_summary_path is None:
                            if submitted_job_name is None:
                                raise SystemExit("HPO bridge did not expose a submitted AML job name")
                            continuation_result = _invoke_hpo_to_fixed_train_bridge(
                                release_record_path=release_record_path,
                                selection_path=Path(path_selection_result["selection_path"]),
                                candidate_manifest_path=Path(candidate_result["candidate_manifest_path"]),
                                validation_summary_path=Path(candidate_result["validation_summary_path"]),
                                hpo_smoke_summary_path=cast(Path, selected_bridge_summary_path),
                                base_train_config_path=args.train_config,
                                release_config_path=args.release_config,
                                hpo_job_name=submitted_job_name,
                                output_dir=hpo_continuation_output_dir,
                            )
                            continuation_bridge = BRIDGE_HPO_TO_FIXED_TRAIN
                            continuation_bridge_status = str(
                                continuation_result.get("status") or "unknown"
                            )
                            continuation_bridge_summary_path = cast(
                                Path,
                                continuation_result["summary_path"],
                            )
                            release_candidate_job_name = cast(
                                Optional[str],
                                continuation_result["job_name"],
                            )
                            release_train_config_path = cast(
                                Optional[str],
                                continuation_result["train_config_path"],
                            )

                    if release_candidate_job_name is None or release_train_config_path is None:
                        summary = _build_summary(
                            status=STATUS_BLOCKED,
                            mode=args.mode,
                            release_mode=args.release_mode,
                            final_stage=FINAL_STAGE_SELECTED_BRIDGE,
                            release_record_path=release_record_path,
                            decision=decision,
                            candidate=candidate_summary,
                            path_selection=path_selection_summary,
                            selected_bridge=selected_bridge,
                            selected_bridge_status=selected_bridge_status,
                            selected_bridge_summary_path=selected_bridge_summary_path,
                            submitted_job_name=submitted_job_name,
                            continuation_bridge=continuation_bridge,
                            continuation_bridge_status=continuation_bridge_status,
                            continuation_bridge_summary_path=continuation_bridge_summary_path,
                            resumed_from_continuation_summary_path=resumed_continuation_summary_path,
                        )
                        _write_loop_outputs(
                            summary=summary,
                            summary_path=summary_path,
                            report_path=report_path,
                            manifest_path=manifest_path,
                            release_record_path=release_record_path,
                            decision=decision,
                            current_data=args.current_data,
                            reference_data=args.reference_data,
                            data_config_path=args.data_config,
                            train_config_path=args.train_config,
                            hpo_config_path=args.hpo_config,
                        )
                        raise SystemExit("Release continuation could not resolve a promotable fixed-train job")

                    release_candidate = _resolve_release_candidate(
                        selected_path=selected_path,
                        submitted_job_name=release_candidate_job_name,
                        release_train_config_path=release_train_config_path,
                        release_config_path=args.release_config,
                        data_config_path=args.data_config,
                        output_dir=output_dir,
                    )
                    promotion_status = str(
                        _release_candidate_value(release_candidate, "promotion_status") or "unknown"
                    )
                    release_candidate_job_name = cast(
                        Optional[str],
                        _release_candidate_value(release_candidate, "job_name"),
                    )
                    release_train_config_path = cast(
                        Optional[str],
                        _release_candidate_value(release_candidate, "release_train_config_path"),
                    )
                    release_eligible = bool(
                        _release_candidate_value(release_candidate, "eligible")
                    )
                    if not release_eligible:
                        summary = _build_summary(
                            status=STATUS_BLOCKED,
                            mode=args.mode,
                            release_mode=args.release_mode,
                            final_stage=FINAL_STAGE_PROMOTION_GATE,
                            release_record_path=release_record_path,
                            decision=decision,
                            candidate=candidate_summary,
                            path_selection=path_selection_summary,
                            selected_bridge=selected_bridge,
                            selected_bridge_status=selected_bridge_status,
                            selected_bridge_summary_path=selected_bridge_summary_path,
                            submitted_job_name=submitted_job_name,
                            continuation_bridge=continuation_bridge,
                            continuation_bridge_status=continuation_bridge_status,
                            continuation_bridge_summary_path=continuation_bridge_summary_path,
                            release_candidate_job_name=release_candidate_job_name,
                            promotion_status=promotion_status,
                            resumed_from_continuation_summary_path=resumed_continuation_summary_path,
                        )
                        _write_loop_outputs(
                            summary=summary,
                            summary_path=summary_path,
                            report_path=report_path,
                            manifest_path=manifest_path,
                            release_record_path=release_record_path,
                            decision=decision,
                            current_data=args.current_data,
                            reference_data=args.reference_data,
                            data_config_path=args.data_config,
                            train_config_path=args.train_config,
                            hpo_config_path=args.hpo_config,
                        )
                        raise SystemExit("Promotion decision did not permit release continuation")

                    release_result = _invoke_release_bridge(
                        job_name=cast(str, release_candidate_job_name),
                        release_config_path=args.release_config,
                        data_config_path=args.data_config,
                        train_config_path=cast(str, release_train_config_path),
                        output_dir=output_dir,
                        deploy=args.deploy,
                        allow_lineage_mismatch=args.allow_lineage_mismatch,
                        force_reregister=args.force_reregister,
                    )
                    release_status = cast(Optional[str], release_result["status"])
                    continued_release_record_path = cast(
                        Optional[Path],
                        release_result["release_record_path"],
                    )
                    if release_status != STATUS_SUCCEEDED:
                        summary = _build_summary(
                            status=STATUS_FAILED,
                            mode=args.mode,
                            release_mode=args.release_mode,
                            final_stage=FINAL_STAGE_RELEASE,
                            release_record_path=release_record_path,
                            decision=decision,
                            candidate=candidate_summary,
                            path_selection=path_selection_summary,
                            selected_bridge=selected_bridge,
                            selected_bridge_status=selected_bridge_status,
                            selected_bridge_summary_path=selected_bridge_summary_path,
                            submitted_job_name=submitted_job_name,
                            continuation_bridge=continuation_bridge,
                            continuation_bridge_status=continuation_bridge_status,
                            continuation_bridge_summary_path=continuation_bridge_summary_path,
                            release_candidate_job_name=release_candidate_job_name,
                            promotion_status=promotion_status,
                            release_attempted=True,
                            release_status=release_status,
                            continued_release_record_path=continued_release_record_path,
                            resumed_from_continuation_summary_path=resumed_continuation_summary_path,
                        )
                        _write_loop_outputs(
                            summary=summary,
                            summary_path=summary_path,
                            report_path=report_path,
                            manifest_path=manifest_path,
                            release_record_path=release_record_path,
                            decision=decision,
                            current_data=args.current_data,
                            reference_data=args.reference_data,
                            data_config_path=args.data_config,
                            train_config_path=args.train_config,
                            hpo_config_path=args.hpo_config,
                        )
                        raise SystemExit("Release continuation failed")

                    if args.release_mode == RELEASE_MODE_AFTER_PROMOTION:
                        summary = _build_summary(
                            status=STATUS_SUCCEEDED,
                            mode=args.mode,
                            release_mode=args.release_mode,
                            final_stage=FINAL_STAGE_RELEASE,
                            release_record_path=release_record_path,
                            decision=decision,
                            candidate=candidate_summary,
                            path_selection=path_selection_summary,
                            selected_bridge=selected_bridge,
                            selected_bridge_status=selected_bridge_status,
                            selected_bridge_summary_path=selected_bridge_summary_path,
                            submitted_job_name=submitted_job_name,
                            continuation_bridge=continuation_bridge,
                            continuation_bridge_status=continuation_bridge_status,
                            continuation_bridge_summary_path=continuation_bridge_summary_path,
                            release_candidate_job_name=release_candidate_job_name,
                            promotion_status=promotion_status,
                            release_attempted=True,
                            release_status=release_status,
                            continued_release_record_path=continued_release_record_path,
                            resumed_from_continuation_summary_path=resumed_continuation_summary_path,
                        )
                    else:
                        monitor_handoff_result = _invoke_monitor_handoff_bridge(
                            release_record_path=cast(Path, continued_release_record_path),
                            release_config_path=args.release_config,
                            capture_config_path=args.capture_config,
                            monitor_config_path=args.monitor_config,
                            probe_requests=list(args.probe_request),
                            output_dir=output_dir,
                        )
                        monitor_handoff_status = cast(Optional[str], monitor_handoff_result["status"])
                        monitor_handoff_summary_path = cast(
                            Optional[Path],
                            monitor_handoff_result["summary_path"],
                        )
                        handoff_succeeded = monitor_handoff_status in SUCCESSFUL_MONITOR_HANDOFF_STATUSES
                        summary = _build_summary(
                            status=STATUS_SUCCEEDED if handoff_succeeded else STATUS_FAILED,
                            mode=args.mode,
                            release_mode=args.release_mode,
                            final_stage=FINAL_STAGE_MONITOR_HANDOFF,
                            release_record_path=release_record_path,
                            decision=decision,
                            candidate=candidate_summary,
                            path_selection=path_selection_summary,
                            selected_bridge=selected_bridge,
                            selected_bridge_status=selected_bridge_status,
                            selected_bridge_summary_path=selected_bridge_summary_path,
                            submitted_job_name=submitted_job_name,
                            continuation_bridge=continuation_bridge,
                            continuation_bridge_status=continuation_bridge_status,
                            continuation_bridge_summary_path=continuation_bridge_summary_path,
                            release_candidate_job_name=release_candidate_job_name,
                            promotion_status=promotion_status,
                            release_attempted=True,
                            release_status=release_status,
                            continued_release_record_path=continued_release_record_path,
                            monitor_handoff_attempted=True,
                            monitor_handoff_status=monitor_handoff_status,
                            monitor_handoff_summary_path=monitor_handoff_summary_path,
                            resumed_from_continuation_summary_path=resumed_continuation_summary_path,
                        )
                        if not handoff_succeeded:
                            _write_loop_outputs(
                                summary=summary,
                                summary_path=summary_path,
                                report_path=report_path,
                                manifest_path=manifest_path,
                                release_record_path=release_record_path,
                                decision=decision,
                                current_data=args.current_data,
                                reference_data=args.reference_data,
                                data_config_path=args.data_config,
                                train_config_path=args.train_config,
                                hpo_config_path=args.hpo_config,
                            )
                            raise SystemExit("Post-release monitor handoff failed")

    _write_loop_outputs(
        summary=summary,
        summary_path=summary_path,
        report_path=report_path,
        manifest_path=manifest_path,
        release_record_path=release_record_path,
        decision=decision,
        current_data=args.current_data,
        reference_data=args.reference_data,
        data_config_path=args.data_config,
        train_config_path=args.train_config,
        hpo_config_path=args.hpo_config,
    )
    print(f"Retraining loop summary: {summary_path}")


if __name__ == "__main__":
    main()
