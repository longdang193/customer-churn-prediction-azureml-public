"""
@meta
name: release_monitoring_evaluator
type: utility
domain: monitoring
responsibility:
  - Evaluate bounded monitoring readiness from release artifacts.
  - Prefer release evidence first and optional capture evidence second.
inputs:
  - release_record.json
  - configs/monitor.yaml
  - Optional capture JSONL evidence
outputs:
  - Monitor summary payloads
  - Operator-readable monitoring recommendations
tags:
  - monitoring
  - release
  - evaluation
lifecycle:
  status: active
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, cast

import yaml  # type: ignore[import-untyped]


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _as_string_list(value: object) -> list[str]:
    return [str(item) for item in _as_sequence(value)]


def _load_yaml_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _load_retraining_policy_config(config: Mapping[str, object]) -> dict[str, object]:
    policy = _as_mapping(_as_mapping(config.get("monitor")).get("retraining_policy"))
    path_recommendation = _as_mapping(policy.get("path_recommendation"))
    return {
        "enabled": bool(policy.get("enabled", True)),
        "policy_version": int(policy.get("policy_version", 1)),
        "candidate_capture_statuses": _as_string_list(
            policy.get("candidate_capture_statuses")
        )
        or ["class_balance_exceeded"],
        "investigate_monitor_statuses": _as_string_list(
            policy.get("investigate_monitor_statuses")
        )
        or ["degraded", "blocked"],
        "default_training_path": str(policy.get("default_training_path", "fixed_train")),
        "path_recommendation": {
            "enabled": bool(path_recommendation.get("enabled", True)),
            "fixed_train_for_drift_severity": _as_string_list(
                path_recommendation.get("fixed_train_for_drift_severity")
            )
            or ["low", "medium"],
            "model_sweep_for_drift_severity": _as_string_list(
                path_recommendation.get("model_sweep_for_drift_severity")
            )
            or ["high"],
            "repeated_signal_training_path": str(
                path_recommendation.get("repeated_signal_training_path", "model_sweep")
            ),
            "candidate_default_training_path": str(
                path_recommendation.get("candidate_default_training_path", "fixed_train")
            ),
        },
    }


def _severity_rank(level: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(level, 0)


def _derive_path_recommendation(
    *,
    policy_config: Mapping[str, object],
    trigger: str,
    drift_severity: str,
    signal_persistence: str,
) -> tuple[str, list[str]]:
    recommendation_config = _as_mapping(policy_config.get("path_recommendation"))
    if not bool(recommendation_config.get("enabled", True)):
        return "none", ["path_recommendation_disabled"]

    if trigger != "retraining_candidate":
        return "none", [trigger]

    if signal_persistence == "repeated_signal":
        return (
            str(recommendation_config.get("repeated_signal_training_path", "model_sweep")),
            ["repeated_signal"],
        )

    if drift_severity in set(
        _as_string_list(recommendation_config.get("model_sweep_for_drift_severity"))
    ):
        return "model_sweep", [f"drift_severity_{drift_severity}"]

    if drift_severity in set(
        _as_string_list(recommendation_config.get("fixed_train_for_drift_severity"))
    ):
        return (
            str(recommendation_config.get("candidate_default_training_path", "fixed_train")),
            ["bounded_refresh_candidate"],
        )

    return (
        str(recommendation_config.get("candidate_default_training_path", "fixed_train")),
        ["candidate_default_training_path"],
    )


def _prediction_distribution(records: list[Mapping[str, object]]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for record in records:
        for output in _as_sequence(record.get("outputs")):
            key = str(output)
            distribution[key] = distribution.get(key, 0) + 1
    return distribution


def _load_capture_records(capture_path: Path) -> list[Mapping[str, object]]:
    records: list[Mapping[str, object]] = []
    capture_files = [capture_path]
    if capture_path.is_dir():
        capture_files = sorted(capture_path.rglob("*.jsonl"))
    for capture_file in capture_files:
        for line in capture_file.read_text(encoding="utf-8").splitlines():
            raw_line = line.strip()
            if not raw_line:
                continue
            loaded = json.loads(raw_line)
            if isinstance(loaded, Mapping):
                records.append(cast(Mapping[str, object], loaded))
    return records


def _evaluate_capture(
    *,
    capture_output_path: str | None,
    expected_feature_count: int,
    min_capture_records: int,
    max_single_class_share: float,
) -> dict[str, object]:
    checks = {
        "capture_retrievable": False,
        "capture_schema_consistent": False,
        "prediction_distribution_available": False,
        "prediction_class_balance_within_threshold": False,
    }
    if not capture_output_path:
        return {
            "capture_status": "not_configured",
            "capture_record_count": 0,
            "prediction_distribution": {},
            "checks": checks,
        }

    capture_path = Path(capture_output_path)
    if not capture_path.exists():
        return {
            "capture_status": "missing",
            "capture_record_count": 0,
            "prediction_distribution": {},
            "checks": checks,
        }

    records = _load_capture_records(capture_path)
    checks["capture_retrievable"] = True
    checks["capture_schema_consistent"] = bool(records) and all(
        int(record.get("feature_count", -1)) == expected_feature_count
        for record in records
    )
    prediction_distribution = _prediction_distribution(records)
    checks["prediction_distribution_available"] = bool(prediction_distribution)
    max_class_share = 0.0
    if prediction_distribution:
        total_predictions = sum(prediction_distribution.values())
        if total_predictions:
            max_class_share = max(prediction_distribution.values()) / total_predictions
    checks["prediction_class_balance_within_threshold"] = (
        bool(prediction_distribution) and max_class_share <= max_single_class_share
    )
    status = "healthy"
    if len(records) < min_capture_records:
        status = "insufficient_records"
    elif not checks["capture_schema_consistent"]:
        status = "schema_inconsistent"
    elif not checks["prediction_distribution_available"]:
        status = "prediction_distribution_unavailable"
    elif not checks["prediction_class_balance_within_threshold"]:
        status = "class_balance_exceeded"

    return {
        "capture_status": status,
        "capture_record_count": len(records),
        "prediction_distribution": prediction_distribution,
        "max_single_class_share": max_class_share,
        "checks": checks,
    }


def _deployment_capture_summary(
    release_record: Mapping[str, object],
    deployment: Mapping[str, object],
) -> dict[str, object]:
    release_summary = _as_mapping(release_record.get("deployment_capture"))
    if release_summary:
        return {
            "status": str(release_summary.get("status", "unknown")),
            "mode": release_summary.get("mode"),
            "enabled": bool(release_summary.get("enabled")),
            "warnings": list(_as_sequence(release_summary.get("warnings"))),
            "output_path": release_summary.get("output_path"),
            "evidence_plane": str(release_summary.get("evidence_plane", "deployment_owned")),
        }

    return {
        "status": str(deployment.get("inference_capture_status", "disabled")),
        "mode": deployment.get("inference_capture_mode"),
        "enabled": bool(deployment.get("inference_capture_enabled")),
        "warnings": list(_as_sequence(deployment.get("inference_capture_warnings"))),
        "output_path": deployment.get("inference_capture_output_path"),
        "evidence_plane": "deployment_owned",
    }


def _caller_capture_summary(
    *,
    capture_status: str,
    capture_record_count: int,
    capture_path: str | None,
) -> dict[str, object]:
    return {
        "status": capture_status,
        "record_count": capture_record_count,
        "output_path": capture_path,
        "evidence_plane": "caller_side",
    }


def _build_blocked_result(message: str) -> dict[str, object]:
    return {
        "monitor_status": "blocked",
        "evidence_level": "none",
        "monitoring_handoff_status": "blocked",
        "runtime_contract": "unknown",
        "capture_status": "not_evaluated",
        "capture_record_count": 0,
        "prediction_distribution": {},
        "checks": {
            "release_record_present": False,
            "deployment_succeeded": False,
            "canary_passed": False,
            "capture_retrievable": False,
            "capture_schema_consistent": False,
            "prediction_distribution_available": False,
        },
        "retraining_policy": {
            "policy_enabled": True,
            "policy_version": 1,
            "trigger": "investigate_before_retraining",
            "reason_codes": ["release_record_missing"],
            "recommended_training_path": None,
            "path_recommendation": "none",
            "path_recommendation_reason_codes": ["investigate_before_retraining"],
            "drift_severity": "high",
            "signal_persistence": "single_event",
            "policy_confidence": "weak",
            "requires_dataset_freeze": True,
            "requires_data_validation": True,
            "requires_human_review": True,
            "next_step": "Recover the release artifact, then review monitor evidence before opening a retraining candidate.",
        },
        "recommended_action": message,
    }


def _build_retraining_policy(
    *,
    policy_config: Mapping[str, object],
    monitor_status: str,
    capture_status: str,
    capture_evidence_source: str,
    evidence_level: str,
    runtime_contract: str,
) -> dict[str, object]:
    enabled = bool(policy_config.get("enabled", True))
    policy_version = int(policy_config.get("policy_version", 1))
    if not enabled:
        return {
            "policy_enabled": False,
            "policy_version": policy_version,
            "trigger": "policy_disabled",
            "reason_codes": ["policy_disabled"],
            "recommended_training_path": None,
            "path_recommendation": "none",
            "path_recommendation_reason_codes": ["policy_disabled"],
            "drift_severity": "low",
            "signal_persistence": "single_event",
            "policy_confidence": "weak",
            "requires_dataset_freeze": False,
            "requires_data_validation": False,
            "requires_human_review": False,
            "next_step": "Retraining policy is disabled; rely on operator judgment outside monitor outputs.",
        }

    candidate_capture_statuses = set(
        _as_string_list(policy_config.get("candidate_capture_statuses"))
    )
    investigate_monitor_statuses = set(
        _as_string_list(policy_config.get("investigate_monitor_statuses"))
    )
    default_training_path = str(policy_config.get("default_training_path", "fixed_train"))

    if capture_evidence_source == "caller_side" and capture_status in candidate_capture_statuses:
        drift_severity = "medium"
        signal_persistence = "single_event"
        path_recommendation, path_reason_codes = _derive_path_recommendation(
            policy_config=policy_config,
            trigger="retraining_candidate",
            drift_severity=drift_severity,
            signal_persistence=signal_persistence,
        )
        return {
            "policy_enabled": True,
            "policy_version": policy_version,
            "trigger": "retraining_candidate",
            "reason_codes": ["prediction_class_balance_exceeded"],
            "recommended_training_path": path_recommendation or default_training_path,
            "path_recommendation": path_recommendation,
            "path_recommendation_reason_codes": path_reason_codes,
            "drift_severity": drift_severity,
            "signal_persistence": signal_persistence,
            "policy_confidence": "moderate",
            "requires_dataset_freeze": True,
            "requires_data_validation": True,
            "requires_human_review": False,
            "next_step": "Freeze the candidate dataset window, run validate_data, then submit a fixed-train refresh if validation passes.",
        }

    if monitor_status in investigate_monitor_statuses:
        reason_code = "monitor_status_blocked" if monitor_status == "blocked" else "monitor_status_degraded"
        if capture_status == "expected_but_unavailable":
            reason_code = "capture_expected_but_unretrieved"
        return {
            "policy_enabled": True,
            "policy_version": policy_version,
            "trigger": "investigate_before_retraining",
            "reason_codes": [reason_code],
            "recommended_training_path": None,
            "path_recommendation": "none",
            "path_recommendation_reason_codes": ["investigate_before_retraining"],
            "drift_severity": "medium",
            "signal_persistence": "single_event",
            "policy_confidence": "moderate",
            "requires_dataset_freeze": True,
            "requires_data_validation": True,
            "requires_human_review": True,
            "next_step": "Investigate the degraded monitoring evidence before opening a retraining candidate or freezing a new dataset snapshot.",
        }

    reason_codes = ["release_evidence_healthy"]
    if monitor_status == "capture_backed":
        reason_codes = ["capture_backed_healthy"]
    if runtime_contract == "repo_owned_scoring_proven" and evidence_level == "repo_owned_inference_capture":
        reason_codes.append("repo_owned_runtime_confirmed")
    return {
        "policy_enabled": True,
        "policy_version": policy_version,
        "trigger": "no_retraining_signal",
        "reason_codes": reason_codes,
        "recommended_training_path": None,
        "path_recommendation": "none",
        "path_recommendation_reason_codes": ["no_retraining_signal"],
        "drift_severity": "low",
        "signal_persistence": "single_event",
        "policy_confidence": "strong",
        "requires_dataset_freeze": False,
        "requires_data_validation": False,
        "requires_human_review": False,
        "next_step": "Continue monitoring; no retraining candidate is recommended from the current evidence.",
    }


def evaluate_release_monitoring(
    *,
    release_record_path: Path,
    config_path: Path,
    output_dir: Path,
    capture_path_override: Path | None = None,
) -> dict[str, object]:
    """Evaluate monitoring readiness from release artifacts."""
    del output_dir
    config = _load_yaml_config(Path(config_path))
    retraining_policy_config = _load_retraining_policy_config(config)
    if not Path(release_record_path).exists():
        return _build_blocked_result(
            f"release_record.json was not found at {Path(release_record_path)}"
        )

    evidence_config = _as_mapping(config.get("monitor")).get("evidence")
    thresholds = _as_mapping(_as_mapping(config.get("monitor")).get("thresholds"))
    allow_release_evidence_only = bool(
        _as_mapping(evidence_config).get("allow_release_evidence_only", True)
    )
    min_capture_records = int(thresholds.get("min_capture_records", 1))
    expected_feature_count = int(thresholds.get("expected_feature_count", 10))
    max_single_class_share = float(thresholds.get("max_single_class_share", 1.0))

    release_record = _load_json(Path(release_record_path))
    deployment = _as_mapping(release_record.get("deployment"))
    monitoring_handoff = _as_mapping(release_record.get("monitoring_handoff"))
    canary_inference = _as_mapping(release_record.get("canary_inference"))
    canary_payload = _as_mapping(canary_inference.get("payload"))
    evidence_level = str(monitoring_handoff.get("evidence_level", "release_evidence_only"))
    monitoring_handoff_status = str(monitoring_handoff.get("status", "unknown"))
    runtime_contract = str(
        deployment.get("repo_owned_scoring_status", "unknown")
    )
    deployment_succeeded = str(deployment.get("deployment_state", "")) == "Succeeded"
    canary_passed = str(canary_payload.get("validation_status", "")) == "passed" and bool(
        deployment.get("smoke_invoked")
    )

    capture_output_path = str(capture_path_override) if capture_path_override else cast(
        Optional[str], monitoring_handoff.get("inference_capture_output_path")
    )
    if capture_output_path is None:
        capture_output_path = cast(Optional[str], deployment.get("inference_capture_output_path"))
    capture_evaluation = _evaluate_capture(
        capture_output_path=capture_output_path,
        expected_feature_count=expected_feature_count,
        min_capture_records=min_capture_records,
        max_single_class_share=max_single_class_share,
    )

    checks = {
        "release_record_present": True,
        "deployment_succeeded": deployment_succeeded,
        "canary_passed": canary_passed,
        **cast(dict[str, bool], capture_evaluation["checks"]),
    }

    inference_capture_enabled = bool(deployment.get("inference_capture_enabled"))
    configured_capture_status = str(deployment.get("inference_capture_status", "disabled"))
    deployment_capture = _deployment_capture_summary(release_record, deployment)

    if checks["capture_retrievable"] and cast(str, capture_evaluation["capture_status"]) == "healthy":
        monitor_status = "capture_backed"
        effective_capture_status = "healthy"
        effective_evidence_level = "repo_owned_inference_capture"
        capture_evidence_source = "caller_side"
        caller_capture_status = "retrieved"
        recommended_action = (
            "Capture evidence is retrievable and consistent; continue with capture-backed monitoring."
        )
    elif (
        checks["capture_retrievable"]
        and cast(str, capture_evaluation["capture_status"]) == "class_balance_exceeded"
    ):
        monitor_status = "degraded"
        effective_capture_status = "class_balance_exceeded"
        effective_evidence_level = evidence_level
        capture_evidence_source = "caller_side"
        caller_capture_status = "retrieved"
        recommended_action = (
            "prediction class share exceeded threshold; inspect production prediction balance "
            "before treating capture-backed monitoring as healthy."
        )
    elif inference_capture_enabled and configured_capture_status == "healthy":
        monitor_status = "degraded"
        effective_capture_status = "expected_but_unavailable"
        effective_evidence_level = evidence_level
        capture_evidence_source = "deployment_configured_but_unretrieved"
        caller_capture_status = "missing"
        recommended_action = (
            "capture evidence was expected but could not be retrieved; inspect the external sink "
            "and serving runtime logs before relying on capture-backed monitoring."
        )
    elif deployment_succeeded and canary_passed and allow_release_evidence_only:
        monitor_status = "limited_but_healthy"
        effective_evidence_level = evidence_level
        capture_evidence_source = "release_evidence_only"
        caller_capture_status = "not_run"
        if runtime_contract == "generated_runtime_still_in_control":
            effective_capture_status = "not_available_for_this_runtime_contract"
        else:
            effective_capture_status = configured_capture_status
        recommended_action = (
            "Release evidence is healthy; continue with bounded monitoring based on release "
            "records, deployment state, and serving logs."
        )
    else:
        monitor_status = "degraded"
        effective_evidence_level = evidence_level
        effective_capture_status = configured_capture_status
        capture_evidence_source = "release_evidence_only"
        caller_capture_status = "not_run"
        recommended_action = (
            "Release monitoring evidence is incomplete or degraded; inspect the release record "
            "and deployment diagnostics before trusting the handoff."
        )

    retraining_policy = _build_retraining_policy(
        policy_config=retraining_policy_config,
        monitor_status=monitor_status,
        capture_status=effective_capture_status,
        capture_evidence_source=capture_evidence_source,
        evidence_level=effective_evidence_level,
        runtime_contract=runtime_contract,
    )

    return {
        "monitor_status": monitor_status,
        "evidence_level": effective_evidence_level,
        "monitoring_handoff_status": monitoring_handoff_status,
        "runtime_contract": runtime_contract,
        "capture_status": effective_capture_status,
        "capture_evidence_source": capture_evidence_source,
        "capture_record_count": capture_evaluation["capture_record_count"],
        "prediction_distribution": capture_evaluation["prediction_distribution"],
        "max_single_class_share": capture_evaluation.get("max_single_class_share", 0.0),
        "deployment_capture": deployment_capture,
        "caller_capture": _caller_capture_summary(
            capture_status=caller_capture_status,
            capture_record_count=int(capture_evaluation["capture_record_count"]),
            capture_path=capture_output_path,
        ),
        "checks": checks,
        "retraining_policy": retraining_policy,
        "recommended_action": recommended_action,
        "release_record_path": str(release_record_path),
        "capture_output_path": capture_output_path,
    }
