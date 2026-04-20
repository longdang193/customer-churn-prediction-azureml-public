"""Pure release-orchestration helpers for promotion and registration flows.

@meta
name: workflow
type: module
domain: release
responsibility:
  - Provide release behavior for `src/release/workflow.py`.
inputs: []
outputs: []
tags:
  - release
capabilities:
  - online-deploy.write-derived-monitoring-handoff-summary-release-record-json
  - online-deploy.surface-deployment-owned-capture-evidence-release-record-json
  - online-deploy.write-enriched-release-record-json-status-model-resolution
  - online-deploy.reuse-matching-approved-registered-model-version-default-run
lifecycle:
  status: active
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping, Optional, Protocol, Sequence, SupportsFloat, Union, cast


APPROVED_STATUS = "approved"
DEFAULT_PRIMARY_METRIC = "f1"
DEFAULT_BASELINE_METRICS = {
    "f1": 0.0,
    "roc_auc": 0.0,
}
LINEAGE_STATUS_VALIDATED = "validated"
LINEAGE_STATUS_UNVERIFIED = "unverified"
LINEAGE_STATUS_MISMATCH_ALLOWED = "mismatch_allowed"


def _coerce_metric_value(value: object) -> float:
    """Normalize model-tag metric values into floats."""
    return float(cast(Union[SupportsFloat, str], value))


class HasModelMetadata(Protocol):
    """Minimal protocol for Azure ML model-like objects."""

    name: str
    version: str
    tags: Mapping[str, str] | None


ReusableModel = Union[Mapping[str, object], HasModelMetadata]


@dataclass(frozen=True)
class ApprovedModelBaseline:
    name: str
    version: str
    primary_metric: str
    metrics: dict[str, float]


def _coerce_model_metadata(model: Mapping[str, object] | HasModelMetadata) -> ApprovedModelBaseline | None:
    if isinstance(model, Mapping):
        raw_tags = model.get("tags")
        tags = cast(Mapping[str, object], raw_tags) if isinstance(raw_tags, Mapping) else {}
        name = str(model.get("name", ""))
        version = str(model.get("version", ""))
    else:
        tags = cast(Mapping[str, object], model.tags or {})
        name = model.name
        version = model.version

    if str(tags.get("approval_status", "")).lower() != APPROVED_STATUS:
        return None

    primary_metric = str(tags.get("primary_metric", DEFAULT_PRIMARY_METRIC))
    metrics = {
        "f1": _coerce_metric_value(tags.get("f1", 0.0)),
        "roc_auc": _coerce_metric_value(tags.get("roc_auc", 0.0)),
    }
    return ApprovedModelBaseline(
        name=name,
        version=version,
        primary_metric=primary_metric,
        metrics=metrics,
    )


def select_latest_approved_model_baseline(
    models: list[ReusableModel],
) -> Optional[ApprovedModelBaseline]:
    """Return the highest-version approved model baseline from registry metadata."""
    approved_models = [
        approved
        for approved in (_coerce_model_metadata(model) for model in models)
        if approved is not None
    ]
    if not approved_models:
        return None

    return max(approved_models, key=lambda model: int(model.version))


def _model_tags(model: ReusableModel) -> Mapping[str, object]:
    if isinstance(model, Mapping):
        raw_tags = model.get("tags")
        return cast(Mapping[str, object], raw_tags) if isinstance(raw_tags, Mapping) else {}
    return cast(Mapping[str, object], model.tags or {})


def _model_version(model: ReusableModel) -> str:
    if isinstance(model, Mapping):
        return str(model.get("version", "0"))
    return str(model.version)


def _is_expected_tag(tags: Mapping[str, object], key: str, expected_value: object) -> bool:
    return str(tags.get(key, "")) == str(expected_value)


def _matches_candidate_model(
    tags: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    effective_lineage: Mapping[str, object],
) -> bool:
    expected_model = candidate_metrics.get("model_name") or effective_lineage.get("best_model")
    if not expected_model:
        return True
    tagged_model = tags.get("candidate_model_name") or tags.get("best_model")
    return tagged_model is not None and str(tagged_model) == str(expected_model)


def _matches_release_lineage(
    tags: Mapping[str, object],
    *,
    source_job_name: str,
    effective_lineage: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
) -> bool:
    required_tags = {
        "approval_status": APPROVED_STATUS,
        "source_job_name": source_job_name,
        "lineage_status": LINEAGE_STATUS_VALIDATED,
    }
    for key in ("data_config", "train_config"):
        if key in effective_lineage:
            required_tags[key] = str(effective_lineage[key])

    has_required_tags = all(
        _is_expected_tag(tags, key, value)
        for key, value in required_tags.items()
    )
    return has_required_tags and _matches_candidate_model(
        tags,
        candidate_metrics,
        effective_lineage,
    )


def select_reusable_registered_model(
    models: list[ReusableModel],
    *,
    source_job_name: str,
    effective_lineage: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
) -> ReusableModel | None:
    """Return the newest approved model that exactly matches this release lineage."""
    matching_models = [
        model
        for model in models
        if _matches_release_lineage(
            _model_tags(model),
            source_job_name=source_job_name,
            effective_lineage=effective_lineage,
            candidate_metrics=candidate_metrics,
        )
    ]
    if not matching_models:
        return None

    return max(matching_models, key=lambda model: int(_model_version(model)))


def build_baseline_metrics_payload(
    approved_baseline: Optional[ApprovedModelBaseline],
) -> dict[str, object]:
    """Build the promotion baseline payload, falling back to bootstrap zeros."""
    if approved_baseline is None:
        return {
            "model_name": "bootstrap-baseline",
            "primary_metric": DEFAULT_PRIMARY_METRIC,
            **DEFAULT_BASELINE_METRICS,
        }

    return {
        "model_name": approved_baseline.name,
        "model_version": approved_baseline.version,
        "primary_metric": approved_baseline.primary_metric,
        **approved_baseline.metrics,
    }


def ensure_promotable_decision(decision: Mapping[str, object]) -> None:
    """Raise when a promotion decision does not allow release."""
    status = str(decision.get("status", "")).lower()
    if status != "promote":
        raise ValueError(f"Promotion decision is {status or 'unknown'}")


def build_job_output_uri(job_name: str, output_name: str) -> str:
    """Construct the Azure ML URI for a job output."""
    return f"azureml://jobs/{job_name}/outputs/{output_name}/paths/"


def build_registered_model_tags(
    *,
    candidate_metrics: Mapping[str, object],
    promotion_decision: Mapping[str, object],
    job_name: str,
    lineage_tags: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Build tags for the newly approved registered model."""
    core_tags = {
        "approval_status": APPROVED_STATUS,
        "source_job_name": job_name,
        "primary_metric": str(
            promotion_decision.get("primary_metric", DEFAULT_PRIMARY_METRIC)
        ),
        "f1": str(candidate_metrics.get("f1", 0.0)),
        "roc_auc": str(candidate_metrics.get("roc_auc", 0.0)),
        "promotion_status": str(promotion_decision.get("status", "unknown")),
        "candidate_model_name": str(candidate_metrics.get("model_name", "unknown")),
    }
    lineage = {key: str(value) for key, value in (lineage_tags or {}).items()}
    return {**lineage, **core_tags}


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _path_parts(value: str) -> list[str]:
    return [part for part in value.replace("\\", "/").split("/") if part]


def _normalize_path(value: object) -> str | None:
    if value is None:
        return None
    raw_value = str(value).strip()
    if not raw_value:
        return None

    parts = _path_parts(raw_value)
    if "configs" in parts:
        configs_index = parts.index("configs")
        return PurePosixPath(*parts[configs_index:]).as_posix()
    return PurePosixPath(parts[-1]).as_posix() if parts else None


def _lineage_path_matches(declared: str | None, source: str | None) -> bool:
    if not declared or not source:
        return True
    if declared == source:
        return True
    return PurePosixPath(declared).name == PurePosixPath(source).name


def _manifest_config_paths(manifest: Mapping[str, object] | None) -> Mapping[str, object]:
    if manifest is None:
        return {}
    config = _as_mapping(manifest.get("config"))
    nested_config_paths = _as_mapping(config.get("config_paths"))
    root_config_paths = _as_mapping(manifest.get("config_paths"))
    return {**root_config_paths, **nested_config_paths}


def _extract_manifest_lineage(
    *,
    train_manifest: Mapping[str, object] | None = None,
    validation_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    train_config_paths = _manifest_config_paths(train_manifest)
    validation_config_paths = _manifest_config_paths(validation_manifest)
    train_tags = _as_mapping(train_manifest.get("tags")) if train_manifest else {}
    validation_tags = _as_mapping(validation_manifest.get("tags")) if validation_manifest else {}
    train_run_context = _as_mapping(train_manifest.get("run_context")) if train_manifest else {}
    validation_run_context = _as_mapping(validation_manifest.get("run_context")) if validation_manifest else {}

    lineage: dict[str, object] = {}
    canonical_train_config = _normalize_path(train_config_paths.get("canonical_train_config"))
    train_config = canonical_train_config or _normalize_path(train_config_paths.get("train_config"))
    data_config = _normalize_path(validation_config_paths.get("data_config"))
    if train_config:
        lineage["train_config"] = train_config
    if data_config:
        lineage["data_config"] = data_config

    best_model = train_tags.get("best_model")
    if best_model:
        lineage["best_model"] = str(best_model)

    best_model_run_id = train_tags.get("best_model_run_id")
    if best_model_run_id:
        lineage["best_model_run_id"] = str(best_model_run_id)

    run_ids = [
        str(run_id)
        for run_id in (
            train_run_context.get("run_id"),
            validation_run_context.get("run_id"),
        )
        if run_id
    ]
    if run_ids:
        lineage["source_child_run_ids"] = run_ids

    return lineage


def build_release_lineage(
    *,
    declared_data_config: str,
    declared_train_config: str,
    train_manifest: Mapping[str, object] | None = None,
    validation_manifest: Mapping[str, object] | None = None,
    allow_mismatch: bool = False,
) -> dict[str, object]:
    """Validate operator-declared release lineage against source job manifests."""
    declared_lineage: dict[str, object] = {
        "data_config": _normalize_path(declared_data_config) or declared_data_config,
        "train_config": _normalize_path(declared_train_config) or declared_train_config,
    }
    source_job_lineage = _extract_manifest_lineage(
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
    )
    warnings: list[str] = []
    errors: list[str] = []

    if train_manifest is None:
        warnings.append("train manifest is unavailable; train_config lineage is unverified")
    if validation_manifest is None:
        warnings.append("validation manifest is unavailable; data_config lineage is unverified")

    for config_key in ("data_config", "train_config"):
        source_config = cast(Optional[str], source_job_lineage.get(config_key))
        declared_config = cast(Optional[str], declared_lineage.get(config_key))
        if source_config and not _lineage_path_matches(declared_config, source_config):
            errors.append(
                f"{config_key} mismatch: declared {declared_config}, source manifest {source_config}"
            )

    if errors and allow_mismatch:
        warnings.extend(errors)
        errors = []
        status = LINEAGE_STATUS_MISMATCH_ALLOWED
    elif errors:
        status = "failed"
    elif source_job_lineage:
        status = LINEAGE_STATUS_VALIDATED
    else:
        status = LINEAGE_STATUS_UNVERIFIED

    effective_lineage: dict[str, object] = dict(declared_lineage)
    if status != "failed":
        effective_lineage.update(source_job_lineage)
    effective_lineage["lineage_status"] = status
    return {
        "declared_lineage": declared_lineage,
        "source_job_lineage": source_job_lineage,
        "effective_lineage": effective_lineage,
        "validation": {
            "status": "override" if status == LINEAGE_STATUS_MISMATCH_ALLOWED else status,
            "warnings": warnings,
            "errors": errors,
        },
    }


def lineage_validation_errors(lineage: Mapping[str, object]) -> list[str]:
    """Return release-lineage validation errors from a lineage payload."""
    validation = _as_mapping(lineage.get("validation"))
    return [str(error) for error in _as_sequence(validation.get("errors"))]


def _safe_mapping_copy(mapping: Mapping[str, object] | None) -> dict[str, object]:
    return dict(mapping or {})


def _monitoring_handoff_status(
    *,
    release_status: str,
    endpoint_name: object,
    deployment_name: object,
    deployment: Mapping[str, object],
    failure: Mapping[str, object] | None,
    canary_inference: Mapping[str, object] | None,
) -> str:
    if release_status == "failed" and failure is not None and not endpoint_name and not deployment_name:
        return "deploy_failed_before_handoff"

    if not endpoint_name and not deployment_name:
        return "not_deployed"

    deployment_state = str(deployment.get("deployment_state") or "")
    traffic_updated = bool(deployment.get("traffic_updated"))
    smoke_invoked = bool(deployment.get("smoke_invoked"))
    finalization_timed_out = bool(deployment.get("finalization_timed_out"))

    failure_stage = ""
    if failure is not None:
        failure_stage = str(failure.get("failure_stage", ""))

    payload = _as_mapping(canary_inference.get("payload")) if canary_inference else {}
    canary_validation_status = str(payload.get("validation_status", "")) if payload else ""
    inference_capture_enabled = bool(deployment.get("inference_capture_enabled"))
    inference_capture_status = str(deployment.get("inference_capture_status") or "")

    if (
        release_status == "succeeded"
        and deployment_state == "Succeeded"
        and traffic_updated
        and smoke_invoked
        and canary_validation_status == "passed"
    ):
        if inference_capture_enabled and inference_capture_status == "healthy":
            return "ready_for_repo_owned_inference_capture_handoff"
        if inference_capture_enabled and inference_capture_status == "degraded":
            return "capture_degraded_after_deploy"
        return "ready_for_basic_monitoring_handoff"

    if finalization_timed_out or failure_stage == "deployment_finalization_timeout":
        return "deploy_incomplete_or_timed_out"

    if release_status == "failed" and deployment_state == "Succeeded" and traffic_updated:
        return "canary_failed_after_deploy"

    if release_status == "failed":
        return "deploy_failed_before_handoff"

    return "not_deployed"


def _monitoring_operator_summary(
    status: str,
    *,
    deployment_state: object,
    smoke_invoked: bool,
) -> str:
    if status == "ready_for_repo_owned_inference_capture_handoff":
        return (
            "Deployment succeeded, canary scoring passed, and repo-owned inference capture "
            "is configured for ongoing monitoring evidence."
        )
    if status == "capture_degraded_after_deploy":
        return (
            "Deployment and canary scoring succeeded, but repo-owned inference capture "
            "degraded during or after deployment; inspect capture warnings and serving logs."
        )
    if status == "ready_for_basic_monitoring_handoff":
        return (
            "Deployment succeeded and canary scoring passed; use the release record, "
            "Azure endpoint state, and deployment logs as the current monitoring handoff evidence."
        )
    if status == "canary_failed_after_deploy":
        return (
            "Deployment provisioning reached a usable state, but the canary path failed before "
            "or during scoring; investigate the release record first, then Azure endpoint state and logs."
        )
    if status == "deploy_incomplete_or_timed_out":
        return (
            "Deployment did not reach a confirmed healthy terminal state within the local release "
            "finalization budget; treat monitoring handoff as incomplete."
        )
    if status == "deploy_failed_before_handoff":
        return (
            "Deployment failed before a usable monitoring handoff was established; inspect the "
            "failure details and Azure deployment diagnostics."
        )
    return (
        "No endpoint deployment was requested for this release record, so monitoring handoff is "
        "not applicable yet."
    )


def _build_deployment_capture_summary(
    deployment: Mapping[str, object],
) -> dict[str, object]:
    enabled = bool(deployment.get("inference_capture_enabled"))
    mode = deployment.get("inference_capture_mode")
    status = str(deployment.get("inference_capture_status") or "")
    if not status:
        status = "disabled" if not enabled else "unknown"
    return {
        "status": status,
        "mode": mode,
        "enabled": enabled,
        "warnings": list(_as_sequence(deployment.get("inference_capture_warnings"))),
        "output_path": deployment.get("inference_capture_output_path"),
        "evidence_plane": "deployment_owned",
    }


def build_monitoring_handoff(
    *,
    release_status: str,
    endpoint_name: object,
    deployment_name: object,
    deployment: Mapping[str, object],
    failure: Mapping[str, object] | None = None,
    canary_inference: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a bounded monitoring handoff summary from release evidence."""
    payload = _as_mapping(canary_inference.get("payload")) if canary_inference else {}
    canary_validation_status = payload.get("validation_status")
    smoke_invoked = bool(deployment.get("smoke_invoked"))
    handoff_status = _monitoring_handoff_status(
        release_status=release_status,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        deployment=deployment,
        failure=failure,
        canary_inference=canary_inference,
    )
    inference_capture_enabled = bool(deployment.get("inference_capture_enabled"))
    inference_capture_mode = deployment.get("inference_capture_mode")
    inference_capture_status = deployment.get("inference_capture_status")
    inference_capture_warnings = list(_as_sequence(deployment.get("inference_capture_warnings")))
    evidence_level = (
        "repo_owned_inference_capture"
        if handoff_status == "ready_for_repo_owned_inference_capture_handoff"
        else "release_evidence_only"
    )
    return {
        "status": handoff_status,
        "evidence_level": evidence_level,
        "deployment_state": deployment.get("deployment_state"),
        "traffic_updated": bool(deployment.get("traffic_updated")),
        "canary_validation_status": canary_validation_status,
        "canary_invoked": smoke_invoked,
        "inference_capture_enabled": inference_capture_enabled,
        "inference_capture_mode": inference_capture_mode,
        "inference_capture_status": inference_capture_status,
        "inference_capture_warnings": inference_capture_warnings,
        "inference_capture_output_path": deployment.get("inference_capture_output_path"),
        "operator_summary": _monitoring_operator_summary(
            handoff_status,
            deployment_state=deployment.get("deployment_state"),
            smoke_invoked=smoke_invoked,
        ),
        "evidence_sources": [
            "release_record.json",
            "azure_endpoint_state",
            "azure_deployment_logs",
        ],
    }


def build_release_record(
    *,
    job_name: str,
    registered_model_name: str,
    registered_model_version: str,
    promotion_decision: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    endpoint_name: Optional[str] = None,
    deployment_name: Optional[str] = None,
    registered_model_metadata: Mapping[str, object] | None = None,
    lineage: Mapping[str, object] | None = None,
    release_config: Mapping[str, object] | None = None,
    deployment_metadata: Mapping[str, object] | None = None,
    artifacts: Mapping[str, object] | None = None,
    warnings: Sequence[str] | None = None,
    status: str = "succeeded",
    model_resolution: str | None = None,
    failure: Mapping[str, object] | None = None,
    canary_inference: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a machine-readable release record."""
    deployment: dict[str, object] = {
        "endpoint_name": endpoint_name,
        "deployment_name": deployment_name,
    }
    deployment.update(_safe_mapping_copy(deployment_metadata))
    registered_model: dict[str, object] = {
        "name": registered_model_name,
        "version": registered_model_version,
    }
    registered_model.update(_safe_mapping_copy(registered_model_metadata))
    record: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "job_name": job_name,
        "registered_model": registered_model,
        "promotion_decision": dict(promotion_decision),
        "candidate_metrics": dict(candidate_metrics),
        "deployment": deployment,
        "lineage": _safe_mapping_copy(lineage),
        "release_config": _safe_mapping_copy(release_config),
        "artifacts": _safe_mapping_copy(artifacts),
        "warnings": list(warnings or []),
    }
    if model_resolution is not None:
        record["model_resolution"] = model_resolution
    if failure is not None:
        record["failure"] = dict(failure)
    if canary_inference is not None:
        record["canary_inference"] = dict(canary_inference)
    record["deployment_capture"] = _build_deployment_capture_summary(deployment)
    record["monitoring_handoff"] = build_monitoring_handoff(
        release_status=status,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        deployment=deployment,
        failure=failure,
        canary_inference=canary_inference,
    )
    return record
