"""
@meta
name: run_release
type: script
domain: release-orchestration
responsibility:
  - Download promotion and candidate-metrics artifacts for a completed training job.
  - Register only promoted MLflow model outputs in Azure ML.
  - Optionally deploy the approved model to the managed online endpoint.
inputs:
  - config.env
  - Completed Azure ML fixed-training job name
  - Job outputs: promotion_decision, candidate_metrics, mlflow_model
outputs:
  - Registered Azure ML model
  - Optional managed online deployment
  - Machine-readable release record
tags:
  - azure-ml
  - release
  - deployment
features:
  - online-endpoint-deployment
capabilities:
  - online-deploy.register-selected-promoted-model-bundle-azure-ml
  - online-deploy.validate-release-time-data-train-config-lineage-against
  - fixed-train.release-lineage-evidence
  - online-deploy.create-update-managed-online-endpoint-deployment-approved-registered
  - online-deploy.validate-configured-endpoint-smoke-payload-before-invoking-managed
  - online-deploy.write-canary-inference-metadata-release-record-json-including
  - online-deploy.hand-off-deployed-artifacts-release-metadata-smoke-test
  - online-deploy.provide-release-evidence-monitor-stage-retraining-policy-can
invariants:
  - online-deploy.deployment-consumes-promoted-registered-model-than-loosely
  - online-deploy.every-deployment-should-leave-behind-smoke-test
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from dotenv import load_dotenv

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.azureml import deploy_registered_model, get_ml_client, register_promoted_model
from src.azureml.registry import find_reusable_registered_model
from src.config.assets import build_asset_lineage_tags, load_asset_manifest
from src.config.runtime import (
    get_data_asset_config,
    get_reference_data_asset_config,
    get_release_config,
)
from src.inference import build_canary_inference_record
from src.release import (
    build_release_lineage,
    build_release_record,
    ensure_promotable_decision,
    lineage_validation_errors,
)


def _download_json_output(ml_client: Any, job_name: str, output_name: str, target_dir: Path) -> dict[str, object]:
    """Download a JSON job output and return its parsed payload."""
    output_dir = target_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    ml_client.jobs.download(name=job_name, download_path=str(output_dir), output_name=output_name)
    json_files = sorted(output_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON file found after downloading output '{output_name}'.")
    return json.loads(json_files[0].read_text(encoding="utf-8"))


def _download_optional_json_output(
    ml_client: Any,
    job_name: str,
    output_name: str,
    target_dir: Path,
) -> dict[str, object] | None:
    """Download an optional JSON output, returning None when the output is unavailable."""
    try:
        return _download_json_output(ml_client, job_name, output_name, target_dir)
    except Exception as error:
        print(f"Optional output '{output_name}' unavailable: {error}")
        return None


def _download_optional_output_dir(
    ml_client: Any,
    job_name: str,
    output_name: str,
    target_dir: Path,
) -> Path | None:
    """Download an optional directory output, returning None when unavailable."""
    output_dir = target_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        ml_client.jobs.download(
            name=job_name,
            download_path=str(output_dir),
            output_name=output_name,
        )
    except Exception as error:
        print(f"Optional output '{output_name}' unavailable: {error}")
        return None
    return output_dir


def _registered_model_metadata(registered_model: object) -> dict[str, object]:
    """Build a safe metadata view for release records."""
    metadata: dict[str, object] = {}
    for key in ("path", "type", "tags"):
        value = getattr(registered_model, key, None)
        if value is not None:
            metadata[key] = value
    return metadata


def _build_failure(stage: str, error: Exception) -> dict[str, object]:
    return {
        "failure_stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _deployment_metadata_from_result(
    deployment_result: dict[str, object],
    *,
    instance_type: str | None,
    instance_count: int | None,
    smoke_payload: str | None,
) -> dict[str, object]:
    return {
        "instance_type": instance_type,
        "instance_count": instance_count,
        "smoke_payload": smoke_payload,
        "smoke_test_response": deployment_result.get("smoke_test_response"),
        "deployment_state": deployment_result.get("deployment_state"),
        "recovery_used": deployment_result.get("recovery_used"),
        "finalization_timed_out": deployment_result.get("finalization_timed_out"),
        "traffic_updated": deployment_result.get("traffic_updated"),
        "smoke_invoked": deployment_result.get("smoke_invoked"),
        "inference_capture_enabled": deployment_result.get("inference_capture_enabled"),
        "inference_capture_mode": deployment_result.get("inference_capture_mode"),
        "inference_capture_status": deployment_result.get("inference_capture_status"),
        "inference_capture_warnings": deployment_result.get("inference_capture_warnings"),
        "inference_capture_output_path": deployment_result.get("inference_capture_output_path"),
        "repo_owned_scoring_expected": deployment_result.get("repo_owned_scoring_expected"),
        "repo_owned_scoring_observed": deployment_result.get("repo_owned_scoring_observed"),
        "repo_owned_scoring_status": deployment_result.get("repo_owned_scoring_status"),
        "repo_owned_scoring_log_markers": deployment_result.get("repo_owned_scoring_log_markers"),
        "repo_owned_scoring_warnings": deployment_result.get("repo_owned_scoring_warnings"),
    }


def _write_release_record(
    *,
    release_record_path: Path,
    job_name: str,
    registered_model: object,
    promotion_decision: dict[str, object],
    candidate_metrics: dict[str, object],
    deployment_result: dict[str, object],
    release_lineage: dict[str, object],
    safe_release_config: dict[str, object],
    deployment_metadata: dict[str, object],
    artifacts: Mapping[str, object],
    warnings: list[str],
    status: str,
    model_resolution: str,
    failure: dict[str, object] | None = None,
    canary_inference: dict[str, object] | None = None,
) -> dict[str, object]:
    release_record = build_release_record(
        job_name=job_name,
        registered_model_name=str(getattr(registered_model, "name")),
        registered_model_version=str(getattr(registered_model, "version")),
        promotion_decision=promotion_decision,
        candidate_metrics=candidate_metrics,
        endpoint_name=deployment_result.get("endpoint_name"),
        deployment_name=deployment_result.get("deployment_name"),
        registered_model_metadata=_registered_model_metadata(registered_model),
        lineage=release_lineage,
        release_config=safe_release_config,
        deployment_metadata=deployment_metadata,
        artifacts=artifacts,
        warnings=warnings,
        status=status,
        model_resolution=model_resolution,
        failure=failure,
        canary_inference=canary_inference,
    )
    release_record_path.parent.mkdir(parents=True, exist_ok=True)
    release_record_path.write_text(json.dumps(release_record, indent=2), encoding="utf-8")
    return release_record


def main() -> None:
    """Register and optionally deploy a promoted Azure ML model."""
    parser = argparse.ArgumentParser(
        description="Register and optionally deploy the promoted model from a fixed training job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--job-name", required=True, help="Completed Azure ML fixed-training job name")
    parser.add_argument("--config", default="config.env", help="Path to config.env")
    parser.add_argument(
        "--download-dir",
        default=".release-artifacts",
        help="Directory used to download pipeline outputs and write release metadata",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Create or update the managed online deployment after registration",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
        help="Data config path to record in release lineage tags",
    )
    parser.add_argument(
        "--train-config",
        default="configs/train.yaml",
        help="Training config path to record in release lineage tags",
    )
    parser.add_argument(
        "--allow-lineage-mismatch",
        action="store_true",
        help="Allow release when CLI config declarations disagree with source job manifests",
    )
    parser.add_argument(
        "--force-reregister",
        action="store_true",
        help="Register a new approved model version even when a matching one already exists",
    )
    args = parser.parse_args()

    load_dotenv(args.config)

    release_config = get_release_config(args.config)
    current_asset = get_data_asset_config(args.config)
    reference_asset = get_reference_data_asset_config(args.config)
    ml_client = get_ml_client(args.config)

    download_dir = Path(args.download_dir) / args.job_name
    download_dir.mkdir(parents=True, exist_ok=True)

    promotion_decision = _download_json_output(
        ml_client,
        args.job_name,
        "promotion_decision",
        download_dir,
    )
    candidate_metrics = _download_json_output(
        ml_client,
        args.job_name,
        "candidate_metrics",
        download_dir,
    )
    train_manifest = _download_optional_json_output(
        ml_client,
        args.job_name,
        "train_manifest",
        download_dir,
    )
    validation_manifest = _download_optional_json_output(
        ml_client,
        args.job_name,
        "validation_manifest",
        download_dir,
    )
    release_lineage = build_release_lineage(
        declared_data_config=args.data_config,
        declared_train_config=args.train_config,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        allow_mismatch=args.allow_lineage_mismatch,
    )
    lineage_errors = lineage_validation_errors(release_lineage)
    if lineage_errors:
        raise SystemExit(
            "Release lineage validation failed: " + "; ".join(lineage_errors)
        )

    effective_lineage = release_lineage.get("effective_lineage", {})
    effective_data_config = str(
        effective_lineage.get("data_config", args.data_config)
        if isinstance(effective_lineage, dict)
        else args.data_config
    )
    effective_train_config = str(
        effective_lineage.get("train_config", args.train_config)
        if isinstance(effective_lineage, dict)
        else args.train_config
    )
    asset_manifest = load_asset_manifest()
    lineage_tags = build_asset_lineage_tags(
        current_data_asset_name=current_asset["data_asset_name"],
        current_data_asset_version=current_asset["data_asset_version"],
        reference_data_asset_name=reference_asset["data_asset_name"],
        reference_data_asset_version=reference_asset["data_asset_version"],
        data_config_path=effective_data_config,
        train_config_path=effective_train_config,
        manifest=asset_manifest,
    )
    if isinstance(effective_lineage, dict):
        lineage_tags.update(
            {
                str(key): str(value)
                for key, value in effective_lineage.items()
                if not isinstance(value, list)
            }
        )
    ensure_promotable_decision(promotion_decision)

    release_warnings = [
        str(warning)
        for warning in release_lineage.get("validation", {}).get("warnings", [])
        if isinstance(release_lineage.get("validation"), dict)
    ]
    artifacts = {
        "release_dir": str(download_dir),
        "train_manifest": str(download_dir / "train_manifest") if train_manifest else None,
        "validation_manifest": (
            str(download_dir / "validation_manifest")
            if validation_manifest
            else None
        ),
    }
    safe_release_config = {
        key: release_config[key]
        for key in (
            "model_name",
            "endpoint_name",
            "deployment_name",
            "instance_type",
            "instance_count",
            "smoke_payload",
        )
        if key in release_config
    }

    reusable_model = None
    if not args.force_reregister:
        reusable_model = find_reusable_registered_model(
            ml_client,
            model_name=release_config["model_name"],
            job_name=args.job_name,
            effective_lineage=(
                dict(effective_lineage)
                if isinstance(effective_lineage, dict)
                else {}
            ),
            candidate_metrics=candidate_metrics,
        )

    if reusable_model is None:
        registered_model = register_promoted_model(
            ml_client,
            job_name=args.job_name,
            model_name=release_config["model_name"],
            candidate_metrics=candidate_metrics,
            promotion_decision=promotion_decision,
            lineage_tags=lineage_tags,
        )
        model_resolution = "registered"
    else:
        registered_model = reusable_model
        model_resolution = "reused"

    deployment_result: dict[str, object] = {
        "endpoint_name": release_config["endpoint_name"] if args.deploy else None,
        "deployment_name": release_config["deployment_name"] if args.deploy else None,
        "deployment_state": None,
        "recovery_used": False,
        "finalization_timed_out": False,
        "traffic_updated": False,
        "smoke_invoked": False,
    }
    smoke_payload_path = Path(release_config["smoke_payload"]).resolve()
    model_bundle_path = (
        _download_optional_output_dir(ml_client, args.job_name, "mlflow_model", download_dir)
        if args.deploy
        else None
    )
    deployment_metadata: dict[str, object] = {
        "instance_type": release_config["instance_type"] if args.deploy else None,
        "instance_count": int(release_config["instance_count"]) if args.deploy else None,
        "smoke_payload": str(smoke_payload_path) if args.deploy else None,
        "smoke_test_response": None,
        "deployment_state": None,
        "recovery_used": False,
        "finalization_timed_out": False,
        "traffic_updated": False,
        "smoke_invoked": False,
        "inference_capture_enabled": False,
        "inference_capture_mode": "release_evidence_only",
        "inference_capture_status": "disabled",
        "inference_capture_warnings": [],
        "inference_capture_output_path": None,
        "repo_owned_scoring_expected": False,
        "repo_owned_scoring_observed": False,
        "repo_owned_scoring_status": None,
        "repo_owned_scoring_log_markers": [],
        "repo_owned_scoring_warnings": [],
    }
    release_record_path = download_dir / "release_record.json"
    canary_inference: dict[str, object] | None = None
    if args.deploy:
        try:
            deployment_result = deploy_registered_model(
                ml_client,
                registered_model=registered_model,
                endpoint_name=release_config["endpoint_name"],
                deployment_name=release_config["deployment_name"],
                instance_type=release_config["instance_type"],
                instance_count=int(release_config["instance_count"]),
                sample_data_path=smoke_payload_path,
                model_bundle_path=model_bundle_path,
                asset_manifest=asset_manifest,
            )
            deployment_metadata = _deployment_metadata_from_result(
                deployment_result,
                instance_type=release_config["instance_type"],
                instance_count=int(release_config["instance_count"]),
                smoke_payload=str(smoke_payload_path),
            )
            if "failure" in deployment_result:
                _write_release_record(
                    release_record_path=release_record_path,
                    job_name=args.job_name,
                    registered_model=registered_model,
                    promotion_decision=promotion_decision,
                    candidate_metrics=candidate_metrics,
                    deployment_result=deployment_result,
                    release_lineage=release_lineage,
                    safe_release_config=safe_release_config,
                    deployment_metadata=deployment_metadata,
                    artifacts=artifacts,
                    warnings=release_warnings,
                    status="failed",
                    model_resolution=model_resolution,
                    failure=deployment_result["failure"] if isinstance(deployment_result["failure"], dict) else None,
                    canary_inference=canary_inference,
                )
                raise SystemExit(
                    "Release deployment did not reach a successful terminal state. "
                    f"See {release_record_path}"
                )
            payload_summary = deployment_result.get("payload_summary")
            if isinstance(payload_summary, dict):
                canary_inference = build_canary_inference_record(
                    payload_summary=payload_summary,
                    endpoint_name=str(deployment_result.get("endpoint_name")),
                    deployment_name=str(deployment_result.get("deployment_name")),
                    model_name=str(getattr(registered_model, "name")),
                    model_version=str(getattr(registered_model, "version")),
                    response=deployment_result.get("smoke_test_response"),
                )
        except Exception as error:
            _write_release_record(
                release_record_path=release_record_path,
                job_name=args.job_name,
                registered_model=registered_model,
                promotion_decision=promotion_decision,
                candidate_metrics=candidate_metrics,
                deployment_result=deployment_result,
                release_lineage=release_lineage,
                safe_release_config=safe_release_config,
                deployment_metadata=deployment_metadata,
                artifacts=artifacts,
                warnings=release_warnings,
                status="failed",
                model_resolution=model_resolution,
                failure=_build_failure("deployment", error),
                canary_inference=canary_inference,
            )
            raise

    _write_release_record(
        release_record_path=release_record_path,
        job_name=args.job_name,
        registered_model=registered_model,
        promotion_decision=promotion_decision,
        candidate_metrics=candidate_metrics,
        deployment_result=deployment_result,
        release_lineage=release_lineage,
        safe_release_config=safe_release_config,
        deployment_metadata=deployment_metadata,
        artifacts=artifacts,
        warnings=release_warnings,
        status="succeeded",
        model_resolution=model_resolution,
        canary_inference=canary_inference,
    )

    if model_resolution == "registered":
        print(f"Registered approved model {registered_model.name}:{registered_model.version}")
    else:
        print(f"Reused approved model {registered_model.name}:{registered_model.version}")
    print(f"Release record: {release_record_path}")
    if args.deploy:
        print(
            f"Deployed to {deployment_result['endpoint_name']}/{deployment_result['deployment_name']}"
        )


if __name__ == "__main__":
    main()
