"""
@meta
type: test
scope: unit
domain: release
covers:
  - Release record generation for promoted registrations and deployments
excludes:
  - Real Azure ML calls
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations


def test_build_release_record_captures_registration_and_deployment_metadata() -> None:
    """
    @proves online-deploy.write-enriched-release-record-json-status-model-resolution
    @proves online-deploy.write-canary-inference-metadata-release-record-json-including
    @proves online-deploy.write-derived-monitoring-handoff-summary-release-record-json
    @proves online-deploy.surface-deployment-owned-capture-evidence-release-record-json
    """
    from src.release.workflow import build_release_record

    record = build_release_record(
        job_name="job-123",
        registered_model_name="churn-model",
        registered_model_version="9",
        promotion_decision={"status": "promote", "primary_metric": "f1"},
        candidate_metrics={"model_name": "xgboost", "f1": 0.81, "roc_auc": 0.88},
        endpoint_name="churn-endpoint",
        deployment_name="blue",
        registered_model_metadata={
            "path": "azureml://jobs/job-123/outputs/mlflow_model/paths/",
            "type": "mlflow_model",
            "tags": {"train_config": "train_smoke.yaml"},
        },
        lineage={
            "declared_lineage": {"train_config": "configs/train_smoke.yaml"},
            "source_job_lineage": {"train_config": "train_smoke.yaml"},
            "effective_lineage": {"train_config": "train_smoke.yaml"},
            "validation": {"status": "validated", "warnings": [], "errors": []},
        },
        release_config={"model_name": "churn-model", "smoke_payload": "sample-data.json"},
        deployment_metadata={
            "instance_type": "Standard_D2as_v4",
            "instance_count": 1,
            "smoke_payload": "sample-data.json",
            "smoke_test_response": {"result": [0]},
        },
        artifacts={"release_dir": ".release-artifacts/job-123"},
        warnings=["duplicate source job registration candidate"],
    )

    assert record["job_name"] == "job-123"
    assert record["registered_model"]["name"] == "churn-model"
    assert record["registered_model"]["version"] == "9"
    assert record["registered_model"]["path"] == "azureml://jobs/job-123/outputs/mlflow_model/paths/"
    assert record["registered_model"]["tags"]["train_config"] == "train_smoke.yaml"
    assert record["promotion_decision"]["status"] == "promote"
    assert record["deployment"]["endpoint_name"] == "churn-endpoint"
    assert record["deployment"]["smoke_payload"] == "sample-data.json"
    assert record["deployment"]["smoke_test_response"] == {"result": [0]}
    assert record["lineage"]["validation"]["status"] == "validated"
    assert record["release_config"]["smoke_payload"] == "sample-data.json"
    assert record["artifacts"]["release_dir"] == ".release-artifacts/job-123"
    assert record["warnings"] == ["duplicate source job registration candidate"]
    assert record["monitoring_handoff"]["status"] == "not_deployed"
    assert record["monitoring_handoff"]["evidence_level"] == "release_evidence_only"
    assert record["deployment_capture"]["status"] == "disabled"
    assert record["deployment_capture"]["enabled"] is False
    assert record["deployment_capture"]["evidence_plane"] == "deployment_owned"


def test_build_release_record_captures_status_resolution_and_failure() -> None:
    from src.release.workflow import build_release_record

    record = build_release_record(
        job_name="job-123",
        registered_model_name="churn-model",
        registered_model_version="10",
        promotion_decision={"status": "promote", "primary_metric": "f1"},
        candidate_metrics={"model_name": "rf", "f1": 0.8, "roc_auc": 0.85},
        status="failed",
        model_resolution="reused",
        failure={
            "failure_stage": "deployment",
            "error_type": "RuntimeError",
            "error_message": "quota exceeded",
        },
    )

    assert record["status"] == "failed"
    assert record["model_resolution"] == "reused"
    assert record["failure"]["failure_stage"] == "deployment"
    assert record["failure"]["error_type"] == "RuntimeError"
    assert record["failure"]["error_message"] == "quota exceeded"
    assert record["monitoring_handoff"]["status"] == "deploy_failed_before_handoff"
    assert record["monitoring_handoff"]["canary_invoked"] is False


def test_build_release_record_preserves_deployment_finalization_metadata() -> None:
    from src.release.workflow import build_release_record

    record = build_release_record(
        job_name="job-123",
        registered_model_name="churn-model",
        registered_model_version="10",
        promotion_decision={"status": "promote", "primary_metric": "f1"},
        candidate_metrics={"model_name": "rf", "f1": 0.8, "roc_auc": 0.85},
        endpoint_name="churn-endpoint",
        deployment_name="blue",
        deployment_metadata={
            "deployment_state": "Creating",
            "recovery_used": True,
            "finalization_timed_out": True,
            "traffic_updated": False,
            "smoke_invoked": False,
            "smoke_test_response": None,
        },
        status="failed",
        model_resolution="reused",
        failure={
            "failure_stage": "deployment_finalization_timeout",
            "error_type": "TimeoutError",
            "error_message": "local release wait expired",
        },
    )

    assert record["deployment"]["deployment_state"] == "Creating"
    assert record["deployment"]["recovery_used"] is True
    assert record["deployment"]["finalization_timed_out"] is True
    assert record["deployment"]["traffic_updated"] is False
    assert record["deployment"]["smoke_invoked"] is False
    assert record["failure"]["failure_stage"] == "deployment_finalization_timeout"
    assert record["monitoring_handoff"]["status"] == "deploy_incomplete_or_timed_out"
    assert record["monitoring_handoff"]["deployment_state"] == "Creating"


def test_build_release_record_marks_successful_deploy_as_ready_for_basic_monitoring_handoff() -> None:
    """
    @proves online-deploy.write-derived-monitoring-handoff-summary-release-record-json
    """
    from src.release.workflow import build_release_record

    record = build_release_record(
        job_name="job-123",
        registered_model_name="churn-model",
        registered_model_version="10",
        promotion_decision={"status": "promote", "primary_metric": "f1"},
        candidate_metrics={"model_name": "rf", "f1": 0.8, "roc_auc": 0.85},
        endpoint_name="churn-endpoint",
        deployment_name="blue",
        deployment_metadata={
            "deployment_state": "Succeeded",
            "recovery_used": False,
            "finalization_timed_out": False,
            "traffic_updated": True,
            "smoke_invoked": True,
            "smoke_test_response": "ok",
        },
        canary_inference={
            "payload": {
                "path": "sample-data.json",
                "format": "input_data_2d_array",
                "row_count": 1,
                "feature_count": 10,
                "validation_status": "passed",
            },
            "endpoint": {"endpoint_name": "churn-endpoint", "deployment_name": "blue"},
            "model": {"name": "churn-model", "version": "10"},
            "response": {"preview": "ok"},
        },
    )

    assert record["monitoring_handoff"]["status"] == "ready_for_basic_monitoring_handoff"
    assert record["monitoring_handoff"]["evidence_level"] == "release_evidence_only"
    assert record["monitoring_handoff"]["deployment_state"] == "Succeeded"
    assert record["monitoring_handoff"]["canary_validation_status"] == "passed"
    assert record["monitoring_handoff"]["canary_invoked"] is True


def test_build_release_record_preserves_repo_owned_scoring_contract_metadata() -> None:
    """
    @proves online-deploy.configure-approved-model-repo-owned-src-inference-score
    """
    from src.release.workflow import build_release_record

    record = build_release_record(
        job_name="job-123",
        registered_model_name="churn-model",
        registered_model_version="10",
        promotion_decision={"status": "promote", "primary_metric": "f1"},
        candidate_metrics={"model_name": "rf", "f1": 0.8, "roc_auc": 0.85},
        endpoint_name="churn-endpoint",
        deployment_name="blue",
        deployment_metadata={
            "deployment_state": "Succeeded",
            "recovery_used": False,
            "finalization_timed_out": False,
            "traffic_updated": True,
            "smoke_invoked": True,
            "smoke_test_response": "ok",
            "repo_owned_scoring_expected": True,
            "repo_owned_scoring_observed": True,
            "repo_owned_scoring_status": "repo_owned_scoring_proven",
            "repo_owned_scoring_log_markers": [
                "REPO_OWNED_SCORER_INIT=",
                "REPO_OWNED_SCORER_RUN=",
            ],
            "repo_owned_scoring_warnings": [],
        },
        canary_inference={
            "payload": {
                "path": "sample-data.json",
                "format": "input_data_2d_array",
                "row_count": 1,
                "feature_count": 10,
                "validation_status": "passed",
            }
        },
    )

    assert record["deployment"]["repo_owned_scoring_expected"] is True
    assert record["deployment"]["repo_owned_scoring_observed"] is True
    assert record["deployment"]["repo_owned_scoring_status"] == "repo_owned_scoring_proven"
    assert record["deployment"]["repo_owned_scoring_log_markers"] == [
        "REPO_OWNED_SCORER_INIT=",
        "REPO_OWNED_SCORER_RUN=",
    ]


def test_build_release_record_marks_payload_validation_failure_as_canary_failed_after_deploy() -> None:
    from src.release.workflow import build_release_record

    record = build_release_record(
        job_name="job-123",
        registered_model_name="churn-model",
        registered_model_version="10",
        promotion_decision={"status": "promote", "primary_metric": "f1"},
        candidate_metrics={"model_name": "rf", "f1": 0.8, "roc_auc": 0.85},
        endpoint_name="churn-endpoint",
        deployment_name="blue",
        deployment_metadata={
            "deployment_state": "Succeeded",
            "recovery_used": False,
            "finalization_timed_out": False,
            "traffic_updated": True,
            "smoke_invoked": False,
            "smoke_test_response": None,
        },
        status="failed",
        model_resolution="reused",
        failure={
            "failure_stage": "deployment",
            "error_type": "ValueError",
            "error_message": "Endpoint payload input_data row 0 expected 10 features, got 3.",
        },
    )

    assert record["monitoring_handoff"]["status"] == "canary_failed_after_deploy"
    assert record["monitoring_handoff"]["deployment_state"] == "Succeeded"
    assert record["monitoring_handoff"]["traffic_updated"] is True
    assert record["monitoring_handoff"]["canary_invoked"] is False


def test_build_release_record_marks_disabled_release_evidence_mode_as_basic_handoff() -> None:
    from src.release.workflow import build_release_record

    record = build_release_record(
        job_name="job-123",
        registered_model_name="churn-model",
        registered_model_version="10",
        promotion_decision={"status": "promote", "primary_metric": "f1"},
        candidate_metrics={"model_name": "rf", "f1": 0.8, "roc_auc": 0.85},
        endpoint_name="churn-endpoint",
        deployment_name="blue",
        deployment_metadata={
            "deployment_state": "Succeeded",
            "recovery_used": False,
            "finalization_timed_out": False,
            "traffic_updated": True,
            "smoke_invoked": True,
            "smoke_test_response": "ok",
            "inference_capture_enabled": False,
            "inference_capture_mode": "release_evidence_only",
            "inference_capture_status": "disabled",
            "inference_capture_warnings": [
                "Azure ML data_collector is disabled for repo-owned online scoring; using release-evidence-only monitoring handoff."
            ],
            "inference_capture_output_path": None,
        },
        canary_inference={
            "payload": {
                "path": "sample-data.json",
                "format": "input_data_2d_array",
                "row_count": 1,
                "feature_count": 10,
                "validation_status": "passed",
            },
            "endpoint": {"endpoint_name": "churn-endpoint", "deployment_name": "blue"},
            "model": {"name": "churn-model", "version": "10"},
            "response": {"preview": "ok"},
        },
    )

    assert record["monitoring_handoff"]["status"] == "ready_for_basic_monitoring_handoff"
    assert record["monitoring_handoff"]["inference_capture_mode"] == "release_evidence_only"
    assert record["monitoring_handoff"]["inference_capture_enabled"] is False
    assert record["monitoring_handoff"]["inference_capture_status"] == "disabled"
    assert record["deployment_capture"] == {
        "status": "disabled",
        "mode": "release_evidence_only",
        "enabled": False,
        "warnings": [
            "Azure ML data_collector is disabled for repo-owned online scoring; using release-evidence-only monitoring handoff."
        ],
        "output_path": None,
        "evidence_plane": "deployment_owned",
    }


def test_build_release_record_marks_capture_warning_as_degraded_handoff() -> None:
    from src.release.workflow import build_release_record

    record = build_release_record(
        job_name="job-123",
        registered_model_name="churn-model",
        registered_model_version="10",
        promotion_decision={"status": "promote", "primary_metric": "f1"},
        candidate_metrics={"model_name": "rf", "f1": 0.8, "roc_auc": 0.85},
        endpoint_name="churn-endpoint",
        deployment_name="blue",
        deployment_metadata={
            "deployment_state": "Succeeded",
            "recovery_used": False,
            "finalization_timed_out": False,
            "traffic_updated": True,
            "smoke_invoked": True,
            "smoke_test_response": "ok",
            "inference_capture_enabled": True,
            "inference_capture_mode": "jsonl_file",
            "inference_capture_status": "degraded",
            "inference_capture_warnings": ["capture sink failed to initialize"],
            "inference_capture_output_path": "/tmp/inference-capture",
        },
        canary_inference={
            "payload": {
                "path": "sample-data.json",
                "format": "input_data_2d_array",
                "row_count": 1,
                "feature_count": 10,
                "validation_status": "passed",
            },
            "endpoint": {"endpoint_name": "churn-endpoint", "deployment_name": "blue"},
            "model": {"name": "churn-model", "version": "10"},
            "response": {"preview": "ok"},
        },
    )

    assert record["monitoring_handoff"]["status"] == "capture_degraded_after_deploy"
    assert record["monitoring_handoff"]["inference_capture_enabled"] is True
    assert record["monitoring_handoff"]["inference_capture_status"] == "degraded"
    assert record["monitoring_handoff"]["inference_capture_warnings"] == [
        "capture sink failed to initialize"
    ]
    assert record["deployment_capture"]["status"] == "degraded"
    assert record["deployment_capture"]["enabled"] is True
    assert record["deployment_capture"]["mode"] == "jsonl_file"


def test_build_release_record_marks_repo_owned_capture_as_ready_monitoring_handoff() -> None:
    """
    @proves online-deploy.surface-deployment-owned-capture-evidence-release-record-json
    """
    from src.release.workflow import build_release_record

    record = build_release_record(
        job_name="job-123",
        registered_model_name="churn-model",
        registered_model_version="10",
        promotion_decision={"status": "promote", "primary_metric": "f1"},
        candidate_metrics={"model_name": "rf", "f1": 0.8, "roc_auc": 0.85},
        endpoint_name="churn-endpoint",
        deployment_name="blue",
        deployment_metadata={
            "deployment_state": "Succeeded",
            "recovery_used": False,
            "finalization_timed_out": False,
            "traffic_updated": True,
            "smoke_invoked": True,
            "smoke_test_response": "ok",
            "inference_capture_enabled": True,
            "inference_capture_mode": "jsonl_file",
            "inference_capture_status": "healthy",
            "inference_capture_output_path": "/mnt/capture",
            "inference_capture_warnings": [],
        },
        canary_inference={
            "payload": {
                "path": "sample-data.json",
                "format": "input_data_2d_array",
                "row_count": 1,
                "feature_count": 10,
                "validation_status": "passed",
            }
        },
    )

    assert record["monitoring_handoff"]["status"] == "ready_for_repo_owned_inference_capture_handoff"
    assert record["monitoring_handoff"]["inference_capture_enabled"] is True
    assert record["monitoring_handoff"]["inference_capture_status"] == "healthy"
    assert record["monitoring_handoff"]["inference_capture_output_path"] == "/mnt/capture"
    assert record["deployment_capture"]["status"] == "healthy"
    assert record["deployment_capture"]["output_path"] == "/mnt/capture"


def test_build_release_record_marks_repo_owned_capture_warning_as_degraded() -> None:
    from src.release.workflow import build_release_record

    record = build_release_record(
        job_name="job-123",
        registered_model_name="churn-model",
        registered_model_version="10",
        promotion_decision={"status": "promote", "primary_metric": "f1"},
        candidate_metrics={"model_name": "rf", "f1": 0.8, "roc_auc": 0.85},
        endpoint_name="churn-endpoint",
        deployment_name="blue",
        deployment_metadata={
            "deployment_state": "Succeeded",
            "recovery_used": False,
            "finalization_timed_out": False,
            "traffic_updated": True,
            "smoke_invoked": True,
            "smoke_test_response": "ok",
            "inference_capture_enabled": True,
            "inference_capture_mode": "jsonl_file",
            "inference_capture_status": "degraded",
            "inference_capture_output_path": "/mnt/capture",
            "inference_capture_warnings": ["permission denied"],
        },
        canary_inference={
            "payload": {
                "path": "sample-data.json",
                "format": "input_data_2d_array",
                "row_count": 1,
                "feature_count": 10,
                "validation_status": "passed",
            }
        },
    )

    assert record["monitoring_handoff"]["status"] == "capture_degraded_after_deploy"
    assert record["monitoring_handoff"]["inference_capture_status"] == "degraded"
    assert record["monitoring_handoff"]["inference_capture_warnings"] == ["permission denied"]


def test_build_release_record_keeps_configured_external_capture_at_basic_handoff_until_proven() -> None:
    """
    @proves online-deploy.externalize-repo-owned-inference-capture-azure-accessible-jsonl
    """
    from src.release.workflow import build_release_record

    record = build_release_record(
        job_name="job-123",
        registered_model_name="churn-model",
        registered_model_version="10",
        promotion_decision={"status": "promote", "primary_metric": "f1"},
        candidate_metrics={"model_name": "rf", "f1": 0.8, "roc_auc": 0.85},
        endpoint_name="churn-endpoint",
        deployment_name="blue",
        deployment_metadata={
            "deployment_state": "Succeeded",
            "recovery_used": False,
            "finalization_timed_out": False,
            "traffic_updated": True,
            "smoke_invoked": True,
            "smoke_test_response": "ok",
            "inference_capture_enabled": True,
            "inference_capture_mode": "workspaceblobstore_jsonl",
            "inference_capture_status": "configured",
            "inference_capture_output_path": "azureblob://workspaceblobstore/monitoring/monitoring/inference_capture",
            "inference_capture_warnings": [],
        },
        canary_inference={
            "payload": {
                "path": "sample-data.json",
                "format": "input_data_2d_array",
                "row_count": 1,
                "feature_count": 10,
                "validation_status": "passed",
            }
        },
    )

    assert record["monitoring_handoff"]["status"] == "ready_for_basic_monitoring_handoff"
    assert record["monitoring_handoff"]["evidence_level"] == "release_evidence_only"
    assert record["monitoring_handoff"]["inference_capture_status"] == "configured"
