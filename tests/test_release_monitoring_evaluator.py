"""
@meta
type: test
scope: unit
domain: monitoring
covers:
  - First monitor-stage evaluator for release handoff artifacts
excludes:
  - Real Azure ML calls
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import pytest


def _local_temp_dir() -> Path:
    temp_dir = Path(".tmp-tests") / f"monitor-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def _write_capture_records(capture_path: Path, records: list[dict[str, object]]) -> None:
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    capture_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )


def _build_release_record(
    *,
    inference_capture_enabled: bool = False,
    inference_capture_mode: str = "release_evidence_only",
    inference_capture_status: str = "disabled",
    inference_capture_output_path: str | None = None,
    inference_capture_warnings: list[str] | None = None,
) -> dict[str, object]:
    from src.release.workflow import build_release_record

    return build_release_record(
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
            "inference_capture_enabled": inference_capture_enabled,
            "inference_capture_mode": inference_capture_mode,
            "inference_capture_status": inference_capture_status,
            "inference_capture_output_path": inference_capture_output_path,
            "inference_capture_warnings": inference_capture_warnings or [],
            "repo_owned_scoring_expected": True,
            "repo_owned_scoring_observed": False,
            "repo_owned_scoring_status": "generated_runtime_still_in_control",
            "repo_owned_scoring_log_markers": [],
            "repo_owned_scoring_warnings": [
                "Azure deployment logs did not show repo-owned scorer proof markers after canary invoke."
            ],
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


def test_evaluate_release_record_blocks_when_release_record_is_missing() -> None:
    """
    @proves monitor.load-validate-release-records-produced-run-release-py
    """
    from src.monitoring.evaluate_release import evaluate_release_monitoring

    temp_path = _local_temp_dir()
    try:
        result = evaluate_release_monitoring(
            release_record_path=temp_path / "missing.json",
            config_path=Path("configs/monitor.yaml"),
            output_dir=temp_path / "monitor-output",
        )

        assert result["monitor_status"] == "blocked"
        assert result["retraining_policy"]["trigger"] == "investigate_before_retraining"
        assert result["checks"]["release_record_present"] is False
        assert "release_record.json was not found" in result["recommended_action"]
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def test_evaluate_release_record_marks_release_evidence_only_handoff_as_limited_but_healthy(
) -> None:
    """
    @proves monitor.classify-monitoring-outcomes-limited-but-healthy-capture-backed
    @proves monitor.evaluate-always-release-evidence-checks-even-when-capture
    @proves monitor.preserve-truthful-monitoring-semantics-whether-release-record-reports
    @proves monitor.expose-deployment-owned-capture-truth-caller-side-capture
    """
    from src.monitoring.evaluate_release import evaluate_release_monitoring

    temp_path = _local_temp_dir()
    try:
        release_record_path = temp_path / "release_record.json"
        release_record_path.write_text(
            json.dumps(_build_release_record()),
            encoding="utf-8",
        )

        result = evaluate_release_monitoring(
            release_record_path=release_record_path,
            config_path=Path("configs/monitor.yaml"),
            output_dir=temp_path / "monitor-output",
        )

        assert result["monitor_status"] == "limited_but_healthy"
        assert result["evidence_level"] == "release_evidence_only"
        assert result["monitoring_handoff_status"] == "ready_for_basic_monitoring_handoff"
        assert result["runtime_contract"] == "generated_runtime_still_in_control"
        assert result["capture_status"] == "not_available_for_this_runtime_contract"
        assert result["capture_evidence_source"] == "release_evidence_only"
        assert result["deployment_capture"]["status"] == "disabled"
        assert result["caller_capture"]["status"] == "not_run"
        assert result["retraining_policy"]["trigger"] == "no_retraining_signal"
        assert result["retraining_policy"]["recommended_training_path"] is None
        assert result["retraining_policy"]["path_recommendation"] == "none"
        assert result["retraining_policy"]["path_recommendation_reason_codes"] == [
            "no_retraining_signal"
        ]
        assert result["retraining_policy"]["drift_severity"] == "low"
        assert result["retraining_policy"]["signal_persistence"] == "single_event"
        assert result["retraining_policy"]["policy_confidence"] == "strong"
        assert result["retraining_policy"]["requires_dataset_freeze"] is False
        assert result["retraining_policy"]["requires_data_validation"] is False
        assert result["checks"]["release_record_present"] is True
        assert result["checks"]["deployment_succeeded"] is True
        assert result["checks"]["canary_passed"] is True
        assert result["checks"]["capture_retrievable"] is False
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def test_evaluate_release_record_marks_capture_backed_when_fixture_records_are_retrievable(
) -> None:
    """
    @proves monitor.classify-monitoring-outcomes-limited-but-healthy-capture-backed
    @proves monitor.optionally-consume-externally-retrievable-sampled-inference-records-file
    @proves monitor.enforce-bounded-production-checks-including-minimum-sample-count
    @proves monitor.expose-deployment-owned-capture-truth-caller-side-capture
    """
    from src.monitoring.evaluate_release import evaluate_release_monitoring

    temp_path = _local_temp_dir()
    try:
        capture_path = temp_path / "capture" / "records.jsonl"
        _write_capture_records(
            capture_path,
            [
                {
                    "timestamp_utc": "2026-04-14T12:00:00Z",
                    "feature_count": 10,
                    "row_count": 1,
                    "outputs": [0],
                },
                {
                    "timestamp_utc": "2026-04-14T12:01:00Z",
                    "feature_count": 10,
                    "row_count": 1,
                    "outputs": [1],
                },
            ],
        )

        release_record_path = temp_path / "release_record.json"
        release_record_path.write_text(
            json.dumps(
                _build_release_record(
                    inference_capture_enabled=True,
                    inference_capture_mode="workspaceblobstore_jsonl",
                    inference_capture_status="healthy",
                    inference_capture_output_path=str(capture_path),
                )
            ),
            encoding="utf-8",
        )

        result = evaluate_release_monitoring(
            release_record_path=release_record_path,
            config_path=Path("configs/monitor.yaml"),
            output_dir=temp_path / "monitor-output",
        )

        assert result["monitor_status"] == "capture_backed"
        assert result["evidence_level"] == "repo_owned_inference_capture"
        assert result["capture_status"] == "healthy"
        assert result["capture_evidence_source"] == "caller_side"
        assert result["deployment_capture"]["status"] == "healthy"
        assert result["caller_capture"]["status"] == "retrieved"
        assert result["retraining_policy"]["trigger"] == "no_retraining_signal"
        assert result["retraining_policy"]["recommended_training_path"] is None
        assert result["retraining_policy"]["path_recommendation"] == "none"
        assert result["retraining_policy"]["path_recommendation_reason_codes"] == [
            "no_retraining_signal"
        ]
        assert result["retraining_policy"]["drift_severity"] == "low"
        assert result["retraining_policy"]["signal_persistence"] == "single_event"
        assert result["retraining_policy"]["policy_confidence"] == "strong"
        assert result["capture_record_count"] == 2
        assert result["checks"]["capture_retrievable"] is True
        assert result["checks"]["capture_schema_consistent"] is True
        assert result["checks"]["prediction_distribution_available"] is True
        assert result["prediction_distribution"] == {"0": 1, "1": 1}
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def test_evaluate_release_record_reads_capture_records_from_directory_tree() -> None:
    """
    @proves monitor.optionally-consume-externally-retrievable-sampled-inference-records-file
    """
    from src.monitoring.evaluate_release import evaluate_release_monitoring

    temp_path = _local_temp_dir()
    try:
        capture_root = temp_path / "capture-root"
        _write_capture_records(
            capture_root / "2026" / "04" / "14" / "a.jsonl",
            [
                {
                    "timestamp_utc": "2026-04-14T12:00:00Z",
                    "feature_count": 10,
                    "row_count": 1,
                    "outputs": [0],
                }
            ],
        )
        _write_capture_records(
            capture_root / "2026" / "04" / "15" / "b.jsonl",
            [
                {
                    "timestamp_utc": "2026-04-15T12:01:00Z",
                    "feature_count": 10,
                    "row_count": 1,
                    "outputs": [1],
                }
            ],
        )

        release_record_path = temp_path / "release_record.json"
        release_record_path.write_text(
            json.dumps(
                _build_release_record(
                    inference_capture_enabled=True,
                    inference_capture_mode="workspaceblobstore_jsonl",
                    inference_capture_status="healthy",
                    inference_capture_output_path=str(capture_root),
                )
            ),
            encoding="utf-8",
        )

        result = evaluate_release_monitoring(
            release_record_path=release_record_path,
            config_path=Path("configs/monitor.yaml"),
            output_dir=temp_path / "monitor-output",
        )

        assert result["monitor_status"] == "capture_backed"
        assert result["capture_record_count"] == 2
        assert result["prediction_distribution"] == {"0": 1, "1": 1}
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def test_evaluate_release_record_downgrades_missing_capture_sink_to_degraded() -> None:
    from src.monitoring.evaluate_release import evaluate_release_monitoring

    temp_path = _local_temp_dir()
    try:
        release_record_path = temp_path / "release_record.json"
        release_record_path.write_text(
            json.dumps(
                _build_release_record(
                    inference_capture_enabled=True,
                    inference_capture_mode="workspaceblobstore_jsonl",
                    inference_capture_status="healthy",
                    inference_capture_output_path=str(temp_path / "missing-capture.jsonl"),
                )
            ),
            encoding="utf-8",
        )

        result = evaluate_release_monitoring(
            release_record_path=release_record_path,
            config_path=Path("configs/monitor.yaml"),
            output_dir=temp_path / "monitor-output",
        )

        assert result["monitor_status"] == "degraded"
        assert result["checks"]["capture_retrievable"] is False
        assert result["capture_evidence_source"] == "deployment_configured_but_unretrieved"
        assert result["caller_capture"]["status"] == "missing"
        assert result["retraining_policy"]["trigger"] == "investigate_before_retraining"
        assert result["retraining_policy"]["path_recommendation"] == "none"
        assert result["retraining_policy"]["path_recommendation_reason_codes"] == [
            "investigate_before_retraining"
        ]
        assert result["retraining_policy"]["drift_severity"] == "medium"
        assert result["retraining_policy"]["signal_persistence"] == "single_event"
        assert result["retraining_policy"]["policy_confidence"] == "moderate"
        assert "capture evidence was expected but could not be retrieved" in result[
            "recommended_action"
        ]
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def test_evaluate_release_record_degrades_when_single_class_share_exceeds_threshold() -> None:
    """
    @proves monitor.emit-retraining-decision-json-plus-embedded-retraining-policy
    @proves monitor.enrich-retraining-policy-bounded-path-recommendation-evidence-such
    @proves monitor.recommend-bounded-next-steps-such-dataset-freeze-validate
    @proves monitor.enforce-bounded-production-checks-including-minimum-sample-count
    """
    from src.monitoring.evaluate_release import evaluate_release_monitoring

    temp_path = _local_temp_dir()
    try:
        capture_path = temp_path / "capture" / "records.jsonl"
        _write_capture_records(
            capture_path,
            [
                {
                    "timestamp_utc": "2026-04-14T12:00:00Z",
                    "feature_count": 10,
                    "row_count": 1,
                    "outputs": [0],
                },
                {
                    "timestamp_utc": "2026-04-14T12:01:00Z",
                    "feature_count": 10,
                    "row_count": 1,
                    "outputs": [0],
                },
                {
                    "timestamp_utc": "2026-04-14T12:02:00Z",
                    "feature_count": 10,
                    "row_count": 1,
                    "outputs": [0],
                },
                {
                    "timestamp_utc": "2026-04-14T12:03:00Z",
                    "feature_count": 10,
                    "row_count": 1,
                    "outputs": [1],
                },
            ],
        )
        config_path = temp_path / "monitor.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "monitor:",
                    "  evidence:",
                    "    require_release_record: true",
                    "    allow_release_evidence_only: true",
                    "    require_capture_for_capture_checks: true",
                    "  thresholds:",
                    "    min_capture_records: 1",
                    "    expected_feature_count: 10",
                    "    max_single_class_share: 0.7",
                    "  retraining_policy:",
                    "    enabled: true",
                    "    policy_version: 1",
                    "    candidate_capture_statuses:",
                    "      - class_balance_exceeded",
                    "    investigate_monitor_statuses:",
                    "      - degraded",
                    "      - blocked",
                    "    default_training_path: fixed_train",
                    "  output:",
                    "    write_json_summary: true",
                    "    write_markdown_report: true",
                ]
            ),
            encoding="utf-8",
        )
        release_record_path = temp_path / "release_record.json"
        release_record_path.write_text(
            json.dumps(
                _build_release_record(
                    inference_capture_enabled=True,
                    inference_capture_mode="workspaceblobstore_jsonl",
                    inference_capture_status="healthy",
                    inference_capture_output_path=str(capture_path),
                )
            ),
            encoding="utf-8",
        )

        result = evaluate_release_monitoring(
            release_record_path=release_record_path,
            config_path=config_path,
            output_dir=temp_path / "monitor-output",
        )

        assert result["monitor_status"] == "degraded"
        assert result["checks"]["prediction_class_balance_within_threshold"] is False
        assert result["retraining_policy"]["trigger"] == "retraining_candidate"
        assert result["retraining_policy"]["recommended_training_path"] == "fixed_train"
        assert result["retraining_policy"]["path_recommendation"] == "fixed_train"
        assert result["retraining_policy"]["path_recommendation_reason_codes"] == [
            "bounded_refresh_candidate"
        ]
        assert result["retraining_policy"]["drift_severity"] == "medium"
        assert result["retraining_policy"]["signal_persistence"] == "single_event"
        assert result["retraining_policy"]["policy_confidence"] == "moderate"
        assert result["retraining_policy"]["requires_dataset_freeze"] is True
        assert result["retraining_policy"]["requires_data_validation"] is True
        assert "prediction_class_balance_exceeded" in result["retraining_policy"]["reason_codes"]
        assert "prediction class share exceeded threshold" in result["recommended_action"]
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def test_run_monitor_writes_bounded_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    @proves monitor.emit-bounded-monitoring-artifacts-such-monitor-summary-json
    @proves monitor.emit-retraining-decision-json-plus-embedded-retraining-policy
    """
    import run_monitor

    temp_path = _local_temp_dir()
    try:
        release_record_path = temp_path / "release_record.json"
        release_record_path.write_text(
            json.dumps(_build_release_record()),
            encoding="utf-8",
        )
        output_dir = temp_path / "monitor-output"

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_monitor.py",
                "--release-record",
                str(release_record_path),
                "--config",
                "configs/monitor.yaml",
                "--output-dir",
                str(output_dir),
            ],
        )

        run_monitor.main()

        summary = json.loads((output_dir / "monitor_summary.json").read_text(encoding="utf-8"))
        report = (output_dir / "monitor_report.md").read_text(encoding="utf-8")
        retraining_decision = json.loads(
            (output_dir / "retraining_decision.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (output_dir / "monitor_manifest" / "step_manifest.json").read_text(encoding="utf-8")
        )

        assert summary["monitor_status"] == "limited_but_healthy"
        assert summary["retraining_policy"]["trigger"] == "no_retraining_signal"
        assert summary["retraining_policy"]["path_recommendation"] == "none"
        assert summary["retraining_policy"]["policy_confidence"] == "strong"
        assert "# Monitoring Report" in report
        assert "## Retraining Policy" in report
        assert "Path recommendation" in report
        assert "Policy confidence" in report
        assert retraining_decision["trigger"] == "no_retraining_signal"
        assert retraining_decision["path_recommendation"] == "none"
        assert retraining_decision["policy_confidence"] == "strong"
        assert retraining_decision["recommendation_summary"]["trigger"] == "no_retraining_signal"
        assert retraining_decision["recommendation_summary"]["path_recommendation"] == "none"
        assert retraining_decision["recommendation_summary"]["policy_confidence"] == "strong"
        assert (
            retraining_decision["recommendation_summary"]["recommended_action"]
            == summary["recommended_action"]
        )
        assert manifest["status"] == "succeeded"
        assert manifest["outputs"]["monitor_summary"]["path"].endswith("monitor_summary.json")
        assert manifest["outputs"]["retraining_decision"]["path"].endswith("retraining_decision.json")
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def test_run_monitor_accepts_capture_path_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    @proves monitor.support-capture-path-override-scheduled-operator-driven-runs
    """
    import run_monitor

    temp_path = _local_temp_dir()
    try:
        release_record_path = temp_path / "release_record.json"
        release_record_path.write_text(
            json.dumps(
                _build_release_record(
                    inference_capture_enabled=True,
                    inference_capture_mode="workspaceblobstore_jsonl",
                    inference_capture_status="healthy",
                    inference_capture_output_path="azureblob://workspaceblobstore/monitoring/inference_capture",
                )
            ),
            encoding="utf-8",
        )
        capture_path = temp_path / "capture" / "records.jsonl"
        _write_capture_records(
            capture_path,
            [
                {
                    "timestamp_utc": "2026-04-14T12:00:00Z",
                    "feature_count": 10,
                    "row_count": 1,
                    "outputs": [0],
                },
                {
                    "timestamp_utc": "2026-04-14T12:01:00Z",
                    "feature_count": 10,
                    "row_count": 1,
                    "outputs": [1],
                }
            ],
        )
        output_dir = temp_path / "monitor-output"

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_monitor.py",
                "--release-record",
                str(release_record_path),
                "--config",
                "configs/monitor.yaml",
                "--capture-path",
                str(capture_path),
                "--output-dir",
                str(output_dir),
            ],
        )

        run_monitor.main()

        summary = json.loads((output_dir / "monitor_summary.json").read_text(encoding="utf-8"))
        assert summary["monitor_status"] == "capture_backed"
        assert summary["capture_output_path"] == str(capture_path)
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def test_run_monitor_supports_split_output_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_monitor

    temp_path = _local_temp_dir()
    try:
        release_record_path = temp_path / "release_record.json"
        release_record_path.write_text(
            json.dumps(_build_release_record()),
            encoding="utf-8",
        )
        output_dir = temp_path / "monitor-output"
        summary_path = temp_path / "named-outputs" / "summary" / "monitor_summary.json"
        report_path = temp_path / "named-outputs" / "report" / "monitor_report.md"
        manifest_path = temp_path / "named-outputs" / "manifest" / "step_manifest.json"

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_monitor.py",
                "--release-record",
                str(release_record_path),
                "--config",
                "configs/monitor.yaml",
                "--output-dir",
                str(output_dir),
                "--summary-output",
                str(summary_path),
                "--report-output",
                str(report_path),
                "--manifest-output",
                str(manifest_path),
            ],
        )

        run_monitor.main()

        assert summary_path.exists()
        assert report_path.exists()
        assert manifest_path.exists()
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)
