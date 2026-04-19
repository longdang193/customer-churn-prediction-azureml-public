"""
@meta
type: test
scope: unit
domain: inference-monitoring
covers:
  - Caller-side endpoint invocation capture wrapper
  - Local JSONL capture mode
  - Capture manifest output
excludes:
  - Real Azure ML endpoint calls
  - Real Azure Blob writes
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


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"client-capture-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def _write_payload(path: Path) -> None:
    path.write_text(
        json.dumps({"input_data": [[float(index) for index in range(10)]]}),
        encoding="utf-8",
    )


def test_invoke_with_capture_writes_redacted_local_jsonl_record() -> None:
    """
    @proves monitor.provide-caller-side-inference-capture-wrapper-invokes-managed
    """
    from src.inference.client_capture import (
        CallerCaptureRequest,
        CallerInferenceCaptureSettings,
        invoke_with_capture,
    )

    temp_dir = _make_temp_dir()
    try:
        payload_path = temp_dir / "payload.json"
        _write_payload(payload_path)

        result = invoke_with_capture(
            request=CallerCaptureRequest(
                endpoint_name="churn-endpoint",
                deployment_name="churn-deployment",
                request_file=payload_path,
                model_name="churn-model",
                model_version="12",
            ),
            settings=CallerInferenceCaptureSettings(
                enabled=True,
                mode="caller_local_jsonl",
                sample_rate=1.0,
                max_rows_per_request=1,
                redact_inputs=True,
                output_path=str(temp_dir / "capture"),
            ),
            endpoint_invoker=lambda **_: "[1]",
            random_value_factory=lambda: 0.0,
        )

        capture_files = list((temp_dir / "capture").rglob("*.jsonl"))
        assert result.response == "[1]"
        assert result.response_preview == "[1]"
        assert result.capture_status == "healthy"
        assert result.capture_path == str(capture_files[0])
        record = json.loads(capture_files[0].read_text(encoding="utf-8"))
        assert record["capture_source"] == "caller_side_wrapper"
        assert record["endpoint_name"] == "churn-endpoint"
        assert record["deployment_name"] == "churn-deployment"
        assert record["model_name"] == "churn-model"
        assert record["model_version"] == "12"
        assert record["feature_count"] == 10
        assert record["inputs"] == [{"feature_count": 10}]
        assert record["outputs"] == [1]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_invoke_with_capture_can_store_unredacted_rows_for_smoke_fixtures() -> None:
    from src.inference.client_capture import (
        CallerCaptureRequest,
        CallerInferenceCaptureSettings,
        invoke_with_capture,
    )

    temp_dir = _make_temp_dir()
    try:
        payload_path = temp_dir / "payload.json"
        _write_payload(payload_path)

        invoke_with_capture(
            request=CallerCaptureRequest(
                endpoint_name="churn-endpoint",
                deployment_name="churn-deployment",
                request_file=payload_path,
            ),
            settings=CallerInferenceCaptureSettings(
                enabled=True,
                mode="caller_local_jsonl",
                sample_rate=1.0,
                max_rows_per_request=1,
                redact_inputs=False,
                output_path=str(temp_dir / "capture"),
            ),
            endpoint_invoker=lambda **_: "[0]",
            random_value_factory=lambda: 0.0,
        )

        capture_file = next((temp_dir / "capture").rglob("*.jsonl"))
        record = json.loads(capture_file.read_text(encoding="utf-8"))
        assert record["inputs"] == [
            {
                str(index): float(index)
                for index in range(10)
            }
        ]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_blob_capture_mode_reports_missing_env_vars() -> None:
    from src.inference.client_capture import (
        CallerCaptureRequest,
        CallerInferenceCaptureSettings,
        invoke_with_capture,
    )

    temp_dir = _make_temp_dir()
    try:
        payload_path = temp_dir / "payload.json"
        _write_payload(payload_path)

        result = invoke_with_capture(
            request=CallerCaptureRequest(
                endpoint_name="churn-endpoint",
                deployment_name="churn-deployment",
                request_file=payload_path,
            ),
            settings=CallerInferenceCaptureSettings(
                enabled=True,
                mode="caller_blob_jsonl",
                sample_rate=1.0,
                max_rows_per_request=1,
                redact_inputs=True,
                output_path="monitoring/inference_capture",
                storage_connection_string_env="MISSING_CONNECTION",
                storage_container_env="MISSING_CONTAINER",
            ),
            endpoint_invoker=lambda **_: "[0]",
        )

        assert result.capture_status == "degraded"
        assert result.capture_path is None
        assert "MISSING_CONNECTION" in result.warnings[0]
        assert "MISSING_CONTAINER" in result.warnings[1]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_blob_capture_mode_writes_with_env_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.inference.client_capture import (
        CallerCaptureRequest,
        CallerInferenceCaptureSettings,
        invoke_with_capture,
    )

    temp_dir = _make_temp_dir()
    captured: dict[str, object] = {}

    class FakeBlobClient:
        def upload_blob(self, data: str, *, overwrite: bool = False) -> None:
            captured["data"] = data
            captured["overwrite"] = overwrite

    class FakeServiceClient:
        account_name = "workspaceblobstore"

        def get_blob_client(self, *, container: str, blob: str) -> FakeBlobClient:
            captured["container"] = container
            captured["blob"] = blob
            return FakeBlobClient()

    try:
        payload_path = temp_dir / "payload.json"
        _write_payload(payload_path)
        monkeypatch.setenv("CAPTURE_CONNECTION", "UseDevelopmentStorage=true")
        monkeypatch.setenv("CAPTURE_CONTAINER", "monitoring")

        result = invoke_with_capture(
            request=CallerCaptureRequest(
                endpoint_name="churn-endpoint",
                deployment_name="churn-deployment",
                request_file=payload_path,
            ),
            settings=CallerInferenceCaptureSettings(
                enabled=True,
                mode="caller_blob_jsonl",
                sample_rate=1.0,
                max_rows_per_request=1,
                redact_inputs=True,
                output_path="monitoring/inference_capture",
                storage_connection_string_env="CAPTURE_CONNECTION",
                storage_container_env="CAPTURE_CONTAINER",
            ),
            endpoint_invoker=lambda **_: "[0]",
            random_value_factory=lambda: 0.0,
            blob_service_client_factory=lambda _: FakeServiceClient(),
        )

        assert result.capture_status == "healthy"
        assert result.capture_path is not None
        assert captured["container"] == "monitoring"
        assert str(captured["blob"]).startswith(
            "monitoring/inference_capture/"
        )
        assert json.loads(str(captured["data"]).strip())["capture_source"] == (
            "caller_side_wrapper"
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_caller_capture_settings_reads_yaml_defaults() -> None:
    from src.inference.client_capture import load_caller_capture_settings

    temp_dir = _make_temp_dir()
    try:
        config_path = temp_dir / "inference_capture.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "inference_capture:",
                    "  enabled: true",
                    "  mode: caller_local_jsonl",
                    "  sample_rate: 0.25",
                    "  max_rows_per_request: 2",
                    "  redact_inputs: true",
                    "  output_path: .tmp-tests/client-capture",
                    "  storage_connection_string_env: CAPTURE_CONNECTION",
                    "  storage_container_env: CAPTURE_CONTAINER",
                ]
            ),
            encoding="utf-8",
        )

        settings = load_caller_capture_settings(config_path)

        assert settings.enabled is True
        assert settings.mode == "caller_local_jsonl"
        assert settings.sample_rate == 0.25
        assert settings.max_rows_per_request == 2
        assert settings.redact_inputs is True
        assert settings.output_path == ".tmp-tests/client-capture"
        assert settings.storage_connection_string_env == "CAPTURE_CONNECTION"
        assert settings.storage_container_env == "CAPTURE_CONTAINER"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_caller_capture_settings_preserves_zero_sample_rate() -> None:
    from src.inference.client_capture import load_caller_capture_settings

    temp_dir = _make_temp_dir()
    try:
        config_path = temp_dir / "inference_capture.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "inference_capture:",
                    "  enabled: true",
                    "  mode: caller_local_jsonl",
                    "  sample_rate: 0",
                    "  output_path: .tmp-tests/client-capture",
                ]
            ),
            encoding="utf-8",
        )

        settings = load_caller_capture_settings(config_path)

        assert settings.sample_rate == 0.0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_caller_written_records_feed_monitor_capture_backed() -> None:
    from src.inference.client_capture import (
        CallerCaptureRequest,
        CallerInferenceCaptureSettings,
        invoke_with_capture,
    )
    from src.monitoring.evaluate_release import evaluate_release_monitoring
    from tests.test_release_monitoring_evaluator import _build_release_record

    temp_dir = _make_temp_dir()
    try:
        payload_path = temp_dir / "payload.json"
        _write_payload(payload_path)

        first_result = invoke_with_capture(
            request=CallerCaptureRequest(
                endpoint_name="churn-endpoint",
                deployment_name="churn-deployment",
                request_file=payload_path,
            ),
            settings=CallerInferenceCaptureSettings(
                enabled=True,
                mode="caller_local_jsonl",
                sample_rate=1.0,
                max_rows_per_request=1,
                redact_inputs=True,
                output_path=str(temp_dir / "capture"),
            ),
            endpoint_invoker=lambda **_: "[0]",
            random_value_factory=lambda: 0.0,
        )
        second_result = invoke_with_capture(
            request=CallerCaptureRequest(
                endpoint_name="churn-endpoint",
                deployment_name="churn-deployment",
                request_file=payload_path,
            ),
            settings=CallerInferenceCaptureSettings(
                enabled=True,
                mode="caller_local_jsonl",
                sample_rate=1.0,
                max_rows_per_request=1,
                redact_inputs=True,
                output_path=str(temp_dir / "capture"),
            ),
            endpoint_invoker=lambda **_: "[1]",
            random_value_factory=lambda: 0.0,
        )
        release_record_path = temp_dir / "release_record.json"
        assert Path(str(first_result.capture_path)).exists()
        assert Path(str(second_result.capture_path)).exists()
        release_record_path.write_text(
            json.dumps(
                _build_release_record(
                    inference_capture_enabled=True,
                    inference_capture_mode="caller_local_jsonl",
                    inference_capture_status="healthy",
                    inference_capture_output_path=str(temp_dir / "capture"),
                )
            ),
            encoding="utf-8",
        )

        summary = evaluate_release_monitoring(
            release_record_path=release_record_path,
            config_path=Path("configs/monitor.yaml"),
            output_dir=temp_dir / "monitor",
        )

        assert summary["monitor_status"] == "capture_backed"
        assert summary["capture_record_count"] == 2
        assert summary["prediction_distribution"] == {"0": 1, "1": 1}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_run_inference_capture_writes_manifest_with_response_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    @proves monitor.provide-caller-side-inference-capture-wrapper-invokes-managed
    """
    import run_inference_capture

    temp_dir = _make_temp_dir()
    try:
        payload_path = temp_dir / "payload.json"
        _write_payload(payload_path)
        config_path = temp_dir / "inference_capture.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "inference_capture:",
                    "  enabled: true",
                    "  mode: caller_local_jsonl",
                    "  output_path: " + str(temp_dir / "capture"),
                ]
            ),
            encoding="utf-8",
        )
        manifest_path = temp_dir / "capture_manifest.json"

        class FakeEndpoints:
            def invoke(self, **_: object) -> str:
                return "[1]"

        class FakeClient:
            online_endpoints = FakeEndpoints()

        monkeypatch.setattr(
            run_inference_capture,
            "get_ml_client",
            lambda _: FakeClient(),
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_inference_capture.py",
                "--endpoint-name",
                "churn-endpoint",
                "--deployment-name",
                "churn-deployment",
                "--request-file",
                str(payload_path),
                "--config",
                str(config_path),
                "--output-manifest",
                str(manifest_path),
                "--model-name",
                "churn-model",
                "--model-version",
                "12",
            ],
        )

        run_inference_capture.main()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "succeeded"
        assert manifest["response_preview"] == "[1]"
        assert manifest["capture_status"] == "healthy"
        assert Path(str(manifest["capture_path"])).exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
