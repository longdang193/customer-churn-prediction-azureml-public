"""
@meta
type: test
scope: unit
domain: inference-serving
covers:
  - Repo-owned inference capture record shaping
  - JSONL capture sink writes
  - Sampling and bounded row behavior
excludes:
  - Real Azure Blob or managed online endpoint writes
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import pandas as pd


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"inference-capture-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_build_capture_record_limits_rows_and_redacts_inputs() -> None:
    """
    @proves monitor.provide-caller-side-inference-capture-wrapper-invokes-managed
    """
    from src.inference.capture import (
        RepoOwnedInferenceCaptureSettings,
        build_capture_record,
    )

    settings = RepoOwnedInferenceCaptureSettings(
        enabled=True,
        mode="jsonl_file",
        sample_rate=1.0,
        max_rows_per_request=1,
        capture_inputs=True,
        capture_outputs=True,
        redact_inputs=True,
        output_path="/tmp/inference-capture",
    )

    record = build_capture_record(
        settings=settings,
        endpoint_name="churn-endpoint",
        deployment_name="blue",
        model_name="churn-model",
        model_version="12",
        request_id="request-123",
        input_df=pd.DataFrame(
            [
                {"feature_0": 1.0, "feature_1": 2.0},
                {"feature_0": 3.0, "feature_1": 4.0},
            ]
        ),
        predictions=[0, 1],
    )

    assert record["row_count"] == 2
    assert record["captured_row_count"] == 1
    assert record["feature_count"] == 2
    assert record["inputs"] == [{"feature_count": 2}]
    assert record["outputs"] == [0]


def test_jsonl_capture_sink_writes_partitioned_records() -> None:
    """
    @proves monitor.provide-caller-side-inference-capture-wrapper-invokes-managed
    """
    from src.inference.capture import FileInferenceCaptureSink

    temp_dir = _make_temp_dir()
    try:
        sink = FileInferenceCaptureSink(base_path=temp_dir)
        sink.write_record(
            endpoint_name="churn-endpoint",
            deployment_name="blue",
            record={
                "timestamp_utc": "2026-04-14T00:00:00+00:00",
                "request_id": "request-123",
                "outputs": [0],
            },
        )

        records = list(temp_dir.rglob("*.jsonl"))
        assert len(records) == 1
        payloads = [json.loads(line) for line in records[0].read_text(encoding="utf-8").splitlines()]
        assert payloads == [
            {
                "timestamp_utc": "2026-04-14T00:00:00+00:00",
                "request_id": "request-123",
                "outputs": [0],
            }
        ]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_should_capture_honors_sample_rate() -> None:
    from src.inference.capture import should_capture

    assert should_capture(sample_rate=1.0, random_value=0.99) is True
    assert should_capture(sample_rate=0.5, random_value=0.49) is True
    assert should_capture(sample_rate=0.5, random_value=0.5) is False
    assert should_capture(sample_rate=0.0, random_value=0.0) is False


def test_workspace_blob_capture_settings_load_from_environment(monkeypatch) -> None:
    """
    @proves online-deploy.externalize-repo-owned-inference-capture-azure-accessible-jsonl
    """
    from src.inference.capture import RepoOwnedInferenceCaptureSettings

    monkeypatch.setenv("INFERENCE_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("INFERENCE_CAPTURE_MODE", "workspaceblobstore_jsonl")
    monkeypatch.setenv("INFERENCE_CAPTURE_OUTPUT_PATH", "monitoring/inference_capture")
    monkeypatch.setenv("INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true")
    monkeypatch.setenv("INFERENCE_CAPTURE_STORAGE_CONTAINER", "monitoring")
    monkeypatch.setenv("INFERENCE_CAPTURE_SESSION_ID", "session-123")

    settings = RepoOwnedInferenceCaptureSettings.from_environment()

    assert settings.enabled is True
    assert settings.mode == "workspaceblobstore_jsonl"
    assert settings.output_path == "monitoring/inference_capture"
    assert settings.storage_connection_string == "UseDevelopmentStorage=true"
    assert settings.storage_container == "monitoring"
    assert settings.session_id == "session-123"


def test_azure_blob_capture_sink_writes_jsonl_records_with_session_prefix() -> None:
    """
    @proves online-deploy.externalize-repo-owned-inference-capture-azure-accessible-jsonl
    """
    from src.inference.capture import AzureBlobInferenceCaptureSink
    from setup.download_capture_blob import parse_capture_blob_path

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

    sink = AzureBlobInferenceCaptureSink(
        service_client=FakeServiceClient(),
        container_name="monitoring",
        path_prefix="monitoring/inference_capture",
        session_id="session-123",
    )

    location = sink.write_record(
        endpoint_name="churn-endpoint",
        deployment_name="blue",
        record={
            "timestamp_utc": "2026-04-14T00:00:00+00:00",
            "request_id": "request-123",
            "outputs": [0],
        },
    )

    assert captured["container"] == "monitoring"
    assert captured["blob"] == (
        "monitoring/inference_capture/2026/04/14/churn-endpoint/blue/session-123/request-123.jsonl"
    )
    assert captured["overwrite"] is False
    assert json.loads(str(captured["data"]).strip()) == {
        "timestamp_utc": "2026-04-14T00:00:00+00:00",
        "request_id": "request-123",
        "outputs": [0],
    }
    assert location == (
        "azureblob://workspaceblobstore/monitoring/"
        "monitoring/inference_capture/2026/04/14/churn-endpoint/blue/session-123/request-123.jsonl"
    )
    assert parse_capture_blob_path(location) == (
        "monitoring/inference_capture/2026/04/14/churn-endpoint/blue/session-123/request-123.jsonl"
    )
