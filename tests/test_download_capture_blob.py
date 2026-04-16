"""
@meta
type: test
scope: unit
domain: inference-monitoring
covers:
  - Exact capture blob download helper
excludes:
  - Real Azure Blob network calls
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid

import pytest


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"download-capture-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_parse_capture_blob_path_supports_azureml_uri() -> None:
    from src.monitoring.download_capture_blob import parse_capture_blob_path

    assert parse_capture_blob_path(
        "azureml://datastores/workspaceblobstore/paths/monitoring/session/record.jsonl"
    ) == "monitoring/session/record.jsonl"


def test_parse_capture_blob_path_supports_azureblob_uri() -> None:
    from src.monitoring.download_capture_blob import parse_capture_blob_path

    assert parse_capture_blob_path(
        "azureblob://workspaceblobstore/monitoring/session/record.jsonl"
    ) == "session/record.jsonl"


def test_download_capture_blob_copies_local_file() -> None:
    from src.monitoring.download_capture_blob import download_capture_blob

    temp_dir = _make_temp_dir()
    try:
        source_file = temp_dir / "source.jsonl"
        destination_file = temp_dir / "nested" / "copied.jsonl"
        source_file.write_text('{"prediction":"1"}\n', encoding="utf-8")

        written = download_capture_blob(
            capture_uri=str(source_file),
            output_file=destination_file,
            connection_string_env="IGNORED_CONNECTION",
            container_env="IGNORED_CONTAINER",
        )

        assert written == destination_file
        assert destination_file.read_text(encoding="utf-8") == '{"prediction":"1"}\n'
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_download_capture_blob_retries_blob_not_found_then_succeeds(monkeypatch) -> None:
    from azure.core.exceptions import ResourceNotFoundError
    from src.monitoring.download_capture_blob import download_capture_blob

    temp_dir = _make_temp_dir()
    attempts = {"count": 0}
    sleep_calls: list[float] = []

    class FakeDownloadStream:
        def readall(self) -> bytes:
            return b'{"prediction":"0"}\n'

    class FakeBlobClient:
        def download_blob(self) -> FakeDownloadStream:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ResourceNotFoundError(message="Blob not ready yet")
            return FakeDownloadStream()

    class FakeBlobServiceClient:
        def get_blob_client(self, *, container: str, blob: str) -> FakeBlobClient:
            assert container == "capture-container"
            assert blob == "monitoring/session/record.jsonl"
            return FakeBlobClient()

    monkeypatch.setenv("TEST_CAPTURE_CONNECTION", "UseDevelopmentStorage=true")
    monkeypatch.setenv("TEST_CAPTURE_CONTAINER", "capture-container")
    monkeypatch.setattr(
        "src.monitoring.download_capture_blob.BlobServiceClient.from_connection_string",
        lambda _conn: FakeBlobServiceClient(),
    )
    monkeypatch.setattr(
        "src.monitoring.download_capture_blob.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    try:
        output_file = temp_dir / "record.jsonl"
        written = download_capture_blob(
            capture_uri="azureml://datastores/workspaceblobstore/paths/monitoring/session/record.jsonl",
            output_file=output_file,
            connection_string_env="TEST_CAPTURE_CONNECTION",
            container_env="TEST_CAPTURE_CONTAINER",
        )

        assert written == output_file
        assert output_file.read_text(encoding="utf-8") == '{"prediction":"0"}\n'
        assert attempts["count"] == 2
        assert sleep_calls == [1.0]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.environ.pop("TEST_CAPTURE_CONNECTION", None)
        os.environ.pop("TEST_CAPTURE_CONTAINER", None)


def test_download_capture_blob_raises_after_retry_budget_exhausted(monkeypatch) -> None:
    from azure.core.exceptions import ResourceNotFoundError
    from src.monitoring.download_capture_blob import download_capture_blob

    temp_dir = _make_temp_dir()
    attempts = {"count": 0}

    class FakeBlobClient:
        def download_blob(self) -> None:
            attempts["count"] += 1
            raise ResourceNotFoundError(message="Blob missing")

    class FakeBlobServiceClient:
        def get_blob_client(self, *, container: str, blob: str) -> FakeBlobClient:
            assert container == "capture-container"
            assert blob == "monitoring/session/record.jsonl"
            return FakeBlobClient()

    monkeypatch.setenv("TEST_CAPTURE_CONNECTION", "UseDevelopmentStorage=true")
    monkeypatch.setenv("TEST_CAPTURE_CONTAINER", "capture-container")
    monkeypatch.setattr(
        "src.monitoring.download_capture_blob.BlobServiceClient.from_connection_string",
        lambda _conn: FakeBlobServiceClient(),
    )
    monkeypatch.setattr("src.monitoring.download_capture_blob.time.sleep", lambda _seconds: None)

    try:
        output_file = temp_dir / "record.jsonl"
        with pytest.raises(ResourceNotFoundError):
            download_capture_blob(
                capture_uri="azureml://datastores/workspaceblobstore/paths/monitoring/session/record.jsonl",
                output_file=output_file,
                connection_string_env="TEST_CAPTURE_CONNECTION",
                container_env="TEST_CAPTURE_CONTAINER",
            )
        assert attempts["count"] == 3
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.environ.pop("TEST_CAPTURE_CONNECTION", None)
        os.environ.pop("TEST_CAPTURE_CONTAINER", None)


def test_setup_download_capture_blob_remains_compatibility_wrapper() -> None:
    from setup.download_capture_blob import parse_capture_blob_path as setup_parse_capture_blob_path
    from src.monitoring.download_capture_blob import parse_capture_blob_path

    assert setup_parse_capture_blob_path is parse_capture_blob_path
