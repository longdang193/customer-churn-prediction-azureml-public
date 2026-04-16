"""
@meta
name: inference_capture
type: utility
domain: inference-serving
responsibility:
  - Shape bounded repo-owned online inference capture records.
  - Write sampled inference records to local or Azure-accessible JSONL sinks.
inputs:
  - Scoring inputs and prediction outputs
  - Capture settings
outputs:
  - Bounded inference capture records
  - JSONL files or blobs for later monitoring jobs
tags:
  - inference
  - monitoring
  - deployment
lifecycle:
  status: active
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path, PurePosixPath
import random
from typing import Any, Callable, Protocol
import uuid

import pandas as pd


DEFAULT_CAPTURE_MODE = "release_evidence_only"
JSONL_CAPTURE_MODE = "jsonl_file"
WORKSPACE_BLOBSTORE_JSONL_CAPTURE_MODE = "workspaceblobstore_jsonl"
DEFAULT_LOCAL_CAPTURE_OUTPUT_PATH = "/tmp/repo-owned-inference-capture"
DEFAULT_CAPTURE_CONTAINER = "monitoring"
CAPTURE_WARNING_PREFIX = "INFERENCE_CAPTURE_WARNING="
CAPTURE_PATH_PREFIX = "INFERENCE_CAPTURE_PATH="


class SupportsInferenceCaptureSink(Protocol):
    def write_record(
        self,
        *,
        endpoint_name: str,
        deployment_name: str,
        record: dict[str, object],
    ) -> str:
        ...


class SupportsBlobServiceClient(Protocol):
    account_name: str

    def get_blob_client(self, *, container: str, blob: str) -> Any:
        ...

    def get_container_client(self, container: str) -> Any:
        ...


@dataclass(frozen=True)
class RepoOwnedInferenceCaptureSettings:
    enabled: bool
    mode: str
    sample_rate: float
    max_rows_per_request: int
    capture_inputs: bool
    capture_outputs: bool
    redact_inputs: bool
    output_path: str
    storage_connection_string: str | None = None
    storage_container: str | None = None
    session_id: str | None = None

    @classmethod
    def from_environment(cls) -> "RepoOwnedInferenceCaptureSettings":
        return cls(
            enabled=_env_bool("INFERENCE_CAPTURE_ENABLED", default=False),
            mode=_env_str("INFERENCE_CAPTURE_MODE", default=DEFAULT_CAPTURE_MODE),
            sample_rate=_env_float("INFERENCE_CAPTURE_SAMPLE_RATE", default=1.0),
            max_rows_per_request=_env_int("INFERENCE_CAPTURE_MAX_ROWS", default=5),
            capture_inputs=_env_bool("INFERENCE_CAPTURE_INPUTS", default=True),
            capture_outputs=_env_bool("INFERENCE_CAPTURE_OUTPUTS", default=True),
            redact_inputs=_env_bool("INFERENCE_CAPTURE_REDACT_INPUTS", default=False),
            output_path=_env_str(
                "INFERENCE_CAPTURE_OUTPUT_PATH",
                default=DEFAULT_LOCAL_CAPTURE_OUTPUT_PATH,
            ),
            storage_connection_string=_env_optional_str(
                "INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING"
            ),
            storage_container=_env_optional_str("INFERENCE_CAPTURE_STORAGE_CONTAINER"),
            session_id=_env_optional_str("INFERENCE_CAPTURE_SESSION_ID"),
        )


def _env_str(name: str, *, default: str) -> str:
    import os

    return os.getenv(name, default).strip() or default


def _env_bool(name: str, *, default: bool) -> bool:
    import os

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, *, default: float) -> float:
    import os

    value = os.getenv(name)
    return float(value) if value else default


def _env_int(name: str, *, default: int) -> int:
    import os

    value = os.getenv(name)
    return int(value) if value else default


def _env_optional_str(name: str) -> str | None:
    import os

    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def should_capture(*, sample_rate: float, random_value: float | None = None) -> bool:
    if sample_rate <= 0.0:
        return False
    if sample_rate >= 1.0:
        return True
    probe = random.random() if random_value is None else random_value
    return probe < sample_rate


def _bounded_rows(frame: pd.DataFrame, *, max_rows: int, redact_inputs: bool) -> list[object]:
    bounded = frame.head(max_rows)
    if redact_inputs:
        return [{"feature_count": len(bounded.columns)} for _ in range(len(bounded))]
    return bounded.to_dict(orient="records")


def build_capture_record(
    *,
    settings: RepoOwnedInferenceCaptureSettings,
    endpoint_name: str,
    deployment_name: str,
    model_name: str,
    model_version: str,
    request_id: str,
    input_df: pd.DataFrame,
    predictions: list[Any],
) -> dict[str, object]:
    captured_row_count = min(len(input_df.index), settings.max_rows_per_request)
    record: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "endpoint_name": endpoint_name,
        "deployment_name": deployment_name,
        "model_name": model_name,
        "model_version": model_version,
        "row_count": len(input_df.index),
        "captured_row_count": captured_row_count,
        "feature_count": len(input_df.columns),
    }
    if settings.capture_inputs:
        record["inputs"] = _bounded_rows(
            input_df,
            max_rows=settings.max_rows_per_request,
            redact_inputs=settings.redact_inputs,
        )
    if settings.capture_outputs:
        record["outputs"] = list(predictions[: settings.max_rows_per_request])
    return record


def build_capture_relative_path(
    *,
    path_prefix: str,
    endpoint_name: str,
    deployment_name: str,
    timestamp_utc: str,
    request_id: str,
    session_id: str | None,
) -> PurePosixPath:
    partition = timestamp_utc.split("T", 1)[0].replace("-", "/")
    path = PurePosixPath(path_prefix) if path_prefix else PurePosixPath()
    path = path / partition / endpoint_name / deployment_name
    if session_id:
        path = path / session_id
    return path / f"{request_id}.jsonl"


@dataclass
class FileInferenceCaptureSink:
    base_path: Path

    def write_record(
        self,
        *,
        endpoint_name: str,
        deployment_name: str,
        record: dict[str, object],
    ) -> str:
        relative_path = build_capture_relative_path(
            path_prefix="",
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            timestamp_utc=str(record.get("timestamp_utc", "")),
            request_id=str(record.get("request_id", "")),
            session_id=None,
        )
        target_path = self.base_path / Path(str(relative_path))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(record, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return str(target_path)


@dataclass
class AzureBlobInferenceCaptureSink:
    service_client: SupportsBlobServiceClient
    container_name: str
    path_prefix: str
    session_id: str | None = None

    def write_record(
        self,
        *,
        endpoint_name: str,
        deployment_name: str,
        record: dict[str, object],
    ) -> str:
        blob_name = build_capture_relative_path(
            path_prefix=self.path_prefix,
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            timestamp_utc=str(record.get("timestamp_utc", "")),
            request_id=str(record.get("request_id", "")),
            session_id=self.session_id,
        ).as_posix()
        blob_client = self.service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name,
        )
        blob_client.upload_blob(
            json.dumps(record, separators=(",", ":")) + "\n",
            overwrite=False,
        )
        return (
            f"azureblob://{self.service_client.account_name}/"
            f"{self.container_name}/{blob_name}"
        )


@dataclass
class RepoOwnedInferenceCaptureRuntime:
    settings: RepoOwnedInferenceCaptureSettings
    sink: SupportsInferenceCaptureSink | None
    endpoint_name: str
    deployment_name: str
    model_name: str
    model_version: str
    random_value_factory: Callable[[], float] = random.random
    status: str = "disabled"
    warnings: list[str] = field(default_factory=list)
    last_written_path: str | None = None

    def __post_init__(self) -> None:
        if not self.settings.enabled:
            self.status = "disabled"
            return
        if self.settings.mode not in {
            JSONL_CAPTURE_MODE,
            WORKSPACE_BLOBSTORE_JSONL_CAPTURE_MODE,
        }:
            self.status = "disabled"
            self._record_warning(
                f"unsupported inference capture mode: {self.settings.mode}"
            )
            return
        if self.sink is None:
            self.status = "degraded"
            self._record_warning("inference capture sink is unavailable")
            return
        self.status = "healthy"

    def maybe_capture(self, *, input_df: pd.DataFrame, predictions: list[Any]) -> None:
        if (
            not self.settings.enabled
            or self.settings.mode
            not in {JSONL_CAPTURE_MODE, WORKSPACE_BLOBSTORE_JSONL_CAPTURE_MODE}
            or self.sink is None
        ):
            return
        if not should_capture(
            sample_rate=self.settings.sample_rate,
            random_value=self.random_value_factory(),
        ):
            return
        request_id = uuid.uuid4().hex
        record = build_capture_record(
            settings=self.settings,
            endpoint_name=self.endpoint_name,
            deployment_name=self.deployment_name,
            model_name=self.model_name,
            model_version=self.model_version,
            request_id=request_id,
            input_df=input_df,
            predictions=predictions,
        )
        try:
            written_path = self.sink.write_record(
                endpoint_name=self.endpoint_name,
                deployment_name=self.deployment_name,
                record=record,
            )
        except Exception as error:  # pragma: no cover - defensive for serving runtime
            self.status = "degraded"
            self._record_warning(f"inference capture write failed: {error}")
            return
        self.last_written_path = written_path
        logging.info("%s%s", CAPTURE_PATH_PREFIX, written_path)

    def _record_warning(self, warning: str) -> None:
        if warning not in self.warnings:
            self.warnings.append(warning)
            logging.warning("%s%s", CAPTURE_WARNING_PREFIX, warning)


def create_blob_service_client(connection_string: str) -> SupportsBlobServiceClient:
    from azure.storage.blob import BlobServiceClient  # type: ignore[import-not-found]

    return BlobServiceClient.from_connection_string(connection_string)


def create_capture_runtime(
    *,
    settings: RepoOwnedInferenceCaptureSettings,
    endpoint_name: str,
    deployment_name: str,
    model_name: str,
    model_version: str,
    random_value_factory: Callable[[], float] = random.random,
    blob_service_client_factory: Callable[[str], SupportsBlobServiceClient] = create_blob_service_client,
) -> RepoOwnedInferenceCaptureRuntime:
    sink: SupportsInferenceCaptureSink | None = None
    if settings.enabled and settings.mode == JSONL_CAPTURE_MODE:
        sink = FileInferenceCaptureSink(Path(settings.output_path))
    elif settings.enabled and settings.mode == WORKSPACE_BLOBSTORE_JSONL_CAPTURE_MODE:
        if settings.storage_connection_string and settings.storage_container:
            sink = AzureBlobInferenceCaptureSink(
                service_client=blob_service_client_factory(
                    settings.storage_connection_string
                ),
                container_name=settings.storage_container,
                path_prefix=settings.output_path,
                session_id=settings.session_id,
            )
    return RepoOwnedInferenceCaptureRuntime(
        settings=settings,
        sink=sink,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        model_name=model_name,
        model_version=model_version,
        random_value_factory=random_value_factory,
    )
