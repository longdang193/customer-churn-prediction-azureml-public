"""
@meta
name: inference_client_capture
type: utility
domain: inference-monitoring
responsibility:
  - Invoke managed online endpoints from a caller-side wrapper.
  - Write bounded request/response inference capture records outside the endpoint runtime.
inputs:
  - Endpoint payload file
  - Caller capture settings
  - Endpoint invocation callable
outputs:
  - Endpoint response
  - JSONL capture records
tags:
  - inference
  - monitoring
  - deployment
features:
  - release-monitoring-evaluator
capabilities:
  - monitor.provide-caller-side-inference-capture-wrapper-invokes-managed
lifecycle:
  status: active
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Protocol
import uuid

import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from .capture import (
    AzureBlobInferenceCaptureSink,
    FileInferenceCaptureSink,
    JSONL_CAPTURE_MODE,
    RepoOwnedInferenceCaptureSettings,
    SupportsBlobServiceClient,
    SupportsInferenceCaptureSink,
    build_capture_record,
    create_blob_service_client,
    should_capture,
)
from .payloads import (
    DEFAULT_ENDPOINT_FEATURE_COUNT,
    preview_response,
    validate_endpoint_payload,
    validate_endpoint_payload_file,
)


CALLER_LOCAL_JSONL_MODE = "caller_local_jsonl"
CALLER_BLOB_JSONL_MODE = "caller_blob_jsonl"
DEFAULT_CALLER_CAPTURE_OUTPUT_PATH = ".tmp-tests/inference_capture"
DEFAULT_CALLER_CAPTURE_SOURCE = "caller_side_wrapper"
DEFAULT_SAMPLE_RATE = 1.0
DEFAULT_MAX_ROWS_PER_REQUEST = 5


class EndpointInvoker(Protocol):
    def __call__(
        self,
        *,
        endpoint_name: str,
        deployment_name: str,
        request_file: Path,
    ) -> object:
        ...


@dataclass(frozen=True)
class CallerInferenceCaptureSettings:
    enabled: bool
    mode: str
    sample_rate: float = 1.0
    max_rows_per_request: int = 5
    redact_inputs: bool = True
    output_path: str = DEFAULT_CALLER_CAPTURE_OUTPUT_PATH
    storage_connection_string_env: str | None = "INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING"
    storage_container_env: str | None = "INFERENCE_CAPTURE_STORAGE_CONTAINER"


@dataclass(frozen=True)
class CallerCaptureRequest:
    endpoint_name: str
    deployment_name: str
    request_file: Path
    model_name: str = "unknown-model"
    model_version: str = "unknown-version"
    expected_feature_count: int = DEFAULT_ENDPOINT_FEATURE_COUNT


@dataclass(frozen=True)
class CallerCaptureResult:
    response: object
    response_preview: str
    capture_status: str
    capture_path: str | None
    warnings: list[str] = field(default_factory=list)

    def to_manifest_record(self) -> dict[str, object]:
        return {
            "status": "succeeded",
            "response_preview": self.response_preview,
            "capture_status": self.capture_status,
            "capture_path": self.capture_path,
            "warnings": self.warnings,
        }


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _bool_value(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _float_value(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"Expected numeric config value, got {type(value).__name__}.")
    return float(value)


def _int_value(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"Expected integer config value, got {type(value).__name__}.")
    return int(value)


def _load_payload_rows(payload_path: Path, *, expected_feature_count: int) -> list[list[float]]:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Endpoint payload must be a JSON object with an 'input_data' field.")
    return validate_endpoint_payload(
        payload,
        expected_feature_count=expected_feature_count,
    )


def _parse_response_outputs(response: object) -> list[Any]:
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return [response]
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    if isinstance(response, list):
        return response
    return [response]


def load_caller_capture_settings(config_path: Path) -> CallerInferenceCaptureSettings:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = loaded if isinstance(loaded, Mapping) else {}
    capture = _as_mapping(config.get("inference_capture"))
    return CallerInferenceCaptureSettings(
        enabled=_bool_value(capture.get("enabled"), default=False),
        mode=str(capture.get("mode") or CALLER_LOCAL_JSONL_MODE),
        sample_rate=_float_value(capture.get("sample_rate"), default=DEFAULT_SAMPLE_RATE),
        max_rows_per_request=_int_value(
            capture.get("max_rows_per_request"),
            default=DEFAULT_MAX_ROWS_PER_REQUEST,
        ),
        redact_inputs=_bool_value(capture.get("redact_inputs"), default=True),
        output_path=str(
            capture.get("output_path")
            or DEFAULT_CALLER_CAPTURE_OUTPUT_PATH
        ),
        storage_connection_string_env=str(
            capture.get("storage_connection_string_env")
            or "INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING"
        ),
        storage_container_env=str(
            capture.get("storage_container_env")
            or "INFERENCE_CAPTURE_STORAGE_CONTAINER"
        ),
    )


def _blob_sink_from_environment(
    settings: CallerInferenceCaptureSettings,
    *,
    blob_service_client_factory: Callable[[str], SupportsBlobServiceClient],
) -> tuple[SupportsInferenceCaptureSink | None, list[str]]:
    warnings: list[str] = []
    connection_env_name = settings.storage_connection_string_env
    container_env_name = settings.storage_container_env
    connection_string = os.getenv(connection_env_name or "")
    container_name = os.getenv(container_env_name or "")
    if not connection_string:
        warnings.append(
            f"caller capture connection string env var '{connection_env_name}' is not set"
        )
    if not container_name:
        warnings.append(
            f"caller capture container env var '{container_env_name}' is not set"
        )
    if warnings:
        return None, warnings
    assert connection_string is not None
    assert container_name is not None
    return (
        AzureBlobInferenceCaptureSink(
            service_client=blob_service_client_factory(connection_string),
            container_name=container_name,
            path_prefix=settings.output_path,
        ),
        warnings,
    )


def _build_sink(
    settings: CallerInferenceCaptureSettings,
    *,
    blob_service_client_factory: Callable[[str], SupportsBlobServiceClient],
) -> tuple[SupportsInferenceCaptureSink | None, list[str]]:
    if not settings.enabled:
        return None, []
    if settings.mode == CALLER_LOCAL_JSONL_MODE:
        return FileInferenceCaptureSink(Path(settings.output_path)), []
    if settings.mode == CALLER_BLOB_JSONL_MODE:
        return _blob_sink_from_environment(
            settings,
            blob_service_client_factory=blob_service_client_factory,
        )
    return None, [f"unsupported caller capture mode: {settings.mode}"]


def invoke_with_capture(
    *,
    request: CallerCaptureRequest,
    settings: CallerInferenceCaptureSettings,
    endpoint_invoker: EndpointInvoker,
    random_value_factory: Callable[[], float] | None = None,
    blob_service_client_factory: Callable[[str], SupportsBlobServiceClient] = create_blob_service_client,
) -> CallerCaptureResult:
    """
    Invoke the managed endpoint and emit bounded caller-side capture evidence.

    @capability monitor.provide-caller-side-inference-capture-wrapper-invokes-managed
    """
    validate_endpoint_payload_file(
        request.request_file,
        expected_feature_count=request.expected_feature_count,
    )
    payload_rows = _load_payload_rows(
        request.request_file,
        expected_feature_count=request.expected_feature_count,
    )
    response = endpoint_invoker(
        endpoint_name=request.endpoint_name,
        deployment_name=request.deployment_name,
        request_file=request.request_file,
    )
    if not settings.enabled:
        return CallerCaptureResult(
            response=response,
            response_preview=preview_response(response),
            capture_status="disabled",
            capture_path=None,
        )

    random_probe = random_value_factory() if random_value_factory else None
    if not should_capture(sample_rate=settings.sample_rate, random_value=random_probe):
        return CallerCaptureResult(
            response=response,
            response_preview=preview_response(response),
            capture_status="skipped",
            capture_path=None,
        )

    sink, warnings = _build_sink(
        settings,
        blob_service_client_factory=blob_service_client_factory,
    )
    if sink is None:
        return CallerCaptureResult(
            response=response,
            response_preview=preview_response(response),
            capture_status="degraded",
            capture_path=None,
            warnings=warnings,
        )

    input_df = pd.DataFrame(payload_rows)
    capture_settings = _to_repo_owned_capture_settings(settings)
    record = build_capture_record(
        settings=capture_settings,
        endpoint_name=request.endpoint_name,
        deployment_name=request.deployment_name,
        model_name=request.model_name,
        model_version=request.model_version,
        request_id=uuid.uuid4().hex,
        input_df=input_df,
        predictions=_parse_response_outputs(response),
    )
    record["capture_source"] = DEFAULT_CALLER_CAPTURE_SOURCE
    try:
        capture_path = sink.write_record(
            endpoint_name=request.endpoint_name,
            deployment_name=request.deployment_name,
            record=record,
        )
    except Exception as error:  # pragma: no cover - defensive for cloud writes
        return CallerCaptureResult(
            response=response,
            response_preview=preview_response(response),
            capture_status="degraded",
            capture_path=None,
            warnings=[f"caller capture write failed: {type(error).__name__}: {error}"],
        )
    return CallerCaptureResult(
        response=response,
        response_preview=preview_response(response),
        capture_status="healthy",
        capture_path=capture_path,
    )


def _to_repo_owned_capture_settings(
    settings: CallerInferenceCaptureSettings,
) -> RepoOwnedInferenceCaptureSettings:
    return RepoOwnedInferenceCaptureSettings(
        enabled=settings.enabled,
        mode=JSONL_CAPTURE_MODE,
        sample_rate=settings.sample_rate,
        max_rows_per_request=settings.max_rows_per_request,
        capture_inputs=True,
        capture_outputs=True,
        redact_inputs=settings.redact_inputs,
        output_path=settings.output_path,
    )
