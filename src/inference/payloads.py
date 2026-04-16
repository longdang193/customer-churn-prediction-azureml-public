"""
@meta
name: endpoint_payloads
type: utility
domain: inference
responsibility:
  - Validate Azure ML endpoint smoke-test payloads before invoke.
  - Build sanitized canary inference evidence for release records.
inputs:
  - JSON endpoint request payload files
  - Endpoint, deployment, and registered model coordinates
outputs:
  - Endpoint payload summaries
  - Canary inference record fragments
tags:
  - inference
  - deployment
  - monitoring
lifecycle:
  status: active
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping


DEFAULT_ENDPOINT_FEATURE_COUNT = 10
INPUT_DATA_2D_ARRAY_FORMAT = "input_data_2d_array"
MAX_RESPONSE_PREVIEW_LENGTH = 500


@dataclass(frozen=True)
class EndpointPayloadSummary:
    """Validated endpoint request payload shape summary."""

    payload_path: Path
    row_count: int
    feature_count: int
    payload_format: str = INPUT_DATA_2D_ARRAY_FORMAT
    validation_status: str = "passed"

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable payload summary."""
        return {
            "path": str(self.payload_path),
            "format": self.payload_format,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "validation_status": self.validation_status,
        }


def _load_payload(payload_path: Path) -> Mapping[str, object]:
    if not payload_path.exists():
        raise FileNotFoundError(f"Endpoint payload file not found: {payload_path}")
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Endpoint payload must be valid JSON: {payload_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Endpoint payload must be a JSON object with an 'input_data' field.")
    return payload


def _validate_feature_value(value: object, *, row_index: int, column_index: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "Endpoint payload input_data "
            f"row {row_index} column {column_index} must be numeric."
        )


def _validate_row(row: object, *, row_index: int, expected_feature_count: int) -> None:
    if not isinstance(row, list):
        raise ValueError(f"Endpoint payload input_data row {row_index} must be a list.")
    if len(row) != expected_feature_count:
        raise ValueError(
            "Endpoint payload input_data "
            f"row {row_index} expected {expected_feature_count} features, got {len(row)}."
        )
    for column_index, value in enumerate(row):
        _validate_feature_value(value, row_index=row_index, column_index=column_index)


def validate_endpoint_payload(
    payload: Mapping[str, object],
    *,
    expected_feature_count: int = DEFAULT_ENDPOINT_FEATURE_COUNT,
) -> list[list[float]]:
    """Validate an already-loaded endpoint request payload."""
    if "input_data" not in payload:
        raise ValueError("Endpoint payload must include an 'input_data' field.")

    input_data = payload["input_data"]
    if not isinstance(input_data, list) or not input_data:
        raise ValueError("Endpoint payload 'input_data' must be a non-empty list of rows.")

    validated_rows: list[list[float]] = []
    for row_index, row in enumerate(input_data):
        _validate_row(
            row,
            row_index=row_index,
            expected_feature_count=expected_feature_count,
        )
        validated_rows.append([float(value) for value in row])
    return validated_rows


def validate_endpoint_payload_file(
    payload_path: Path,
    *,
    expected_feature_count: int = DEFAULT_ENDPOINT_FEATURE_COUNT,
) -> EndpointPayloadSummary:
    """Validate the processed-feature Azure ML endpoint request payload."""
    payload = _load_payload(payload_path)
    validated_rows = validate_endpoint_payload(
        payload,
        expected_feature_count=expected_feature_count,
    )

    return EndpointPayloadSummary(
        payload_path=payload_path,
        row_count=len(validated_rows),
        feature_count=expected_feature_count,
    )


def preview_response(response: object) -> str:
    preview = str(response)
    if len(preview) <= MAX_RESPONSE_PREVIEW_LENGTH:
        return preview
    return f"{preview[:MAX_RESPONSE_PREVIEW_LENGTH]}..."


_preview_response = preview_response


def build_canary_inference_record(
    *,
    payload_summary: Mapping[str, object],
    endpoint_name: str | None,
    deployment_name: str | None,
    model_name: str,
    model_version: str,
    response: object,
) -> dict[str, object]:
    """Build sanitized canary inference evidence for a release record."""
    return {
        "payload": dict(payload_summary),
        "endpoint": {
            "endpoint_name": endpoint_name,
            "deployment_name": deployment_name,
        },
        "model": {
            "name": model_name,
            "version": model_version,
        },
        "response": {
            "preview": preview_response(response),
        },
    }
