"""
@meta
name: inference
type: utility
domain: inference
responsibility:
  - Provide reusable inference payload utilities for deployment and monitoring handoff.
inputs:
  - Endpoint request payload files
outputs:
  - Validated payload summaries and canary inference records
tags:
  - inference
  - deployment
  - monitoring
lifecycle:
  status: active
"""

from __future__ import annotations

from .client_capture import (
    CallerCaptureRequest,
    CallerCaptureResult,
    CallerInferenceCaptureSettings,
    invoke_with_capture,
    load_caller_capture_settings,
)
from .payloads import (
    DEFAULT_ENDPOINT_FEATURE_COUNT,
    EndpointPayloadSummary,
    build_canary_inference_record,
    preview_response,
    validate_endpoint_payload,
    validate_endpoint_payload_file,
)


__all__ = [
    "DEFAULT_ENDPOINT_FEATURE_COUNT",
    "CallerCaptureRequest",
    "CallerCaptureResult",
    "CallerInferenceCaptureSettings",
    "EndpointPayloadSummary",
    "build_canary_inference_record",
    "invoke_with_capture",
    "load_caller_capture_settings",
    "preview_response",
    "validate_endpoint_payload",
    "validate_endpoint_payload_file",
]
