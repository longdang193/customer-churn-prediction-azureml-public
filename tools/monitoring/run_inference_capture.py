"""
@meta
name: run_inference_capture
type: script
domain: inference-monitoring
responsibility:
  - Invoke an Azure ML endpoint from a caller-side wrapper.
  - Write bounded JSONL inference capture evidence outside the endpoint runtime.
inputs:
  - Endpoint coordinates
  - Endpoint payload file
  - Caller capture config
outputs:
  - Capture manifest JSON
  - Optional JSONL inference capture record
tags:
  - inference
  - monitoring
  - azure-ml
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from src.azureml import get_ml_client
from src.inference.client_capture import (
    CallerCaptureRequest,
    invoke_with_capture,
    load_caller_capture_settings,
)


def _write_manifest(
    *,
    manifest_path: Path,
    endpoint_name: str,
    deployment_name: str,
    request_file: Path,
    result_record: dict[str, object],
) -> None:
    payload = {
        **result_record,
        "endpoint_name": endpoint_name,
        "deployment_name": deployment_name,
        "request_file": str(request_file),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Invoke a managed online endpoint and write caller-side inference capture evidence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--endpoint-name", required=True, help="Azure ML endpoint name")
    parser.add_argument("--deployment-name", required=True, help="Azure ML deployment name")
    parser.add_argument("--request-file", required=True, help="Endpoint request JSON payload")
    parser.add_argument("--config", default="configs/inference_capture.yaml", help="Caller capture config")
    parser.add_argument("--azure-config", default="config.env", help="Azure workspace config.env path")
    parser.add_argument("--output-manifest", required=True, help="Capture manifest JSON path")
    parser.add_argument("--model-name", default="unknown-model", help="Model name to record in capture evidence")
    parser.add_argument("--model-version", default="unknown-version", help="Model version to record in capture evidence")
    args = parser.parse_args()

    ml_client = get_ml_client(args.azure_config)
    settings = load_caller_capture_settings(Path(args.config))

    def endpoint_invoker(
        *,
        endpoint_name: str,
        deployment_name: str,
        request_file: Path,
    ) -> object:
        return ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            request_file=str(request_file),
        )

    request = CallerCaptureRequest(
        endpoint_name=args.endpoint_name,
        deployment_name=args.deployment_name,
        request_file=Path(args.request_file),
        model_name=args.model_name,
        model_version=args.model_version,
    )
    result = invoke_with_capture(
        request=request,
        settings=settings,
        endpoint_invoker=endpoint_invoker,
    )
    _write_manifest(
        manifest_path=Path(args.output_manifest),
        endpoint_name=args.endpoint_name,
        deployment_name=args.deployment_name,
        request_file=Path(args.request_file),
        result_record=result.to_manifest_record(),
    )
    print(result.response_preview)
    if result.capture_path:
        print(f"CAPTURE_PATH={result.capture_path}")


if __name__ == "__main__":
    main()
