"""
@meta
name: run_release_failure_probe
type: script
domain: release-validation
responsibility:
  - Send one bounded negative request to a deployed Azure ML endpoint.
  - Prove repo-owned scorer failures remain clean and artifact-truthful in cloud.
inputs:
  - Existing release_record.json
  - Intentional bad request payload
  - Azure workspace config.env
outputs:
  - failure_probe_summary.json
tags:
  - azure-ml
  - release
  - validation
  - failure-path
features:
  - online-endpoint-deployment
  - release-monitoring-evaluator
capabilities:
  - online-deploy.support-one-bounded-negative-cloud-probe-intentionally-bypasses
  - monitor.negative-scorer-probes-may-produce-intentional-azure-scoring
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from src.config.runtime import load_azure_config


PROBE_TYPE = "repo_owned_scorer_negative_payload"
INTENTIONAL_FAILURE_STATUS = "intentional_failure_observed"
UNEXPECTED_SUCCESS_STATUS = "unexpected_success"
VALIDATION_BYPASS_REASON = "negative_probe_only"
DEFAULT_BAD_PAYLOAD = "sample-data-invalid-feature-count.json"
DEFAULT_LOG_LINES = 80


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_az_executable() -> str:
    candidates = (
        shutil.which("az"),
        shutil.which("az.cmd"),
        r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd",
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError(
        "Could not locate the Azure CLI executable. Install Azure CLI or add az.cmd to PATH."
    )


def _coerce_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _extract_failure(error_text: str) -> dict[str, object]:
    stripped = error_text.strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    best_line = lines[-1] if lines else stripped
    error_type = "RuntimeError"
    error_message = best_line
    if ": " in best_line:
        candidate_type, candidate_message = best_line.split(": ", maxsplit=1)
        if candidate_type and candidate_message:
            error_type = candidate_type.strip()
            error_message = candidate_message.strip()
    return {
        "source": "repo_owned_scorer",
        "error_type": error_type,
        "error_message": error_message,
        "raw_error": stripped,
    }


def _fetch_deployment_logs(
    *,
    azure_config_path: str,
    endpoint_name: str,
    deployment_name: str,
    lines: int = DEFAULT_LOG_LINES,
) -> str:
    azure_config = load_azure_config(azure_config_path)
    command = [
        _resolve_az_executable(),
        "ml",
        "online-deployment",
        "get-logs",
        "--name",
        deployment_name,
        "--endpoint-name",
        endpoint_name,
        "--resource-group",
        azure_config["resource_group"],
        "--workspace-name",
        azure_config["workspace_name"],
        "--lines",
        str(lines),
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    return result.stdout if result.returncode == 0 else result.stderr


def _best_log_excerpt(log_text: str) -> list[str]:
    interesting_markers = (
        "REPO_OWNED_SCORER_RUN=",
        "ValueError:",
        "TypeError:",
        "RuntimeError:",
        "UserScriptException:",
    )
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    return [line for line in lines if any(marker in line for marker in interesting_markers)]


def _enrich_failure_from_logs(
    *,
    failure: Mapping[str, object],
    log_text: str,
) -> dict[str, object]:
    enriched = dict(failure)
    excerpt = _best_log_excerpt(log_text)
    if excerpt:
        enriched["log_excerpt"] = excerpt
        scorer_line = next(
            (line for line in excerpt if line.startswith("ValueError:")),
            None,
        )
        if scorer_line is not None:
            enriched["error_type"] = "ValueError"
            enriched["error_message"] = scorer_line.split(": ", maxsplit=1)[1]
    return enriched


def _build_summary(
    *,
    status: str,
    release_record_path: Path,
    request_file: Path,
    release_record: Mapping[str, object],
    failure: Mapping[str, object] | None = None,
    response_preview: str | None = None,
) -> dict[str, object]:
    deployment = _coerce_mapping(release_record.get("deployment"))
    registered_model = _coerce_mapping(release_record.get("registered_model"))
    monitoring_handoff = _coerce_mapping(release_record.get("monitoring_handoff"))
    payload: dict[str, object] = {
        "status": status,
        "probe_type": PROBE_TYPE,
        "release_record_path": str(release_record_path),
        "endpoint": {
            "endpoint_name": deployment.get("endpoint_name"),
            "deployment_name": deployment.get("deployment_name"),
        },
        "model": {
            "name": registered_model.get("name"),
            "version": registered_model.get("version"),
        },
        "request": {
            "path": str(request_file),
            "validation_bypass": VALIDATION_BYPASS_REASON,
        },
        "artifact_truth": {
            "release_status": release_record.get("status"),
            "monitor_handoff_status": monitoring_handoff.get("status"),
            "monitor_evidence_level": monitoring_handoff.get("evidence_level"),
            "repo_owned_scoring_status": deployment.get("repo_owned_scoring_status"),
        },
    }
    if failure is not None:
        payload["failure"] = dict(failure)
    if response_preview is not None:
        payload["response_preview"] = response_preview
    return payload


def _invoke_negative_probe(
    *,
    azure_config_path: str,
    endpoint_name: str,
    deployment_name: str,
    request_file: Path,
) -> subprocess.CompletedProcess[str]:
    azure_config = load_azure_config(azure_config_path)
    command = [
        _resolve_az_executable(),
        "ml",
        "online-endpoint",
        "invoke",
        "--name",
        endpoint_name,
        "--deployment-name",
        deployment_name,
        "--request-file",
        str(request_file),
        "--resource-group",
        azure_config["resource_group"],
        "--workspace-name",
        azure_config["workspace_name"],
    ]
    return subprocess.run(command, check=False, text=True, capture_output=True)


def main() -> None:
    """
    @capability monitor.negative-scorer-probes-may-produce-intentional-azure-scoring
    """
    parser = argparse.ArgumentParser(
        description="Run one bounded negative Azure probe against a proven deployed scorer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--release-record", required=True, help="Existing release_record.json path")
    parser.add_argument("--config", default="config.env", help="Azure workspace config.env path")
    parser.add_argument(
        "--request-file",
        default=DEFAULT_BAD_PAYLOAD,
        help="Intentional bad request payload for scorer failure proof",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for the failure summary")
    args = parser.parse_args()

    load_dotenv(args.config)
    release_record_path = Path(args.release_record)
    request_file = Path(args.request_file).resolve()
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "failure_probe_summary.json"

    release_record = _load_json(release_record_path)
    deployment = _coerce_mapping(release_record.get("deployment"))
    endpoint_name = str(deployment.get("endpoint_name") or "")
    deployment_name = str(deployment.get("deployment_name") or "")
    if not endpoint_name or not deployment_name:
        raise SystemExit("Release record is missing endpoint or deployment coordinates.")
    if not request_file.exists():
        raise SystemExit(f"Request file not found: {request_file}")

    probe_result = _invoke_negative_probe(
        azure_config_path=args.config,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        request_file=request_file,
    )

    if probe_result.returncode == 0:
        summary = _build_summary(
            status=UNEXPECTED_SUCCESS_STATUS,
            release_record_path=release_record_path,
            request_file=request_file,
            release_record=release_record,
            response_preview=probe_result.stdout.strip() or None,
        )
        _write_json(summary_path, summary)
        raise SystemExit(
            f"Negative probe did not fail as expected. See {summary_path}"
        )

    failure = _extract_failure(probe_result.stderr or probe_result.stdout)
    log_text = _fetch_deployment_logs(
        azure_config_path=args.config,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
    )
    summary = _build_summary(
        status=INTENTIONAL_FAILURE_STATUS,
        release_record_path=release_record_path,
        request_file=request_file,
        release_record=release_record,
        failure=_enrich_failure_from_logs(failure=failure, log_text=log_text),
    )
    _write_json(summary_path, summary)
    print(f"Failure probe summary: {summary_path}")


if __name__ == "__main__":
    main()
