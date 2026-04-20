"""
@meta
name: run_retraining_hpo_smoke
type: script
domain: retraining-handoff
responsibility:
  - Bridge a validated retraining candidate selected for model_sweep into the HPO smoke pipeline.
  - Preserve lineage and the exact run_hpo_pipeline invocation for dry-run and submit modes.
inputs:
  - release_record.json
  - retraining_path_selection.json
  - retraining_candidate_manifest.json
  - validation_summary.json
  - configs/hpo_smoke.yaml
outputs:
  - retraining_hpo_smoke_handoff.json
  - retraining_hpo_smoke_summary.json
  - hpo_invocation.json
tags:
  - monitoring
  - retraining
  - hpo
  - cli
features:
  - model-training-pipeline
  - notebook-hpo
  - release-monitoring-evaluator
capabilities:
  - fixed-train.accept-hpo-smoke-bridge
  - hpo.submit-end-end-azure-ml-hpo-parent-pipeline
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping


DEFAULT_DATA_CONFIG_PATH = "configs/data_smoke_eval.yaml"
DEFAULT_HPO_CONFIG_PATH = "configs/hpo_smoke.yaml"
STATUS_BLOCKED_BY_VALIDATION = "blocked_by_validation"
STATUS_BLOCKED_BY_SELECTION = "blocked_by_selection"
STATUS_DRY_RUN_READY = "dry_run_ready"
STATUS_SUBMITTED = "submitted"
STATUS_SUBMISSION_FAILED = "submission_failed"
EXPECTED_SELECTED_PATH = "model_sweep"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SubmissionResult:
    job_name: str | None
    studio_url: str | None
    stdout: str
    stderr: str


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_nested_str(payload: Mapping[str, object], *keys: str) -> str | None:
    current: object = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, str) else None


def _validation_status(validation_summary: Mapping[str, object]) -> str:
    status = validation_summary.get("status")
    return status if isinstance(status, str) else "unknown"


def _selected_path(selection_payload: Mapping[str, object]) -> str:
    selected = selection_payload.get("selected_path")
    return selected if isinstance(selected, str) else "unknown"


def _build_command(
    *,
    data_config_path: str,
    hpo_config_path: str,
    current_data_override: str,
    reference_data_override: str,
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "run_hpo_pipeline.py"),
        "--data-config",
        data_config_path,
        "--hpo-config",
        hpo_config_path,
        "--current-data-override",
        current_data_override,
        "--reference-data-override",
        reference_data_override,
    ]


def _parse_submission(stdout: str) -> SubmissionResult:
    job_name: str | None = None
    studio_url: str | None = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("OK Submitted hpo-pipeline sweep:"):
            job_name = line.split(":", 1)[1].strip()
        if line.startswith("View in Azure ML Studio:"):
            studio_url = line.split(":", 1)[1].strip()
    return SubmissionResult(job_name=job_name, studio_url=studio_url, stdout=stdout, stderr="")


def _build_invocation_payload(
    *,
    release_record_path: Path,
    selection_path: Path,
    candidate_manifest_path: Path,
    validation_summary_path: Path,
    data_config_path: str,
    hpo_config_path: str,
    current_data_override: str,
    reference_data_override: str,
) -> dict[str, object]:
    command = _build_command(
        data_config_path=data_config_path,
        hpo_config_path=hpo_config_path,
        current_data_override=current_data_override,
        reference_data_override=reference_data_override,
    )
    return {
        "created_at_utc": _utc_timestamp(),
        "release_record_path": str(release_record_path),
        "selection_path": str(selection_path),
        "candidate_manifest_path": str(candidate_manifest_path),
        "validation_summary_path": str(validation_summary_path),
        "arguments": {
            "data_config": data_config_path,
            "hpo_config": hpo_config_path,
            "current_data_override": current_data_override,
            "reference_data_override": reference_data_override,
        },
        "command": command,
    }


def _build_handoff_payload(
    *,
    release_record_path: Path,
    selection_path: Path,
    candidate_manifest_path: Path,
    validation_summary_path: Path,
    selection_payload: Mapping[str, object],
    candidate_manifest: Mapping[str, object],
    data_config_path: str,
    hpo_config_path: str,
) -> dict[str, object]:
    return {
        "created_at_utc": _utc_timestamp(),
        "release_record_path": str(release_record_path),
        "selection_path": str(selection_path),
        "candidate_manifest_path": str(candidate_manifest_path),
        "validation_summary_path": str(validation_summary_path),
        "decision_source": selection_payload.get("decision_source"),
        "trigger": selection_payload.get("trigger"),
        "selected_path": selection_payload.get("selected_path"),
        "reason_codes": selection_payload.get("reason_codes", []),
        "recommendation_summary": selection_payload.get("recommendation_summary"),
        "current_data": candidate_manifest.get("current_data"),
        "reference_data": candidate_manifest.get("reference_data"),
        "data_config_path": data_config_path,
        "hpo_config_path": hpo_config_path,
    }


def _build_summary(
    *,
    status: str,
    validation_status: str,
    selected_path: str,
    release_record_path: Path,
    selection_path: Path,
    candidate_manifest_path: Path,
    validation_summary_path: Path,
    invocation_path: Path | None,
    submit_attempted: bool,
    submission: SubmissionResult | None,
    selection_payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "status": status,
        "validation_status": validation_status,
        "selected_path": selected_path,
        "trigger": selection_payload.get("trigger"),
        "reason_codes": selection_payload.get("reason_codes", []),
        "recommendation_summary": selection_payload.get("recommendation_summary"),
        "release_record_path": str(release_record_path),
        "selection_path": str(selection_path),
        "candidate_manifest_path": str(candidate_manifest_path),
        "validation_summary_path": str(validation_summary_path),
        "invocation_path": str(invocation_path) if invocation_path is not None else None,
        "submission": {
            "attempted": submit_attempted,
            "job_name": submission.job_name if submission is not None else None,
            "studio_url": submission.studio_url if submission is not None else None,
            "stderr": submission.stderr if submission is not None else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hand off a validated retraining candidate into the smoke HPO path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--release-record", required=True, help="Path to saved release_record.json")
    parser.add_argument(
        "--selection",
        required=True,
        help="Path to retraining_path_selection.json",
    )
    parser.add_argument(
        "--candidate-manifest",
        required=True,
        help="Path to retraining_candidate_manifest.json",
    )
    parser.add_argument(
        "--validation-summary",
        required=True,
        help="Path to validation_summary.json from the candidate bridge",
    )
    parser.add_argument(
        "--data-config",
        default=DEFAULT_DATA_CONFIG_PATH,
        help="Data config to pass to run_hpo_pipeline.py",
    )
    parser.add_argument(
        "--hpo-config",
        default=DEFAULT_HPO_CONFIG_PATH,
        help="HPO config to pass to run_hpo_pipeline.py",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for handoff artifacts")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the smoke HPO pipeline instead of stopping at dry-run output.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    release_record_path = Path(args.release_record).resolve()
    selection_path = Path(args.selection).resolve()
    candidate_manifest_path = Path(args.candidate_manifest).resolve()
    validation_summary_path = Path(args.validation_summary).resolve()

    selection_payload = _load_json(selection_path)
    candidate_manifest = _load_json(candidate_manifest_path)
    validation_summary = _load_json(validation_summary_path)
    validation_status = _validation_status(validation_summary)
    selected_path = _selected_path(selection_payload)

    summary_path = output_dir / "retraining_hpo_smoke_summary.json"
    handoff_path = output_dir / "retraining_hpo_smoke_handoff.json"
    invocation_path = output_dir / "hpo_invocation.json"

    if validation_status != "passed":
        _write_json(
            summary_path,
            _build_summary(
                status=STATUS_BLOCKED_BY_VALIDATION,
                validation_status=validation_status,
                selected_path=selected_path,
                release_record_path=release_record_path,
                selection_path=selection_path,
                candidate_manifest_path=candidate_manifest_path,
                validation_summary_path=validation_summary_path,
                invocation_path=None,
                submit_attempted=False,
                submission=None,
                selection_payload=selection_payload,
            ),
        )
        raise SystemExit(
            f"Validation must be passed before HPO handoff, got {validation_status!r}"
        )

    if selected_path != EXPECTED_SELECTED_PATH:
        _write_json(
            summary_path,
            _build_summary(
                status=STATUS_BLOCKED_BY_SELECTION,
                validation_status=validation_status,
                selected_path=selected_path,
                release_record_path=release_record_path,
                selection_path=selection_path,
                candidate_manifest_path=candidate_manifest_path,
                validation_summary_path=validation_summary_path,
                invocation_path=None,
                submit_attempted=False,
                submission=None,
                selection_payload=selection_payload,
            ),
        )
        raise SystemExit(
            f"Selection must be {EXPECTED_SELECTED_PATH!r} before HPO handoff, got {selected_path!r}"
        )

    current_data_override = _get_nested_str(candidate_manifest, "current_data", "raw")
    reference_data_override = _get_nested_str(candidate_manifest, "reference_data", "raw")
    if current_data_override is None or reference_data_override is None:
        raise SystemExit("Candidate manifest is missing current/reference dataset identifiers")

    _write_json(
        handoff_path,
        _build_handoff_payload(
            release_record_path=release_record_path,
            selection_path=selection_path,
            candidate_manifest_path=candidate_manifest_path,
            validation_summary_path=validation_summary_path,
            selection_payload=selection_payload,
            candidate_manifest=candidate_manifest,
            data_config_path=args.data_config,
            hpo_config_path=args.hpo_config,
        ),
    )
    _write_json(
        invocation_path,
        _build_invocation_payload(
            release_record_path=release_record_path,
            selection_path=selection_path,
            candidate_manifest_path=candidate_manifest_path,
            validation_summary_path=validation_summary_path,
            data_config_path=args.data_config,
            hpo_config_path=args.hpo_config,
            current_data_override=current_data_override,
            reference_data_override=reference_data_override,
        ),
    )

    if not args.submit:
        _write_json(
            summary_path,
            _build_summary(
                status=STATUS_DRY_RUN_READY,
                validation_status=validation_status,
                selected_path=selected_path,
                release_record_path=release_record_path,
                selection_path=selection_path,
                candidate_manifest_path=candidate_manifest_path,
                validation_summary_path=validation_summary_path,
                invocation_path=invocation_path,
                submit_attempted=False,
                submission=None,
                selection_payload=selection_payload,
            ),
        )
        print(f"Retraining HPO smoke summary: {summary_path}")
        return

    command = _build_command(
        data_config_path=args.data_config,
        hpo_config_path=args.hpo_config,
        current_data_override=current_data_override,
        reference_data_override=reference_data_override,
    )
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        submission = SubmissionResult(
            job_name=None,
            studio_url=None,
            stdout=error.stdout or "",
            stderr=error.stderr or "",
        )
        _write_json(
            summary_path,
            _build_summary(
                status=STATUS_SUBMISSION_FAILED,
                validation_status=validation_status,
                selected_path=selected_path,
                release_record_path=release_record_path,
                selection_path=selection_path,
                candidate_manifest_path=candidate_manifest_path,
                validation_summary_path=validation_summary_path,
                invocation_path=invocation_path,
                submit_attempted=True,
                submission=submission,
                selection_payload=selection_payload,
            ),
        )
        raise SystemExit("HPO smoke submission failed; see retraining_hpo_smoke_summary.json") from error

    submission = _parse_submission(completed.stdout)
    _write_json(
        summary_path,
        _build_summary(
            status=STATUS_SUBMITTED,
            validation_status=validation_status,
            selected_path=selected_path,
            release_record_path=release_record_path,
            selection_path=selection_path,
            candidate_manifest_path=candidate_manifest_path,
            validation_summary_path=validation_summary_path,
            invocation_path=invocation_path,
            submit_attempted=True,
            submission=submission,
            selection_payload=selection_payload,
        ),
    )
    print(f"Retraining HPO smoke summary: {summary_path}")


if __name__ == "__main__":
    main()
