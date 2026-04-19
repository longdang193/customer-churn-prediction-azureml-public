"""
@meta
name: run_retraining_candidate
type: script
domain: retraining-handoff
responsibility:
  - Freeze explicit dataset identities after a monitor-stage retraining decision.
  - Emit a validate_data-ready handoff and optionally run the existing validator.
inputs:
  - release_record.json
  - monitor_summary.json or retraining_decision.json
  - Current and reference dataset identifiers
  - configs/data.yaml
outputs:
  - retraining_candidate_manifest.json
  - candidate_summary.json
  - validation_handoff.json
  - Optional validation artifacts
tags:
  - monitoring
  - retraining
  - validation
  - cli
features:
  - online-endpoint-deployment
  - release-monitoring-evaluator
capabilities:
  - online-deploy.provide-enough-release-monitor-provenance-later-retraining-candidate
  - monitor.support-thin-post-monitor-bridge-freezes-explicit-current
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Literal, Mapping, Optional, TypedDict, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from validate_data import load_validation_config, run_validation


TRIGGER_RETRAINING_CANDIDATE = "retraining_candidate"
TRIGGER_NO_RETRAINING_SIGNAL = "no_retraining_signal"
TRIGGER_INVESTIGATE = "investigate_before_retraining"
STATUS_CANDIDATE_OPENED = "candidate_opened"
STATUS_NO_CANDIDATE_OPENED = "no_candidate_opened"
STATUS_INVESTIGATION_REQUIRED = "investigation_required"
IDENTIFIER_KIND_LOCAL = "local_path"
IDENTIFIER_KIND_URI = "uri"
IDENTIFIER_KIND_AZUREML_ASSET = "azureml_asset"


class RetrainingPolicyPayload(TypedDict, total=False):
    trigger: str
    recommended_training_path: Optional[str]
    policy_version: int
    reason_codes: list[str]
    next_step: str


@dataclass(frozen=True)
class DatasetIdentifier:
    raw: str
    normalized: str
    kind: Literal["local_path", "uri", "azureml_asset"]


@dataclass(frozen=True)
class CandidateContext:
    release_record_path: Path
    decision_source_path: Path
    decision_source_kind: Literal["monitor_summary", "retraining_decision"]
    policy: RetrainingPolicyPayload
    current_data: DatasetIdentifier
    reference_data: DatasetIdentifier
    data_config_path: Path


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_identifier(raw_value: str) -> DatasetIdentifier:
    stripped = raw_value.strip()
    if stripped.startswith("azureml:") and not stripped.startswith("azureml://"):
        return DatasetIdentifier(
            raw=stripped,
            normalized=stripped,
            kind=IDENTIFIER_KIND_AZUREML_ASSET,
        )
    if "://" in stripped:
        return DatasetIdentifier(
            raw=stripped,
            normalized=stripped,
            kind=IDENTIFIER_KIND_URI,
        )
    resolved = Path(stripped).expanduser().resolve()
    return DatasetIdentifier(
        raw=stripped,
        normalized=str(resolved),
        kind=IDENTIFIER_KIND_LOCAL,
    )


def _load_policy(
    *,
    monitor_summary_path: Path | None,
    retraining_decision_path: Path | None,
) -> tuple[Path, Literal["monitor_summary", "retraining_decision"], RetrainingPolicyPayload]:
    if retraining_decision_path is not None:
        payload = _load_json(retraining_decision_path)
        return (
            retraining_decision_path,
            "retraining_decision",
            cast(RetrainingPolicyPayload, payload),
        )

    assert monitor_summary_path is not None
    monitor_summary = _load_json(monitor_summary_path)
    retraining_policy = monitor_summary.get("retraining_policy", {})
    if not isinstance(retraining_policy, Mapping):
        raise ValueError("monitor_summary.json did not contain a valid retraining_policy block")
    return (
        monitor_summary_path,
        "monitor_summary",
        cast(RetrainingPolicyPayload, dict(retraining_policy)),
    )


def _build_candidate_manifest(context: CandidateContext) -> dict[str, object]:
    return {
        "frozen_at_utc": _utc_timestamp(),
        "release_record_path": str(context.release_record_path),
        "decision_source": {
            "path": str(context.decision_source_path),
            "kind": context.decision_source_kind,
        },
        "policy_version": context.policy.get("policy_version", 1),
        "trigger": context.policy.get("trigger"),
        "reason_codes": context.policy.get("reason_codes", []),
        "training_path_recommendation": context.policy.get("recommended_training_path"),
        "current_data": asdict(context.current_data),
        "reference_data": asdict(context.reference_data),
        "data_config_path": str(context.data_config_path),
    }


def _build_validation_handoff(
    context: CandidateContext,
    *,
    output_dir: Path,
    run_validation_now: bool,
) -> dict[str, object]:
    validation_dir = output_dir / "validation"
    summary_path = validation_dir / "validation_summary.json"
    manifest_path = validation_dir / "validation_manifest" / "step_manifest.json"
    command = [
        "python",
        "src/validate_data.py",
        "--reference-data",
        context.reference_data.raw,
        "--current-data",
        context.current_data.raw,
        "--output-dir",
        str(validation_dir),
        "--summary-path",
        str(summary_path),
        "--manifest-output",
        str(manifest_path),
        "--config",
        str(context.data_config_path),
    ]
    return {
        "created_at_utc": _utc_timestamp(),
        "run_validation": run_validation_now,
        "current_data": asdict(context.current_data),
        "reference_data": asdict(context.reference_data),
        "data_config_path": str(context.data_config_path),
        "validation_output_dir": str(validation_dir),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "command": command,
    }


def _run_validation_now(context: CandidateContext, output_dir: Path) -> dict[str, object]:
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    summary_path = validation_dir / "validation_summary.json"
    manifest_path = validation_dir / "validation_manifest" / "step_manifest.json"
    config = load_validation_config(str(context.data_config_path))
    summary = run_validation(
        reference_path=Path(context.reference_data.normalized),
        current_path=Path(context.current_data.normalized),
        output_dir=validation_dir,
        summary_path=summary_path,
        config=config,
        config_path=context.data_config_path,
        manifest_output_path=manifest_path,
    )
    return {
        "status": summary["status"],
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
    }


def _build_candidate_summary(
    *,
    status: str,
    context: CandidateContext,
    validation_result: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "trigger": context.policy.get("trigger"),
        "reason_codes": context.policy.get("reason_codes", []),
        "recommended_training_path": context.policy.get("recommended_training_path"),
        "release_record_path": str(context.release_record_path),
        "decision_source_path": str(context.decision_source_path),
        "current_data": asdict(context.current_data),
        "reference_data": asdict(context.reference_data),
        "data_config_path": str(context.data_config_path),
        "validation": validation_result,
    }


def main() -> None:
    """
    @capability monitor.support-thin-post-monitor-bridge-freezes-explicit-current
    """
    parser = argparse.ArgumentParser(
        description="Freeze a retraining candidate and optionally run validate_data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--release-record", required=True, help="Path to saved release_record.json")
    decision_group = parser.add_mutually_exclusive_group(required=True)
    decision_group.add_argument("--monitor-summary", help="Path to monitor_summary.json")
    decision_group.add_argument("--retraining-decision", help="Path to retraining_decision.json")
    parser.add_argument("--current-data", required=True, help="Current dataset identifier or local path")
    parser.add_argument("--reference-data", required=True, help="Reference dataset identifier or local path")
    parser.add_argument("--data-config", required=True, help="Data config path for validate_data")
    parser.add_argument("--output-dir", required=True, help="Output directory for candidate artifacts")
    parser.add_argument(
        "--run-validation",
        action="store_true",
        help="Run validate_data immediately after opening the candidate",
    )
    args = parser.parse_args()

    release_record_path = Path(args.release_record).resolve()
    if not release_record_path.exists():
        raise SystemExit(f"release_record.json was not found at {release_record_path}")

    monitor_summary_path = Path(args.monitor_summary).resolve() if args.monitor_summary else None
    retraining_decision_path = (
        Path(args.retraining_decision).resolve() if args.retraining_decision else None
    )
    decision_source_path, decision_source_kind, policy = _load_policy(
        monitor_summary_path=monitor_summary_path,
        retraining_decision_path=retraining_decision_path,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = CandidateContext(
        release_record_path=release_record_path,
        decision_source_path=decision_source_path,
        decision_source_kind=decision_source_kind,
        policy=policy,
        current_data=_normalize_identifier(args.current_data),
        reference_data=_normalize_identifier(args.reference_data),
        data_config_path=Path(args.data_config).resolve(),
    )

    trigger = str(policy.get("trigger", "unknown"))
    summary_path = output_dir / "candidate_summary.json"

    if trigger == TRIGGER_NO_RETRAINING_SIGNAL:
        _write_json(
            summary_path,
            _build_candidate_summary(
                status=STATUS_NO_CANDIDATE_OPENED,
                context=context,
            ),
        )
        print(f"Retraining candidate summary: {summary_path}")
        return

    if trigger != TRIGGER_RETRAINING_CANDIDATE:
        _write_json(
            summary_path,
            _build_candidate_summary(
                status=STATUS_INVESTIGATION_REQUIRED,
                context=context,
            ),
        )
        print(f"Retraining candidate summary: {summary_path}")
        return

    manifest_path = output_dir / "retraining_candidate_manifest.json"
    handoff_path = output_dir / "validation_handoff.json"
    _write_json(manifest_path, _build_candidate_manifest(context))
    _write_json(
        handoff_path,
        _build_validation_handoff(
            context,
            output_dir=output_dir,
            run_validation_now=bool(args.run_validation),
        ),
    )

    validation_result: dict[str, object] | None = None
    if args.run_validation:
        validation_result = _run_validation_now(context, output_dir)

    _write_json(
        summary_path,
        _build_candidate_summary(
            status=STATUS_CANDIDATE_OPENED,
            context=context,
            validation_result=validation_result,
        ),
    )
    print(f"Retraining candidate summary: {summary_path}")


if __name__ == "__main__":
    main()
