"""
@meta
name: run_retraining_path_selection
type: script
domain: retraining-handoff
responsibility:
  - Select the next retraining path after candidate validation passes.
  - Invoke the proven fixed-train smoke bridge or prepare a truthful HPO handoff.
inputs:
  - release_record.json
  - retraining_decision.json or monitor_summary.json
  - retraining_candidate_manifest.json
  - validation_summary.json
outputs:
  - retraining_path_selection.json
  - retraining_path_selection_summary.json
  - Optional selected_path_invocation.json
  - Optional retraining_hpo_handoff.json
tags:
  - monitoring
  - retraining
  - training
  - cli
features:
  - model-training-pipeline
  - online-endpoint-deployment
  - release-monitoring-evaluator
capabilities:
  - fixed-train.accept-path-selection-artifact
  - fixed-train.path-selection-invokes-bridges
  - online-deploy.provide-enough-release-monitor-provenance-later-post-validation
  - monitor.support-third-thin-bridge-selects-next-retraining-path
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
from typing import Literal, Mapping, TypedDict, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from tools.retraining.run_retraining_fixed_train_smoke import (  # noqa: E402
    DEFAULT_DATA_CONFIG_PATH as DEFAULT_FIXED_TRAIN_DATA_CONFIG_PATH,
)
from tools.retraining.run_retraining_fixed_train_smoke import (  # noqa: E402
    DEFAULT_TRAIN_CONFIG_PATH as DEFAULT_FIXED_TRAIN_CONFIG_PATH,
)


STATUS_BLOCKED = "blocked"
STATUS_DRY_RUN_READY = "dry_run_ready"
STATUS_SELECTED_AND_INVOKED = "selected_and_invoked"
STATUS_PREPARED_HPO_HANDOFF = "prepared_hpo_handoff"
STATUS_INVOCATION_FAILED = "invocation_failed"

PATH_FIXED_TRAIN = "fixed_train"
PATH_MODEL_SWEEP = "model_sweep"
SUPPORTED_TRAINING_PATHS = {PATH_FIXED_TRAIN, PATH_MODEL_SWEEP}
DEFAULT_HPO_CONFIG_PATH = "configs/hpo_smoke.yaml"


class DecisionPayload(TypedDict, total=False):
    trigger: str
    recommended_training_path: str | None
    policy_version: int
    reason_codes: list[str]
    next_step: str


@dataclass(frozen=True)
class SelectionContext:
    release_record_path: Path
    decision_source_path: Path
    decision_source_kind: Literal["monitor_summary", "retraining_decision"]
    decision_payload: DecisionPayload
    candidate_manifest_path: Path
    candidate_manifest: dict[str, object]
    validation_summary_path: Path
    validation_summary: dict[str, object]


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


def _load_decision(
    *,
    monitor_summary_path: Path | None,
    retraining_decision_path: Path | None,
) -> tuple[Path, Literal["monitor_summary", "retraining_decision"], DecisionPayload]:
    if retraining_decision_path is not None:
        payload = _load_json(retraining_decision_path)
        return (
            retraining_decision_path,
            "retraining_decision",
            cast(DecisionPayload, payload),
        )

    assert monitor_summary_path is not None
    monitor_summary = _load_json(monitor_summary_path)
    retraining_policy = monitor_summary.get("retraining_policy", {})
    if not isinstance(retraining_policy, Mapping):
        raise ValueError("monitor_summary.json did not contain a valid retraining_policy block")
    return (
        monitor_summary_path,
        "monitor_summary",
        cast(DecisionPayload, dict(retraining_policy)),
    )


def _validation_status(validation_summary: Mapping[str, object]) -> str:
    status = validation_summary.get("status")
    return status if isinstance(status, str) else "unknown"


def _resolve_recommended_path(context: SelectionContext) -> str | None:
    decision_path = context.decision_payload.get("recommended_training_path")
    if isinstance(decision_path, str) and decision_path in SUPPORTED_TRAINING_PATHS:
        return decision_path

    candidate_path = context.candidate_manifest.get("training_path_recommendation")
    if isinstance(candidate_path, str) and candidate_path in SUPPORTED_TRAINING_PATHS:
        return candidate_path

    return None


def _build_selection_payload(
    *,
    context: SelectionContext,
    selected_path: str | None,
    invoke_selected_path: bool,
    train_config_path: str,
    data_config_path: str,
    hpo_config_path: str | None,
    downstream_entrypoint: str | None,
    downstream_summary_path: Path | None,
) -> dict[str, object]:
    return {
        "created_at_utc": _utc_timestamp(),
        "release_record_path": str(context.release_record_path),
        "decision_source": {
            "path": str(context.decision_source_path),
            "kind": context.decision_source_kind,
        },
        "candidate_manifest_path": str(context.candidate_manifest_path),
        "validation_summary_path": str(context.validation_summary_path),
        "trigger": context.decision_payload.get("trigger"),
        "policy_version": context.decision_payload.get("policy_version", 1),
        "reason_codes": context.decision_payload.get("reason_codes", []),
        "selected_path": selected_path,
        "validation_status": _validation_status(context.validation_summary),
        "invoke_selected_path": invoke_selected_path,
        "current_data": context.candidate_manifest.get("current_data"),
        "reference_data": context.candidate_manifest.get("reference_data"),
        "downstream": {
            "entrypoint": downstream_entrypoint,
            "data_config_path": data_config_path,
            "train_config_path": train_config_path if selected_path == PATH_FIXED_TRAIN else None,
            "hpo_config_path": hpo_config_path if selected_path == PATH_MODEL_SWEEP else None,
            "summary_path": str(downstream_summary_path) if downstream_summary_path else None,
        },
    }


def _build_summary_payload(
    *,
    status: str,
    context: SelectionContext,
    selected_path: str | None,
    downstream_invoked: bool,
    downstream_summary_path: Path | None,
) -> dict[str, object]:
    return {
        "status": status,
        "trigger": context.decision_payload.get("trigger"),
        "reason_codes": context.decision_payload.get("reason_codes", []),
        "selected_path": selected_path,
        "validation_status": _validation_status(context.validation_summary),
        "release_record_path": str(context.release_record_path),
        "decision_source_path": str(context.decision_source_path),
        "candidate_manifest_path": str(context.candidate_manifest_path),
        "validation_summary_path": str(context.validation_summary_path),
        "downstream_invoked": downstream_invoked,
        "downstream_summary_path": str(downstream_summary_path) if downstream_summary_path else None,
    }


def _build_fixed_train_command(
    *,
    context: SelectionContext,
    output_dir: Path,
    data_config_path: str,
    train_config_path: str,
) -> list[str]:
    return [
        sys.executable,
        "run_retraining_fixed_train_smoke.py",
        "--release-record",
        str(context.release_record_path),
        "--candidate-manifest",
        str(context.candidate_manifest_path),
        "--validation-summary",
        str(context.validation_summary_path),
        "--data-config",
        data_config_path,
        "--train-config",
        train_config_path,
        "--output-dir",
        str(output_dir),
    ]


def _invoke_fixed_train_handoff(argv: list[str]) -> Path:
    completed = subprocess.run(
        argv,
        check=True,
        text=True,
        capture_output=True,
    )
    summary_path = Path(argv[-1]) / "retraining_fixed_train_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Fixed-train handoff did not produce summary at {summary_path}. stdout={completed.stdout!r}"
        )
    return summary_path


def _build_hpo_handoff_payload(
    *,
    context: SelectionContext,
    data_config_path: str,
    hpo_config_path: str,
) -> dict[str, object]:
    return {
        "created_at_utc": _utc_timestamp(),
        "release_record_path": str(context.release_record_path),
        "decision_source": {
            "path": str(context.decision_source_path),
            "kind": context.decision_source_kind,
        },
        "candidate_manifest_path": str(context.candidate_manifest_path),
        "validation_summary_path": str(context.validation_summary_path),
        "trigger": context.decision_payload.get("trigger"),
        "reason_codes": context.decision_payload.get("reason_codes", []),
        "selected_path": PATH_MODEL_SWEEP,
        "entrypoint": "run_hpo_pipeline.py",
        "current_data": context.candidate_manifest.get("current_data"),
        "reference_data": context.candidate_manifest.get("reference_data"),
        "data_config_path": data_config_path,
        "hpo_config_path": hpo_config_path,
    }


def main() -> None:
    """
    @capability monitor.support-third-thin-bridge-selects-next-retraining-path
    """
    parser = argparse.ArgumentParser(
        description="Select the next retraining path after candidate validation passes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--release-record", required=True, help="Path to saved release_record.json")
    decision_group = parser.add_mutually_exclusive_group(required=True)
    decision_group.add_argument("--monitor-summary", help="Path to monitor_summary.json")
    decision_group.add_argument("--retraining-decision", help="Path to retraining_decision.json")
    parser.add_argument(
        "--candidate-manifest",
        required=True,
        help="Path to retraining_candidate_manifest.json",
    )
    parser.add_argument(
        "--validation-summary",
        required=True,
        help="Path to validation_summary.json",
    )
    parser.add_argument(
        "--data-config",
        default=DEFAULT_FIXED_TRAIN_DATA_CONFIG_PATH,
        help="Data config used for downstream handoffs.",
    )
    parser.add_argument(
        "--train-config",
        default=DEFAULT_FIXED_TRAIN_CONFIG_PATH,
        help="Train config used when fixed_train is selected.",
    )
    parser.add_argument(
        "--hpo-config",
        default=DEFAULT_HPO_CONFIG_PATH,
        help="HPO config used when model_sweep is selected.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for selection artifacts.")
    parser.add_argument(
        "--invoke-selected-path",
        action="store_true",
        help="Invoke the selected downstream path when the current implementation supports it.",
    )
    args = parser.parse_args()

    release_record_path = Path(args.release_record).resolve()
    monitor_summary_path = Path(args.monitor_summary).resolve() if args.monitor_summary else None
    retraining_decision_path = (
        Path(args.retraining_decision).resolve() if args.retraining_decision else None
    )
    candidate_manifest_path = Path(args.candidate_manifest).resolve()
    validation_summary_path = Path(args.validation_summary).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decision_source_path, decision_source_kind, decision_payload = _load_decision(
        monitor_summary_path=monitor_summary_path,
        retraining_decision_path=retraining_decision_path,
    )
    candidate_manifest = _load_json(candidate_manifest_path)
    validation_summary = _load_json(validation_summary_path)

    context = SelectionContext(
        release_record_path=release_record_path,
        decision_source_path=decision_source_path,
        decision_source_kind=decision_source_kind,
        decision_payload=decision_payload,
        candidate_manifest_path=candidate_manifest_path,
        candidate_manifest=candidate_manifest,
        validation_summary_path=validation_summary_path,
        validation_summary=validation_summary,
    )

    selection_path = output_dir / "retraining_path_selection.json"
    summary_path = output_dir / "retraining_path_selection_summary.json"
    invocation_path = output_dir / "selected_path_invocation.json"
    hpo_handoff_path = output_dir / "retraining_hpo_handoff.json"

    validation_status = _validation_status(validation_summary)
    selected_path = _resolve_recommended_path(context)

    if validation_status != "passed" or selected_path is None:
        _write_json(
            selection_path,
            _build_selection_payload(
                context=context,
                selected_path=selected_path,
                invoke_selected_path=bool(args.invoke_selected_path),
                train_config_path=args.train_config,
                data_config_path=args.data_config,
                hpo_config_path=args.hpo_config,
                downstream_entrypoint=None,
                downstream_summary_path=None,
            ),
        )
        _write_json(
            summary_path,
            _build_summary_payload(
                status=STATUS_BLOCKED,
                context=context,
                selected_path=selected_path,
                downstream_invoked=False,
                downstream_summary_path=None,
            ),
        )
        reason = (
            f"Validation must be passed before path selection, got {validation_status!r}"
            if validation_status != "passed"
            else "Retraining path recommendation was missing or unsupported"
        )
        raise SystemExit(reason)

    if selected_path == PATH_MODEL_SWEEP:
        _write_json(
            hpo_handoff_path,
            _build_hpo_handoff_payload(
                context=context,
                data_config_path=args.data_config,
                hpo_config_path=args.hpo_config,
            ),
        )
        _write_json(
            selection_path,
            _build_selection_payload(
                context=context,
                selected_path=selected_path,
                invoke_selected_path=False,
                train_config_path=args.train_config,
                data_config_path=args.data_config,
                hpo_config_path=args.hpo_config,
                downstream_entrypoint="run_hpo_pipeline.py",
                downstream_summary_path=hpo_handoff_path,
            ),
        )
        _write_json(
            summary_path,
            _build_summary_payload(
                status=STATUS_PREPARED_HPO_HANDOFF,
                context=context,
                selected_path=selected_path,
                downstream_invoked=False,
                downstream_summary_path=hpo_handoff_path,
            ),
        )
        print(f"Retraining path selection summary: {summary_path}")
        return

    downstream_summary_path: Path | None = None
    if args.invoke_selected_path:
        selected_output_dir = output_dir / "selected-path"
        command = _build_fixed_train_command(
            context=context,
            output_dir=selected_output_dir,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
        )
        _write_json(
            invocation_path,
            {
                "created_at_utc": _utc_timestamp(),
                "selected_path": selected_path,
                "command": command,
            },
        )
        try:
            downstream_summary_path = _invoke_fixed_train_handoff(command)
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            _write_json(
                selection_path,
                _build_selection_payload(
                    context=context,
                    selected_path=selected_path,
                    invoke_selected_path=True,
                    train_config_path=args.train_config,
                    data_config_path=args.data_config,
                    hpo_config_path=args.hpo_config,
                    downstream_entrypoint="run_retraining_fixed_train_smoke.py",
                    downstream_summary_path=None,
                ),
            )
            _write_json(
                summary_path,
                _build_summary_payload(
                    status=STATUS_INVOCATION_FAILED,
                    context=context,
                    selected_path=selected_path,
                    downstream_invoked=True,
                    downstream_summary_path=None,
                ),
            )
            raise SystemExit("Selected fixed-train handoff failed") from error

        _write_json(
            selection_path,
            _build_selection_payload(
                context=context,
                selected_path=selected_path,
                invoke_selected_path=True,
                train_config_path=args.train_config,
                data_config_path=args.data_config,
                hpo_config_path=args.hpo_config,
                downstream_entrypoint="run_retraining_fixed_train_smoke.py",
                downstream_summary_path=downstream_summary_path,
            ),
        )
        _write_json(
            summary_path,
            _build_summary_payload(
                status=STATUS_SELECTED_AND_INVOKED,
                context=context,
                selected_path=selected_path,
                downstream_invoked=True,
                downstream_summary_path=downstream_summary_path,
            ),
        )
        print(f"Retraining path selection summary: {summary_path}")
        return

    _write_json(
        selection_path,
        _build_selection_payload(
            context=context,
            selected_path=selected_path,
            invoke_selected_path=False,
            train_config_path=args.train_config,
            data_config_path=args.data_config,
            hpo_config_path=args.hpo_config,
            downstream_entrypoint="run_retraining_fixed_train_smoke.py",
            downstream_summary_path=None,
        ),
    )
    _write_json(
        summary_path,
        _build_summary_payload(
            status=STATUS_DRY_RUN_READY,
            context=context,
            selected_path=selected_path,
            downstream_invoked=False,
            downstream_summary_path=None,
        ),
    )
    print(f"Retraining path selection summary: {summary_path}")


if __name__ == "__main__":
    main()
