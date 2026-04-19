"""
@meta
name: run_monitor
type: script
domain: monitoring
responsibility:
  - Evaluate monitor readiness from release artifacts.
  - Emit bounded monitor outputs and a monitor step manifest.
inputs:
  - release_record.json
  - configs/monitor.yaml
outputs:
  - monitor_summary.json
  - monitor_report.md
  - monitor_manifest/step_manifest.json
tags:
  - monitoring
  - release
  - cli
features:
  - release-monitoring-evaluator
capabilities:
  - monitor.emit-bounded-monitoring-artifacts-such-monitor-summary-json
  - monitor.support-capture-path-override-scheduled-operator-driven-runs
  - monitor.emit-retraining-decision-json-plus-embedded-retraining-policy
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import cast

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.monitoring import evaluate_release_monitoring
from src.utils.step_manifest import (
    build_step_manifest,
    finalize_manifest,
    merge_config,
    merge_section,
)


def _build_report(summary: dict[str, object]) -> str:
    checks = cast(dict[str, object], summary["checks"])
    retraining_policy = cast(dict[str, object], summary.get("retraining_policy", {}))
    lines = [
        "# Monitoring Report",
        "",
        f"- Monitor status: `{summary['monitor_status']}`",
        f"- Evidence level: `{summary['evidence_level']}`",
        f"- Monitoring handoff: `{summary['monitoring_handoff_status']}`",
        f"- Runtime contract: `{summary['runtime_contract']}`",
        f"- Capture status: `{summary['capture_status']}`",
        f"- Recommended action: {summary['recommended_action']}",
        "",
        "## Checks",
    ]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    prediction_distribution = cast(dict[str, object], summary.get("prediction_distribution", {}))
    if prediction_distribution:
        lines.extend(
            [
                "",
                "## Prediction Distribution",
                *[
                    f"- `{label}`: `{count}`"
                    for label, count in prediction_distribution.items()
                ],
            ]
        )
    if retraining_policy:
        lines.extend(
            [
                "",
                "## Retraining Policy",
                f"- Trigger: `{retraining_policy.get('trigger')}`",
                f"- Recommended training path: `{retraining_policy.get('recommended_training_path')}`",
                f"- Path recommendation: `{retraining_policy.get('path_recommendation')}`",
                f"- Policy confidence: `{retraining_policy.get('policy_confidence')}`",
                f"- Signal persistence: `{retraining_policy.get('signal_persistence')}`",
                f"- Requires dataset freeze: `{retraining_policy.get('requires_dataset_freeze')}`",
                f"- Requires data validation: `{retraining_policy.get('requires_data_validation')}`",
                f"- Requires human review: `{retraining_policy.get('requires_human_review')}`",
                f"- Next step: {retraining_policy.get('next_step')}",
            ]
        )
        reason_codes = cast(list[object], retraining_policy.get("reason_codes", []))
        if reason_codes:
            lines.extend(
                [
                    "- Reason codes:",
                    *[f"  - `{reason_code}`" for reason_code in reason_codes],
                ]
            )
        path_reason_codes = cast(
            list[object], retraining_policy.get("path_recommendation_reason_codes", [])
        )
        if path_reason_codes:
            lines.extend(
                [
                    "- Path recommendation reason codes:",
                    *[f"  - `{reason_code}`" for reason_code in path_reason_codes],
                ]
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    """
    @capability monitor.emit-bounded-monitoring-artifacts-such-monitor-summary-json
    @capability monitor.support-capture-path-override-scheduled-operator-driven-runs
    @capability monitor.emit-retraining-decision-json-plus-embedded-retraining-policy
    """
    parser = argparse.ArgumentParser(description="Evaluate monitoring readiness from release artifacts.")
    parser.add_argument("--release-record", required=True, help="Path to release_record.json")
    parser.add_argument("--config", default="configs/monitor.yaml", help="Monitor config path")
    parser.add_argument(
        "--capture-path",
        help="Optional override for externally retrieved capture evidence (file or directory).",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for monitor artifacts")
    parser.add_argument(
        "--summary-output",
        help="Optional explicit path for monitor_summary.json.",
    )
    parser.add_argument(
        "--report-output",
        help="Optional explicit path for monitor_report.md.",
    )
    parser.add_argument(
        "--manifest-output",
        help="Optional explicit path for monitor_manifest/step_manifest.json.",
    )
    parser.add_argument(
        "--retraining-output",
        help="Optional explicit path for retraining_decision.json.",
    )
    args = parser.parse_args()

    release_record_path = Path(args.release_record)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_output) if args.summary_output else output_dir / "monitor_summary.json"
    report_path = Path(args.report_output) if args.report_output else output_dir / "monitor_report.md"
    retraining_path = (
        Path(args.retraining_output)
        if args.retraining_output
        else output_dir / "retraining_decision.json"
    )
    manifest_path = (
        Path(args.manifest_output)
        if args.manifest_output
        else output_dir / "monitor_manifest" / "step_manifest.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    retraining_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    summary = evaluate_release_monitoring(
        release_record_path=release_record_path,
        config_path=config_path,
        output_dir=output_dir,
        capture_path_override=Path(args.capture_path) if args.capture_path else None,
    )

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path.write_text(_build_report(summary), encoding="utf-8")
    retraining_path.write_text(
        json.dumps(cast(dict[str, object], summary.get("retraining_policy", {})), indent=2),
        encoding="utf-8",
    )

    manifest = build_step_manifest(step_name="run_monitor", stage_name="monitor")
    merge_config(
        manifest,
        config_paths={
            "release_record": release_record_path,
            "monitor_config": config_path,
            "capture_path_override": Path(args.capture_path) if args.capture_path else None,
        },
    )
    merge_section(
        manifest,
        "inputs",
        {"release_record": {"path": release_record_path}},
    )
    if args.capture_path:
        merge_section(
            manifest,
            "inputs",
            {"capture_path_override": {"path": args.capture_path}},
        )
    merge_section(
        manifest,
        "outputs",
        {
            "monitor_summary": {"path": summary_path},
            "monitor_report": {"path": report_path},
            "retraining_decision": {"path": retraining_path},
            "monitor_manifest": {"path": manifest_path},
        },
    )
    merge_section(
        manifest,
        "metrics",
        {
            "capture_record_count": summary.get("capture_record_count", 0),
        },
    )
    merge_section(
        manifest,
        "step_specific",
        {
            "monitoring": {
                "monitor_status": summary.get("monitor_status"),
                "evidence_level": summary.get("evidence_level"),
                "runtime_contract": summary.get("runtime_contract"),
                "retraining_trigger": cast(dict[str, object], summary.get("retraining_policy", {})).get("trigger"),
                "recommended_training_path": cast(dict[str, object], summary.get("retraining_policy", {})).get("recommended_training_path"),
                "path_recommendation": cast(dict[str, object], summary.get("retraining_policy", {})).get("path_recommendation"),
                "policy_confidence": cast(dict[str, object], summary.get("retraining_policy", {})).get("policy_confidence"),
                "signal_persistence": cast(dict[str, object], summary.get("retraining_policy", {})).get("signal_persistence"),
            }
        },
    )
    finalize_manifest(
        manifest,
        output_path=manifest_path,
        status="succeeded",
    )


if __name__ == "__main__":
    main()
