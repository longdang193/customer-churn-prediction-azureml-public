"""
@meta
name: promote_model
type: utility
domain: promotion
responsibility:
  - Compare candidate and baseline model metrics against promotion thresholds.
  - Emit a machine-readable promotion decision artifact.
inputs:
  - Candidate metrics JSON
  - Baseline metrics JSON
  - Promotion threshold config
outputs:
  - Promotion decision payload
tags:
  - promotion
  - evaluation
  - release
capabilities:
  - fixed-train.execute-promotion-utility
  - fixed-train.expose-promotion-thresholds
lifecycle:
  status: active
"""

from __future__ import annotations

import json
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from config.runtime import PromotionConfig, load_promotion_config
from utils.mlflow_utils import is_azure_ml
from utils.output_paths import resolve_named_input_file
from utils.step_manifest import (
    build_step_manifest,
    finalize_manifest,
    manifest_path_for_dir,
    merge_config,
    merge_section,
    resolve_manifest_output_path,
    set_failure,
)


PROMOTION_DECISION_FILENAME = "promotion_decision.json"
CANDIDATE_METRICS_FILENAME = "candidate_metrics.json"


@dataclass(frozen=True)
class PromotionDecision:
    status: str
    primary_metric: str
    candidate_score: float
    baseline_score: float
    minimum_improvement: float
    minimum_candidate_score: float
    improvement: float
    reasons: list[str]


def _finalize_manifest_safely(
    manifest: dict[str, Any],
    *,
    output_path: Path,
    status: str,
    mirror_output_path: Path | None = None,
) -> None:
    """Best-effort manifest persistence that must not fail promotion itself."""
    try:
        finalized_manifest_path = finalize_manifest(
            manifest,
            output_path=output_path,
            mirror_output_path=mirror_output_path,
            status=status,
        )
        print(f"STEP_MANIFEST_PATH={finalized_manifest_path}")
    except OSError as exc:
        print(
            "STEP_MANIFEST_WARNING="
            f"failed to persist manifest at {output_path}: {type(exc).__name__}: {exc}"
        )


def _resolve_promotion_decision_output_path(output_path: Path) -> Path:
    """Resolve folder-style AML outputs to a UI-friendly JSON file path."""
    output_path = Path(output_path)
    if output_path.suffix.lower() == ".json":
        return output_path
    return output_path / PROMOTION_DECISION_FILENAME


def evaluate_promotion(
    candidate_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    *,
    primary_metric: str = "f1",
    minimum_improvement: float = 0.0,
    minimum_candidate_score: float = 0.0,
) -> PromotionDecision:
    """Evaluate whether a candidate model should be promoted."""
    candidate_score = float(candidate_metrics.get(primary_metric, 0.0))
    baseline_score = float(baseline_metrics.get(primary_metric, 0.0))
    improvement = candidate_score - baseline_score

    reasons: list[str] = []
    if candidate_score < minimum_candidate_score:
        reasons.append("candidate_below_minimum_score")
    if improvement < minimum_improvement:
        reasons.append("improvement_below_threshold")

    return PromotionDecision(
        status="promote" if not reasons else "reject",
        primary_metric=primary_metric,
        candidate_score=candidate_score,
        baseline_score=baseline_score,
        minimum_improvement=minimum_improvement,
        minimum_candidate_score=minimum_candidate_score,
        improvement=improvement,
        reasons=reasons,
    )


def write_promotion_decision(
    output_path: Path,
    candidate_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    *,
    manifest_output_path: Path | None = None,
    candidate_metrics_path: Path | None = None,
    baseline_metrics_path: Path | None = None,
    config_path: Path | None = None,
    execution_mode: str | None = None,
    primary_metric: str = "f1",
    minimum_improvement: float = 0.0,
    minimum_candidate_score: float = 0.0,
) -> PromotionDecision:
    """Persist the promotion decision as JSON."""
    output_path = _resolve_promotion_decision_output_path(Path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_step_manifest(step_name="promote_model", stage_name="model_promote")
    fallback_manifest_path = manifest_path_for_dir(output_path.parent)
    if manifest_output_path:
        manifest_path = resolve_manifest_output_path(Path(manifest_output_path))
        manifest_mirror_path = None
    else:
        manifest_path = fallback_manifest_path
        manifest_mirror_path = None
    resolved_execution_mode = execution_mode or ("azureml" if is_azure_ml() else "local")
    is_smoke = "smoke" in str(config_path or "").lower()
    merge_section(
        manifest,
        "run_context",
        {
            "execution_mode": resolved_execution_mode,
            "is_smoke": is_smoke,
        },
    )
    merge_config(
        manifest,
        config_paths={"train_config": config_path} if config_path else None,
        resolved={"primary_metric": primary_metric},
    )
    merge_section(
        manifest,
        "inputs",
        {
            "candidate_metrics_path": candidate_metrics_path,
            "baseline_metrics_path": baseline_metrics_path,
            "candidate_model_name": candidate_metrics.get("model_name"),
            "baseline_model_name": baseline_metrics.get("model_name"),
        },
    )
    merge_section(
        manifest,
        "outputs",
        {
            "promotion_decision_output": output_path,
            "manifest_output_path": manifest_path,
            "compatibility_manifest_output_path": manifest_mirror_path,
        },
    )
    merge_section(
        manifest,
        "artifacts",
        {
            "promotion_decision": output_path,
            "step_manifest": manifest_path,
            "compatibility_step_manifest": manifest_mirror_path,
        },
    )
    merge_section(
        manifest,
        "params",
        {
            "primary_metric": primary_metric,
            "minimum_improvement": minimum_improvement,
            "minimum_candidate_score": minimum_candidate_score,
        },
    )
    try:
        decision = evaluate_promotion(
            candidate_metrics,
            baseline_metrics,
            primary_metric=primary_metric,
            minimum_improvement=minimum_improvement,
            minimum_candidate_score=minimum_candidate_score,
        )
        output_path.write_text(json.dumps(asdict(decision), indent=2), encoding="utf-8")
        merge_section(
            manifest,
            "metrics",
            {
                "candidate_score": decision.candidate_score,
                "baseline_score": decision.baseline_score,
                "improvement": decision.improvement,
            },
        )
        merge_section(
            manifest,
            "step_specific",
            {
                "decision_status": decision.status,
                "reasons": decision.reasons,
            },
        )
        _finalize_manifest_safely(
            manifest,
            output_path=manifest_path,
            mirror_output_path=manifest_mirror_path,
            status="success",
        )
        return decision
    except Exception as exc:
        set_failure(manifest, phase="write_promotion_decision", exc=exc)
        _finalize_manifest_safely(
            manifest,
            output_path=manifest_path,
            mirror_output_path=manifest_mirror_path,
            status="failed",
        )
        raise


def write_promotion_decision_from_files(
    *,
    candidate_metrics_path: Path,
    baseline_metrics_path: Path,
    output_path: Path,
    manifest_output_path: Path | None = None,
    config_path: Path | None = None,
    execution_mode: str | None = None,
    primary_metric: str = "f1",
    minimum_improvement: float = 0.0,
    minimum_candidate_score: float = 0.0,
) -> PromotionDecision:
    """Load metrics from files, evaluate promotion, and persist the result."""
    candidate_metrics_path = resolve_named_input_file(
        candidate_metrics_path,
        CANDIDATE_METRICS_FILENAME,
    )
    candidate_metrics = json.loads(candidate_metrics_path.read_text(encoding="utf-8"))
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    return write_promotion_decision(
        output_path=Path(output_path),
        manifest_output_path=Path(manifest_output_path) if manifest_output_path else None,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        candidate_metrics_path=candidate_metrics_path,
        baseline_metrics_path=baseline_metrics_path,
        config_path=config_path,
        execution_mode=execution_mode,
        primary_metric=primary_metric,
        minimum_improvement=minimum_improvement,
        minimum_candidate_score=minimum_candidate_score,
    )


def main() -> None:
    """CLI entrypoint for AML promotion evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate candidate metrics against a baseline and write a promotion decision.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-metrics", required=True, help="Path to candidate metrics JSON")
    parser.add_argument("--baseline-metrics", required=True, help="Path to baseline metrics JSON")
    parser.add_argument("--output", required=True, help="Path to write promotion decision JSON")
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional file or directory path to write the promotion step manifest JSON.",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "train.yaml"),
        help="Training config path that owns promotion thresholds",
    )
    args = parser.parse_args()

    promotion_config = load_promotion_config(Path(args.config))
    decision = write_promotion_decision_from_files(
        candidate_metrics_path=Path(args.candidate_metrics),
        baseline_metrics_path=Path(args.baseline_metrics),
        output_path=Path(args.output),
        manifest_output_path=Path(args.manifest_output) if args.manifest_output else None,
        config_path=Path(args.config),
        primary_metric=promotion_config.primary_metric,
        minimum_improvement=promotion_config.minimum_improvement,
        minimum_candidate_score=promotion_config.minimum_candidate_score,
    )
    print(f"Promotion status: {decision.status}")


if __name__ == "__main__":
    main()
