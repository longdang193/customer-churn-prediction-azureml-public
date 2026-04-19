"""
@meta
name: collect_hpo_results
type: script
domain: hpo
responsibility:
  - Collect best-candidate metrics from HPO sweep branches.
  - Select the best model family for the configured primary metric.
  - Emit summary artifacts and a structured HPO step manifest.
inputs:
  - Candidate metrics files from HPO sweep branches
  - Primary metric name
  - Optional config and lineage metadata
outputs:
  - HPO summary JSON
  - HPO summary Markdown report
  - HPO step manifest
tags:
  - azure-ml
  - hpo
  - summary
features:
  - notebook-hpo
  - model-training-pipeline
capabilities:
  - hpo.emit-hpo-summary-artifacts-hpo-summary-json-hpo
  - hpo.surface-per-family-model-output-mlflow-model-candidate
  - fixed-train.deterministic-hpo-family-selection
  - fixed-train.surface-hpo-tie-break-metadata
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import yaml

from utils.output_paths import resolve_named_input_file, resolve_named_output_file
from utils.step_manifest import (
    STEP_MANIFEST_FILENAME,
    build_step_manifest,
    finalize_manifest,
    merge_config,
    merge_section,
    resolve_manifest_output_path,
    set_failure,
)


METRIC_KEYS_TO_SKIP = {"model_name", "run_id"}
CANDIDATE_METRICS_FILENAME = "candidate_metrics.json"
HPO_SUMMARY_FILENAME = "hpo_summary.json"
HPO_SUMMARY_REPORT_FILENAME = "hpo_summary_report.md"


@dataclass(frozen=True)
class CandidateMetrics:
    model_name: str
    run_id: str
    metrics: dict[str, float]
    source_path: Path


@dataclass(frozen=True)
class HPOSelectionPolicy:
    primary_metric: str
    secondary_metric: str
    family_priority: tuple[str, ...]
    final_fallback: str = "model_name"


@dataclass(frozen=True)
class WinnerSelection:
    candidate: CandidateMetrics
    tie_break_reason: str
    tie_candidates: tuple[str, ...]
    selection_policy: HPOSelectionPolicy


def _read_candidate_metrics(path: Path) -> CandidateMetrics:
    path = resolve_named_input_file(path, CANDIDATE_METRICS_FILENAME)
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_name = str(payload["model_name"])
    run_id = str(payload["run_id"])
    metrics = {
        str(key): float(value)
        for key, value in payload.items()
        if key not in METRIC_KEYS_TO_SKIP
    }
    return CandidateMetrics(
        model_name=model_name,
        run_id=run_id,
        metrics=metrics,
        source_path=path,
    )


def load_candidate_metrics(metric_paths: dict[str, str | None]) -> list[CandidateMetrics]:
    """Load all available candidate-metrics artifacts."""
    results: list[CandidateMetrics] = []
    for path_str in metric_paths.values():
        if not path_str:
            continue
        path = Path(path_str)
        if not path.exists():
            continue
        results.append(_read_candidate_metrics(path))
    return results


def _load_selection_policy(
    *,
    primary_metric: str,
    hpo_config_path: str | None,
) -> HPOSelectionPolicy:
    default_priority = ("logreg", "rf", "xgboost")
    if not hpo_config_path:
        return HPOSelectionPolicy(
            primary_metric=primary_metric,
            secondary_metric="roc_auc",
            family_priority=default_priority,
        )

    config_path = Path(hpo_config_path)
    if not config_path.exists():
        return HPOSelectionPolicy(
            primary_metric=primary_metric,
            secondary_metric="roc_auc",
            family_priority=default_priority,
        )

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    selection = payload.get("selection", {})
    if not isinstance(selection, dict):
        selection = {}

    secondary_metric = str(selection.get("secondary_metric", "roc_auc"))
    family_priority_raw = selection.get("family_priority", list(default_priority))
    family_priority = tuple(
        str(family)
        for family in family_priority_raw
        if isinstance(family, (str, int, float))
    ) or default_priority
    return HPOSelectionPolicy(
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
        family_priority=family_priority,
    )


def _family_priority_rank(model_name: str, family_priority: tuple[str, ...]) -> int:
    try:
        return family_priority.index(model_name)
    except ValueError:
        return len(family_priority)


def select_best_candidate(
    candidates: list[CandidateMetrics],
    *,
    primary_metric: str,
    selection_policy: HPOSelectionPolicy | None = None,
) -> WinnerSelection:
    """Select the top candidate for the configured primary metric."""
    if not candidates:
        raise RuntimeError("No candidate metrics were available for HPO summary.")
    missing_metric = [
        candidate.model_name
        for candidate in candidates
        if primary_metric not in candidate.metrics
    ]
    if missing_metric:
        raise RuntimeError(
            f"Primary metric '{primary_metric}' missing for candidates: {', '.join(missing_metric)}"
        )
    policy = selection_policy or HPOSelectionPolicy(
        primary_metric=primary_metric,
        secondary_metric="roc_auc",
        family_priority=("logreg", "rf", "xgboost"),
    )

    top_primary = max(candidate.metrics[primary_metric] for candidate in candidates)
    primary_tied = [
        candidate
        for candidate in candidates
        if candidate.metrics[primary_metric] == top_primary
    ]
    if len(primary_tied) == 1:
        return WinnerSelection(
            candidate=primary_tied[0],
            tie_break_reason="primary_metric",
            tie_candidates=(primary_tied[0].model_name,),
            selection_policy=policy,
        )

    secondary_metric = policy.secondary_metric
    top_secondary = max(
        candidate.metrics.get(secondary_metric, float("-inf")) for candidate in primary_tied
    )
    secondary_tied = [
        candidate
        for candidate in primary_tied
        if candidate.metrics.get(secondary_metric, float("-inf")) == top_secondary
    ]
    if len(secondary_tied) == 1:
        return WinnerSelection(
            candidate=secondary_tied[0],
            tie_break_reason="secondary_metric",
            tie_candidates=tuple(sorted(candidate.model_name for candidate in primary_tied)),
            selection_policy=policy,
        )

    winner = min(
        secondary_tied,
        key=lambda candidate: (
            _family_priority_rank(candidate.model_name, policy.family_priority),
            candidate.model_name,
            candidate.run_id,
        ),
    )
    same_priority = [
        candidate
        for candidate in secondary_tied
        if _family_priority_rank(candidate.model_name, policy.family_priority)
        == _family_priority_rank(winner.model_name, policy.family_priority)
    ]
    tie_break_reason = "family_priority" if len(same_priority) == 1 else "final_fallback"
    return WinnerSelection(
        candidate=winner,
        tie_break_reason=tie_break_reason,
        tie_candidates=tuple(sorted(candidate.model_name for candidate in secondary_tied)),
        selection_policy=policy,
    )


def _resolve_manifest_reference(path_str: str | None) -> str | None:
    if not path_str:
        return None
    candidate_path = Path(path_str)
    if candidate_path.suffix.lower() == ".json":
        return candidate_path.as_posix()
    return (candidate_path / STEP_MANIFEST_FILENAME).as_posix()


def _normalize_family_artifacts(
    family_manifest_paths: dict[str, dict[str, str | None]],
) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    for family, artifact_map in family_manifest_paths.items():
        resolved_artifacts = {
            key: resolved_path
            for key, value in artifact_map.items()
            if (resolved_path := _resolve_manifest_reference(value)) is not None
        }
        if resolved_artifacts:
            normalized[family] = resolved_artifacts
    return normalized


def _normalize_family_bundle_artifacts(
    family_bundle_paths: dict[str, dict[str, str | None]],
) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    for family, artifact_map in family_bundle_paths.items():
        resolved_artifacts = {
            key: Path(value).as_posix()
            for key, value in artifact_map.items()
            if value is not None
        }
        if resolved_artifacts:
            normalized[family] = resolved_artifacts
    return normalized


def build_hpo_summary(
    *,
    primary_metric: str,
    candidates: list[CandidateMetrics],
    config_paths: dict[str, str | None],
    data_lineage: dict[str, str | None],
    family_manifest_paths: dict[str, dict[str, str | None]],
    family_bundle_paths: dict[str, dict[str, str | None]],
) -> dict[str, Any]:
    """Build the machine-readable HPO summary payload."""
    selection_policy = _load_selection_policy(
        primary_metric=primary_metric,
        hpo_config_path=config_paths.get("hpo_config"),
    )
    winner_selection = select_best_candidate(
        candidates,
        primary_metric=primary_metric,
        selection_policy=selection_policy,
    )
    best_candidate = winner_selection.candidate
    normalized_family_artifacts = _normalize_family_artifacts(family_manifest_paths)
    normalized_family_bundles = _normalize_family_bundle_artifacts(family_bundle_paths)
    return {
        "primary_metric": primary_metric,
        "selection_policy": {
            "primary_metric": winner_selection.selection_policy.primary_metric,
            "secondary_metric": winner_selection.selection_policy.secondary_metric,
            "family_priority": list(winner_selection.selection_policy.family_priority),
            "final_fallback": winner_selection.selection_policy.final_fallback,
        },
        "winner": {
            "model_name": best_candidate.model_name,
            "run_id": best_candidate.run_id,
            "score": best_candidate.metrics[primary_metric],
            "tie_break_reason": winner_selection.tie_break_reason,
            "tie_candidates": list(winner_selection.tie_candidates),
        },
        "candidate_results": [
            {
                "model_name": candidate.model_name,
                "run_id": candidate.run_id,
                "metrics": candidate.metrics,
                "source_path": str(candidate.source_path),
                **normalized_family_bundles.get(candidate.model_name, {}),
                **normalized_family_artifacts.get(candidate.model_name, {}),
            }
            for candidate in candidates
        ],
        "family_bundle_artifacts": normalized_family_bundles,
        "family_artifacts": normalized_family_artifacts,
        "config_paths": {
            key: value
            for key, value in config_paths.items()
            if value is not None
        },
        "data_lineage": {
            key: value
            for key, value in data_lineage.items()
            if value is not None
        },
    }


def write_hpo_summary_report(summary: dict[str, Any], output_path: Path) -> None:
    """Write a concise Markdown summary for operators."""
    winner = summary["winner"]
    tie_break_reason = winner.get("tie_break_reason")
    lines = [
        "# HPO Summary",
        "",
        f"- Primary metric: `{summary['primary_metric']}`",
        f"- Winner: `{winner['model_name']}`",
        f"- Winner run id: `{winner['run_id']}`",
        f"- Winner score: `{winner['score']}`",
    ]
    if tie_break_reason and tie_break_reason != "primary_metric":
        tie_break_label = str(tie_break_reason).replace("_", " ")
        lines.extend(
            [
                f"- Tie break: `{tie_break_label}`",
                f"- Tie candidates: `{', '.join(winner.get('tie_candidates', []))}`",
            ]
        )
    lines.extend(
        [
            "",
        "## Candidate Results",
        "",
        "| Model | Run ID | Metrics |",
        "| --- | --- | --- |",
        ]
    )
    for candidate in summary["candidate_results"]:
        metrics_str = ", ".join(
            f"{key}={value}"
            for key, value in sorted(candidate["metrics"].items())
        )
        lines.append(
            f"| {candidate['model_name']} | {candidate['run_id']} | {metrics_str} |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hpo_summary_artifacts(
    *,
    primary_metric: str,
    metric_paths: dict[str, str | None],
    summary_output: Path,
    report_output: Path,
    manifest_output: Path,
    config_paths: dict[str, str | None],
    data_lineage: dict[str, str | None],
    family_manifest_paths: dict[str, dict[str, str | None]],
    family_bundle_paths: dict[str, dict[str, str | None]],
) -> dict[str, Any]:
    """Write HPO summary artifacts and the summary-step manifest."""
    summary_output = resolve_named_output_file(summary_output, HPO_SUMMARY_FILENAME)
    report_output = resolve_named_output_file(report_output, HPO_SUMMARY_REPORT_FILENAME)
    manifest = build_step_manifest(step_name="collect_hpo_results", stage_name="model_sweep")
    manifest_path = resolve_manifest_output_path(manifest_output)
    merge_config(manifest, config_paths=config_paths, resolved={"primary_metric": primary_metric})
    merge_section(
        manifest,
        "inputs",
        {
            "metric_paths": metric_paths,
            "family_manifest_paths": family_manifest_paths,
            "family_bundle_paths": family_bundle_paths,
        },
    )
    merge_section(
        manifest,
        "outputs",
        {
            "summary_output": summary_output,
            "report_output": report_output,
            "manifest_output": manifest_output,
        },
    )
    merge_section(manifest, "params", {"primary_metric": primary_metric})
    merge_section(manifest, "tags", data_lineage)

    try:
        candidates = load_candidate_metrics(metric_paths)
        summary = build_hpo_summary(
            primary_metric=primary_metric,
            candidates=candidates,
            config_paths=config_paths,
            data_lineage=data_lineage,
            family_manifest_paths=family_manifest_paths,
            family_bundle_paths=family_bundle_paths,
        )
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_hpo_summary_report(summary, report_output)
        merge_section(
            manifest,
            "metrics",
            {
                "candidate_count": len(candidates),
                "winner_score": summary["winner"]["score"],
            },
        )
        merge_section(
            manifest,
            "tags",
            {
                "winner_model": summary["winner"]["model_name"],
                "winner_run_id": summary["winner"]["run_id"],
            },
        )
        merge_section(
            manifest,
            "step_specific",
            {
                "summary": summary,
                "selection_policy": summary["selection_policy"],
                "tie_break_reason": summary["winner"]["tie_break_reason"],
                "tie_candidates": summary["winner"]["tie_candidates"],
            },
        )
        finalize_manifest(manifest, output_path=manifest_path, status="success")
        return summary
    except Exception as exc:
        set_failure(manifest, phase="collect_hpo_results", exc=exc)
        finalize_manifest(manifest, output_path=manifest_path, status="failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect best-candidate metrics from HPO sweep branches."
    )
    parser.add_argument("--primary-metric", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--hpo-config", default=None)
    parser.add_argument("--current-data-asset", default=None)
    parser.add_argument("--current-data-version", default=None)
    parser.add_argument("--reference-data-asset", default=None)
    parser.add_argument("--reference-data-version", default=None)
    parser.add_argument("--logreg-metrics", default=None)
    parser.add_argument("--rf-metrics", default=None)
    parser.add_argument("--xgboost-metrics", default=None)
    parser.add_argument("--logreg-hpo-manifest", default=None)
    parser.add_argument("--rf-hpo-manifest", default=None)
    parser.add_argument("--xgboost-hpo-manifest", default=None)
    parser.add_argument("--logreg-model-output", default=None)
    parser.add_argument("--rf-model-output", default=None)
    parser.add_argument("--xgboost-model-output", default=None)
    parser.add_argument("--logreg-mlflow-model", default=None)
    parser.add_argument("--rf-mlflow-model", default=None)
    parser.add_argument("--xgboost-mlflow-model", default=None)
    parser.add_argument("--logreg-train-manifest", default=None)
    parser.add_argument("--rf-train-manifest", default=None)
    parser.add_argument("--xgboost-train-manifest", default=None)
    args = parser.parse_args()

    write_hpo_summary_artifacts(
        primary_metric=args.primary_metric,
        metric_paths={
            "logreg": args.logreg_metrics,
            "rf": args.rf_metrics,
            "xgboost": args.xgboost_metrics,
        },
        summary_output=Path(args.summary_output),
        report_output=Path(args.report_output),
        manifest_output=Path(args.manifest_output),
        config_paths={
            "data_config": args.data_config,
            "hpo_config": args.hpo_config,
        },
        data_lineage={
            "current_data_asset": args.current_data_asset,
            "current_data_version": args.current_data_version,
            "reference_data_asset": args.reference_data_asset,
            "reference_data_version": args.reference_data_version,
        },
        family_manifest_paths={
            "logreg": {
                "hpo_manifest": args.logreg_hpo_manifest,
                "train_manifest": args.logreg_train_manifest,
            },
            "rf": {
                "hpo_manifest": args.rf_hpo_manifest,
                "train_manifest": args.rf_train_manifest,
            },
            "xgboost": {
                "hpo_manifest": args.xgboost_hpo_manifest,
                "train_manifest": args.xgboost_train_manifest,
            },
        },
        family_bundle_paths={
            "logreg": {
                "model_output": args.logreg_model_output,
                "mlflow_model": args.logreg_mlflow_model,
            },
            "rf": {
                "model_output": args.rf_model_output,
                "mlflow_model": args.rf_mlflow_model,
            },
            "xgboost": {
                "model_output": args.xgboost_model_output,
                "mlflow_model": args.xgboost_mlflow_model,
            },
        },
    )


if __name__ == "__main__":
    main()
