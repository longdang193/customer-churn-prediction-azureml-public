"""
@meta
name: validate_data
type: script
domain: data-validation
responsibility:
  - Compare current training data against a reference dataset before retraining.
  - Emit Evidently artifacts plus a machine-readable validation summary.
inputs:
  - Current dataset path
  - Reference dataset path
  - configs/data.yaml
outputs:
  - HTML and JSON validation reports
  - Validation summary JSON
tags:
  - evidently
  - data-quality
  - drift
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from utils.config_loader import get_config_value, load_config
from utils.mlflow_utils import is_azure_ml
from utils.output_paths import resolve_named_output_file
from utils.step_manifest import (
    add_warning,
    build_step_manifest,
    finalize_manifest,
    manifest_path_for_dir,
    merge_config,
    merge_section,
    set_failure,
)
from utils.type_utils import parse_bool


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"
DEFAULT_HTML_REPORT_NAME = "evidently_report.html"
DEFAULT_JSON_REPORT_NAME = "evidently_report.json"
DEFAULT_VALIDATION_SUMMARY_NAME = "validation_summary.json"


@dataclass(frozen=True)
class ValidationConfig:
    target_column: str
    drift_share_threshold: float = 0.3
    max_missing_fraction: float = 0.2
    max_row_count_delta_fraction: float = 0.5
    fail_on_drift: bool = False
    fail_on_schema_mismatch: bool = True
    fail_on_missing_fraction: bool = True
    fail_on_row_count_delta: bool = False


def _load_tabular_data(path: Path) -> pd.DataFrame:
    """Load one CSV file or concatenate all CSV files in a directory."""
    if not path.exists():
        raise FileNotFoundError(f"Validation data path does not exist: {path}")
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        return pd.concat((pd.read_csv(csv_file) for csv_file in csv_files), ignore_index=True)
    return pd.read_csv(path)


def load_validation_config(config_path: str | None = None) -> ValidationConfig:
    """Load validation thresholds from configs/data.yaml."""
    config = load_config(config_path or str(DEFAULT_CONFIG))
    data_cfg = get_config_value(config, "data", {}) or {}
    validation_cfg = get_config_value(config, "validation", {}) or {}

    return ValidationConfig(
        target_column=str(data_cfg.get("target_column") or "Exited"),
        drift_share_threshold=float(validation_cfg.get("drift_share_threshold", 0.3)),
        max_missing_fraction=float(validation_cfg.get("max_missing_fraction", 0.2)),
        max_row_count_delta_fraction=float(
            validation_cfg.get("max_row_count_delta_fraction", 0.5)
        ),
        fail_on_drift=parse_bool(
            validation_cfg.get("fail_on_drift", False),
            default=False,
        ),
        fail_on_schema_mismatch=parse_bool(
            validation_cfg.get("fail_on_schema_mismatch", True),
            default=True,
        ),
        fail_on_missing_fraction=parse_bool(
            validation_cfg.get("fail_on_missing_fraction", True),
            default=True,
        ),
        fail_on_row_count_delta=parse_bool(
            validation_cfg.get("fail_on_row_count_delta", False),
            default=False,
        ),
    )


def _build_default_report():
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset
    except ImportError as exc:
        raise RuntimeError(
            "Evidently is required to build validation reports. Install the "
            "'evidently' package or inject a report_factory for tests."
        ) from exc

    return (
        Report,
        Dataset,
        DataDefinition,
        DataDriftPreset,
        DataSummaryPreset,
    )


def _create_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    config: ValidationConfig,
    report_factory: Callable[[], Any] | None,
) -> Any:
    if report_factory is not None:
        report = report_factory()
        return report.run(current_data=current_data, reference_data=reference_data)

    (
        Report,
        Dataset,
        DataDefinition,
        DataDriftPreset,
        DataSummaryPreset,
    ) = _build_default_report()
    data_definition = DataDefinition()
    report = Report(
        [
            DataSummaryPreset(),
            DataDriftPreset(drift_share=config.drift_share_threshold),
        ]
    )
    return report.run(
        current_data=Dataset.from_pandas(current_data, data_definition=data_definition),
        reference_data=Dataset.from_pandas(reference_data, data_definition=data_definition),
    )


def _max_missing_fraction(dataframe: pd.DataFrame) -> float:
    if dataframe.empty:
        return 0.0
    return float(dataframe.isna().mean().max())


def _row_count_delta_fraction(reference_rows: int, current_rows: int) -> float:
    denominator = max(reference_rows, 1)
    return abs(current_rows - reference_rows) / denominator


def _serialize_columns(columns: Iterable[str]) -> list[str]:
    return sorted(str(column) for column in columns)


def _extract_drift_summary(report_json_path: Path) -> dict[str, float | None]:
    """Extract dataset-level drift summary from an Evidently JSON report."""
    report_payload = json.loads(report_json_path.read_text(encoding="utf-8"))
    for metric in report_payload.get("metrics", []):
        metric_type = metric.get("config", {}).get("type")
        metric_name = str(metric.get("metric_name", ""))
        if metric_type != "evidently:metric_v2:DriftedColumnsCount" and not metric_name.startswith(
            "DriftedColumnsCount"
        ):
            continue

        value = metric.get("value", {}) or {}
        return {
            "drifted_column_count": float(value.get("count", 0.0)),
            "drifted_column_share": float(value.get("share", 0.0)),
        }

    return {
        "drifted_column_count": None,
        "drifted_column_share": None,
    }


def _build_validation_summary(
    *,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    config: ValidationConfig,
    failed_checks: list[str],
    drift_summary: dict[str, float | None],
    drift_within_threshold: bool | None,
    max_missing_fraction: float,
    row_count_delta_fraction: float,
    missing_fraction_within_threshold: bool,
    row_count_within_threshold: bool,
    schema_match: bool,
    target_column_present: bool,
) -> dict[str, Any]:
    """Build the validation summary payload shared by success and guardrail failures."""
    return {
        "status": "failed" if failed_checks else "passed",
        "checks": {
            "schema_match": schema_match,
            "target_column_present": target_column_present,
            "missing_fraction_within_threshold": missing_fraction_within_threshold,
            "row_count_within_threshold": row_count_within_threshold,
            "drift_within_threshold": drift_within_threshold,
        },
        "failed_checks": failed_checks,
        "drift": drift_summary,
        "row_counts": {
            "reference": len(reference_data),
            "current": len(current_data),
        },
        "column_counts": {
            "reference": len(reference_data.columns),
            "current": len(current_data.columns),
        },
        "max_missing_fraction": max_missing_fraction,
        "row_count_delta_fraction": row_count_delta_fraction,
        "reference_columns": _serialize_columns(reference_data.columns),
        "current_columns": _serialize_columns(current_data.columns),
        "thresholds": asdict(config),
    }


def _write_skipped_report(output_dir: Path, failed_checks: list[str]) -> None:
    """Persist lightweight report artifacts when pre-report guardrails fail."""
    skipped_report = {
        "status": "skipped",
        "reason": "pre_report_guardrail_failed",
        "failed_checks": failed_checks,
    }
    (output_dir / DEFAULT_JSON_REPORT_NAME).write_text(
        json.dumps(skipped_report, indent=2),
        encoding="utf-8",
    )
    (output_dir / DEFAULT_HTML_REPORT_NAME).write_text(
        "<html><body><h1>Validation report skipped</h1>"
        "<p>Pre-report guardrails failed; see validation_summary.json.</p></body></html>",
        encoding="utf-8",
    )


def run_validation(
    *,
    reference_path: Path,
    current_path: Path,
    output_dir: Path,
    summary_path: Path,
    config: ValidationConfig,
    config_path: Path | None = None,
    execution_mode: str | None = None,
    manifest_output_path: Path | None = None,
    report_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Run validation, persist artifacts, and return the summary."""
    summary_path = resolve_named_output_file(summary_path, DEFAULT_VALIDATION_SUMMARY_NAME)
    manifest = build_step_manifest(step_name="validate_data", stage_name="data_validate")
    manifest_path = manifest_path_for_dir(output_dir)
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
        config_paths={"data_config": config_path} if config_path else None,
        resolved={
            "target_column": config.target_column,
            "drift_share_threshold": config.drift_share_threshold,
            "max_missing_fraction": config.max_missing_fraction,
            "max_row_count_delta_fraction": config.max_row_count_delta_fraction,
            "fail_on_schema_mismatch": config.fail_on_schema_mismatch,
            "fail_on_missing_fraction": config.fail_on_missing_fraction,
            "fail_on_row_count_delta": config.fail_on_row_count_delta,
            "fail_on_drift": config.fail_on_drift,
        },
    )
    merge_section(
        manifest,
        "inputs",
        {
            "reference_path": reference_path,
            "current_path": current_path,
        },
    )
    merge_section(
        manifest,
        "outputs",
        {
            "output_dir": output_dir,
            "summary_path": summary_path,
            "manifest_output_path": manifest_output_path,
        },
    )
    merge_section(
        manifest,
        "artifacts",
        {
            "html_report": output_dir / DEFAULT_HTML_REPORT_NAME,
            "json_report": output_dir / DEFAULT_JSON_REPORT_NAME,
            "summary_json": summary_path,
            "step_manifest": manifest_path,
            "declared_step_manifest": manifest_output_path,
        },
    )

    try:
        reference_data = _load_tabular_data(reference_path)
        current_data = _load_tabular_data(current_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        merge_section(
            manifest,
            "step_specific",
            {
                "reference_columns": _serialize_columns(reference_data.columns),
                "current_columns": _serialize_columns(current_data.columns),
            },
        )

        schema_match = set(reference_data.columns) == set(current_data.columns)
        target_column_present = (
            config.target_column in reference_data.columns
            and config.target_column in current_data.columns
        )
        max_missing_fraction = _max_missing_fraction(current_data)
        missing_fraction_within_threshold = max_missing_fraction <= config.max_missing_fraction
        row_count_delta_fraction = _row_count_delta_fraction(
            len(reference_data), len(current_data)
        )
        row_count_within_threshold = (
            row_count_delta_fraction <= config.max_row_count_delta_fraction
        )

        failed_checks: list[str] = []
        if not target_column_present:
            failed_checks.append("target_column_present")
        if config.fail_on_schema_mismatch and not schema_match:
            failed_checks.append("schema_mismatch")
        if config.fail_on_missing_fraction and not missing_fraction_within_threshold:
            failed_checks.append("max_missing_fraction")
        if config.fail_on_row_count_delta and not row_count_within_threshold:
            failed_checks.append("row_count_delta_fraction")

        drift_summary: dict[str, float | None] = {
            "drifted_column_count": None,
            "drifted_column_share": None,
        }
        drift_within_threshold: bool | None = None
        if failed_checks:
            _write_skipped_report(output_dir, failed_checks)
        else:
            report = _create_report(reference_data, current_data, config, report_factory)
            report.save_html(str(output_dir / DEFAULT_HTML_REPORT_NAME))
            report_json_path = output_dir / DEFAULT_JSON_REPORT_NAME
            report.save_json(str(report_json_path))
            drift_summary = _extract_drift_summary(report_json_path)
            drifted_column_share = drift_summary["drifted_column_share"]
            drift_within_threshold = (
                None
                if drifted_column_share is None
                else drifted_column_share <= config.drift_share_threshold
            )
            if config.fail_on_drift and drifted_column_share is None:
                failed_checks.append("drift_metric_missing")
            elif config.fail_on_drift and not drift_within_threshold:
                failed_checks.append("data_drift_detected")

        summary = _build_validation_summary(
            reference_data=reference_data,
            current_data=current_data,
            config=config,
            failed_checks=failed_checks,
            drift_summary=drift_summary,
            drift_within_threshold=drift_within_threshold,
            max_missing_fraction=max_missing_fraction,
            row_count_delta_fraction=row_count_delta_fraction,
            missing_fraction_within_threshold=missing_fraction_within_threshold,
            row_count_within_threshold=row_count_within_threshold,
            schema_match=schema_match,
            target_column_present=target_column_present,
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        merge_section(
            manifest,
            "metrics",
            {
                "reference_row_count": summary["row_counts"]["reference"],
                "current_row_count": summary["row_counts"]["current"],
                "reference_column_count": summary["column_counts"]["reference"],
                "current_column_count": summary["column_counts"]["current"],
                "max_missing_fraction": max_missing_fraction,
                "row_count_delta_fraction": row_count_delta_fraction,
                "drifted_column_count": drift_summary["drifted_column_count"],
                "drifted_column_share": drift_summary["drifted_column_share"],
                "failed_check_count": len(failed_checks),
            },
        )
        merge_section(
            manifest,
            "params",
            {
                "drift_share_threshold": config.drift_share_threshold,
                "max_missing_fraction_threshold": config.max_missing_fraction,
                "max_row_count_delta_fraction_threshold": config.max_row_count_delta_fraction,
                "fail_on_drift": config.fail_on_drift,
            },
        )
        merge_section(
            manifest,
            "step_specific",
            {
                "checks": summary["checks"],
                "failed_checks": failed_checks,
                "summary_status": summary["status"],
            },
        )
        if failed_checks:
            add_warning(manifest, f"Validation checks failed: {', '.join(failed_checks)}")
        finalized_manifest_path = finalize_manifest(
            manifest,
            output_path=manifest_path,
            mirror_output_path=manifest_output_path,
            status=summary["status"],
        )
        print(f"STEP_MANIFEST_PATH={finalized_manifest_path}")
        return summary
    except Exception as exc:
        set_failure(manifest, phase="run_validation", exc=exc)
        finalized_manifest_path = finalize_manifest(
            manifest,
            output_path=manifest_path,
            mirror_output_path=manifest_output_path,
            status="failed",
        )
        print(f"STEP_MANIFEST_PATH={finalized_manifest_path}")
        raise


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(
        description="Validate batch input data before Azure ML training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference-data", required=True, help="Reference dataset path")
    parser.add_argument("--current-data", required=True, help="Current dataset path")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where validation reports should be written",
    )
    parser.add_argument(
        "--summary-path",
        required=True,
        help="Path to the machine-readable validation summary JSON file",
    )
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional file or directory path to write the validation step manifest JSON.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Validation config file",
    )
    args = parser.parse_args()

    summary = run_validation(
        reference_path=Path(args.reference_data),
        current_path=Path(args.current_data),
        output_dir=Path(args.output_dir),
        summary_path=Path(args.summary_path),
        config=load_validation_config(args.config),
        config_path=Path(args.config),
        manifest_output_path=Path(args.manifest_output) if args.manifest_output else None,
    )
    print(json.dumps(summary, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
