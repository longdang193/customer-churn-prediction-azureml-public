"""
@meta
type: test
scope: unit
domain: hpo-summary
covers:
  - HPO candidate summary generation
  - Winner selection for the configured primary metric
  - Summary report and manifest persistence
excludes:
  - Real Azure ML execution
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_write_hpo_summary_artifacts_selects_best_candidate() -> None:
    """
    @proves hpo.emit-hpo-summary-artifacts-hpo-summary-json-hpo
    @proves hpo.surface-per-family-model-output-mlflow-model-candidate
    """
    from src.collect_hpo_results import write_hpo_summary_artifacts

    temp_dir = _make_temp_dir()
    try:
        logreg_metrics = temp_dir / "logreg_metrics"
        rf_metrics = temp_dir / "rf_metrics"
        summary_output = temp_dir / "hpo_summary"
        report_output = temp_dir / "hpo_summary_report"
        manifest_output = temp_dir / "hpo_manifest"
        logreg_metrics.mkdir()
        rf_metrics.mkdir()

        (logreg_metrics / "candidate_metrics.json").write_text(
            json.dumps(
                {
                    "model_name": "logreg",
                    "run_id": "logreg-run",
                    "f1": 0.71,
                    "roc_auc": 0.75,
                }
            ),
            encoding="utf-8",
        )
        (rf_metrics / "candidate_metrics.json").write_text(
            json.dumps(
                {
                    "model_name": "rf",
                    "run_id": "rf-run",
                    "f1": 0.82,
                    "roc_auc": 0.8,
                }
            ),
            encoding="utf-8",
        )

        summary = write_hpo_summary_artifacts(
            primary_metric="f1",
            metric_paths={
                "logreg": str(logreg_metrics),
                "rf": str(rf_metrics),
                "xgboost": None,
            },
            family_manifest_paths={
                "logreg": {
                    "hpo_manifest": str(temp_dir / "logreg_hpo_manifest"),
                    "train_manifest": str(temp_dir / "logreg_train_manifest"),
                },
                "rf": {
                    "hpo_manifest": str(temp_dir / "rf_hpo_manifest"),
                    "train_manifest": str(temp_dir / "rf_train_manifest"),
                },
            },
            family_bundle_paths={
                "logreg": {
                    "model_output": str(temp_dir / "logreg_model_output"),
                    "mlflow_model": str(temp_dir / "logreg_mlflow_model"),
                },
                "rf": {
                    "model_output": str(temp_dir / "rf_model_output"),
                    "mlflow_model": str(temp_dir / "rf_mlflow_model"),
                },
            },
            summary_output=summary_output,
            report_output=report_output,
            manifest_output=manifest_output,
            config_paths={
                "data_config": "configs/data_smoke_eval.yaml",
                "hpo_config": "configs/hpo_smoke.yaml",
            },
            data_lineage={
                "current_data_asset": "churn-data-smoke-eval",
                "reference_data_asset": "churn-data-smoke",
            },
        )

        assert summary["winner"]["model_name"] == "rf"
        assert summary["winner"]["run_id"] == "rf-run"
        assert summary["winner"]["score"] == 0.82
        assert summary["winner"]["tie_break_reason"] == "primary_metric"
        assert (summary_output / "hpo_summary.json").exists()
        assert (report_output / "hpo_summary_report.md").exists()
        assert (manifest_output / "step_manifest.json").exists()
        assert summary["family_artifacts"]["rf"]["hpo_manifest"].endswith(
            "rf_hpo_manifest/step_manifest.json"
        )
        assert summary["family_bundle_artifacts"]["rf"]["model_output"].endswith(
            "rf_model_output"
        )
        assert summary["candidate_results"][1]["mlflow_model"].endswith("rf_mlflow_model")
        assert summary["family_artifacts"]["logreg"]["train_manifest"].endswith(
            "logreg_train_manifest/step_manifest.json"
        )
        manifest = json.loads((manifest_output / "step_manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "success"
        assert manifest["tags"]["winner_model"] == "rf"
        assert manifest["metrics"]["candidate_count"] == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_hpo_summary_artifacts_fails_when_no_candidates_exist() -> None:
    from src.collect_hpo_results import write_hpo_summary_artifacts

    temp_dir = _make_temp_dir()
    try:
        summary_output = temp_dir / "hpo_summary.json"
        report_output = temp_dir / "hpo_summary.md"
        manifest_output = temp_dir / "hpo_manifest"

        try:
            write_hpo_summary_artifacts(
                primary_metric="f1",
                metric_paths={"logreg": None, "rf": None, "xgboost": None},
                family_manifest_paths={},
                family_bundle_paths={},
                summary_output=summary_output,
                report_output=report_output,
                manifest_output=manifest_output,
                config_paths={},
                data_lineage={},
            )
        except RuntimeError as exc:
            assert "No candidate metrics" in str(exc)
        else:
            raise AssertionError("Expected HPO summary generation to fail without candidates.")

        manifest = json.loads((manifest_output / "step_manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "failed"
        assert manifest["step_specific"]["failure"]["phase"] == "collect_hpo_results"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_hpo_summary_artifacts_breaks_primary_metric_ties_with_roc_auc() -> None:
    """
    @proves fixed-train.deterministic-hpo-family-selection
    @proves fixed-train.surface-hpo-tie-break-metadata
    """
    from src.collect_hpo_results import write_hpo_summary_artifacts

    temp_dir = _make_temp_dir()
    try:
        rf_metrics = temp_dir / "rf_metrics.json"
        xgboost_metrics = temp_dir / "xgboost_metrics.json"

        rf_metrics.write_text(
            json.dumps(
                {
                    "model_name": "rf",
                    "run_id": "rf-run",
                    "f1": 0.82,
                    "roc_auc": 0.88,
                }
            ),
            encoding="utf-8",
        )
        xgboost_metrics.write_text(
            json.dumps(
                {
                    "model_name": "xgboost",
                    "run_id": "xgb-run",
                    "f1": 0.82,
                    "roc_auc": 0.91,
                }
            ),
            encoding="utf-8",
        )

        summary = write_hpo_summary_artifacts(
            primary_metric="f1",
            metric_paths={
                "logreg": None,
                "rf": str(rf_metrics),
                "xgboost": str(xgboost_metrics),
            },
            family_manifest_paths={},
            family_bundle_paths={},
            summary_output=temp_dir / "hpo_summary.json",
            report_output=temp_dir / "hpo_summary.md",
            manifest_output=temp_dir / "hpo_manifest",
            config_paths={},
            data_lineage={},
        )

        assert summary["winner"]["model_name"] == "xgboost"
        assert summary["winner"]["run_id"] == "xgb-run"
        assert summary["winner"]["tie_break_reason"] == "secondary_metric"
        assert summary["selection_policy"]["secondary_metric"] == "roc_auc"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_hpo_summary_artifacts_breaks_full_metric_ties_with_family_priority() -> None:
    """
    @proves fixed-train.deterministic-hpo-family-selection
    @proves fixed-train.surface-hpo-tie-break-metadata
    """
    from src.collect_hpo_results import write_hpo_summary_artifacts

    temp_dir = _make_temp_dir()
    try:
        rf_metrics = temp_dir / "rf_metrics.json"
        xgboost_metrics = temp_dir / "xgboost_metrics.json"
        hpo_config = temp_dir / "hpo_smoke.yaml"

        rf_metrics.write_text(
            json.dumps(
                {
                    "model_name": "rf",
                    "run_id": "rf-run",
                    "f1": 0.8571428571428571,
                    "roc_auc": 0.9375,
                }
            ),
            encoding="utf-8",
        )
        xgboost_metrics.write_text(
            json.dumps(
                {
                    "model_name": "xgboost",
                    "run_id": "xgb-run",
                    "f1": 0.8571428571428571,
                    "roc_auc": 0.9375,
                }
            ),
            encoding="utf-8",
        )
        hpo_config.write_text(
            "\n".join(
                [
                    "metric: f1",
                    "selection:",
                    "  secondary_metric: roc_auc",
                    "  family_priority:",
                    "    - logreg",
                    "    - rf",
                    "    - xgboost",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        summary = write_hpo_summary_artifacts(
            primary_metric="f1",
            metric_paths={
                "logreg": None,
                "rf": str(rf_metrics),
                "xgboost": str(xgboost_metrics),
            },
            family_manifest_paths={},
            family_bundle_paths={},
            summary_output=temp_dir / "hpo_summary.json",
            report_output=temp_dir / "hpo_summary.md",
            manifest_output=temp_dir / "hpo_manifest",
            config_paths={"hpo_config": str(hpo_config)},
            data_lineage={},
        )

        assert summary["winner"]["model_name"] == "rf"
        assert summary["winner"]["tie_break_reason"] == "family_priority"
        assert summary["winner"]["tie_candidates"] == ["rf", "xgboost"]
        assert summary["selection_policy"]["family_priority"] == ["logreg", "rf", "xgboost"]
        report = (temp_dir / "hpo_summary.md").read_text(
            encoding="utf-8"
        )
        assert "family priority" in report
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
