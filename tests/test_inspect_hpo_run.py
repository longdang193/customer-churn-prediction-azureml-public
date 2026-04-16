"""
@meta
type: test
scope: unit
domain: hpo-inspection
covers:
  - Downloaded HPO parent-run inspection
  - Per-family manifest and warning aggregation
  - Root wrapper compatibility for the HPO inspection command
excludes:
  - Real Azure ML job download
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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_inspect_downloaded_hpo_run_summarizes_family_artifacts() -> None:
    from inspect_hpo_run import inspect_downloaded_hpo_run

    temp_dir = _make_temp_dir()
    try:
        named_outputs = temp_dir / "named-outputs"
        _write_json(
            named_outputs / "hpo_summary" / "hpo_summary",
            {
                "primary_metric": "f1",
                "winner": {
                    "model_name": "rf",
                    "run_id": "rf-best-run",
                    "score": 0.82,
                },
                "candidate_results": [
                    {
                        "model_name": "rf",
                        "run_id": "rf-best-run",
                        "metrics": {"f1": 0.82, "roc_auc": 0.91},
                        "source_path": "ignored",
                    },
                    {
                        "model_name": "xgboost",
                        "run_id": "xgb-run",
                        "metrics": {"f1": 0.81, "roc_auc": 0.90},
                        "source_path": "ignored",
                    },
                ],
                "family_bundle_artifacts": {
                    "rf": {
                        "model_output": str(named_outputs / "rf_model_output"),
                        "mlflow_model": str(named_outputs / "rf_mlflow_model"),
                    },
                    "xgboost": {
                        "model_output": str(named_outputs / "xgboost_model_output"),
                        "mlflow_model": str(named_outputs / "xgboost_mlflow_model"),
                    },
                },
                "family_artifacts": {
                    "rf": {
                        "hpo_manifest": str(named_outputs / "rf_hpo_manifest" / "step_manifest.json"),
                        "train_manifest": str(
                            named_outputs / "rf_train_manifest" / "step_manifest.json"
                        ),
                    },
                    "xgboost": {
                        "hpo_manifest": str(
                            named_outputs / "xgboost_hpo_manifest" / "step_manifest.json"
                        ),
                        "train_manifest": str(
                            named_outputs / "xgboost_train_manifest" / "step_manifest.json"
                        ),
                    },
                },
            },
        )
        _write_json(
            named_outputs / "validation_summary" / "validation_summary",
            {
                "status": "passed",
                "drift": {"drifted_column_share": 0.2857},
            },
        )
        (named_outputs / "rf_model_output").mkdir(parents=True, exist_ok=True)
        (named_outputs / "rf_model_output" / "rf_model.pkl").write_text("model", encoding="utf-8")
        (named_outputs / "rf_mlflow_model").mkdir(parents=True, exist_ok=True)
        (named_outputs / "rf_mlflow_model" / "MLmodel").write_text("mlflow", encoding="utf-8")
        (named_outputs / "winner_model_output").mkdir(parents=True, exist_ok=True)
        (named_outputs / "winner_model_output" / "rf_model.pkl").write_text("model", encoding="utf-8")
        (named_outputs / "winner_mlflow_model").mkdir(parents=True, exist_ok=True)
        (named_outputs / "winner_mlflow_model" / "MLmodel").write_text("mlflow", encoding="utf-8")
        (named_outputs / "winner_train_manifest").mkdir(parents=True, exist_ok=True)
        (named_outputs / "winner_train_manifest" / "step_manifest.json").write_text("{}", encoding="utf-8")
        (named_outputs / "winner_hpo_manifest").mkdir(parents=True, exist_ok=True)
        (named_outputs / "winner_hpo_manifest" / "step_manifest.json").write_text("{}", encoding="utf-8")
        (named_outputs / "winner_train_config").mkdir(parents=True, exist_ok=True)
        (named_outputs / "winner_train_config" / "train_config.yaml").write_text(
            "training:\n  models:\n    - rf\n",
            encoding="utf-8",
        )
        (named_outputs / "winner_manifest").mkdir(parents=True, exist_ok=True)
        (named_outputs / "winner_manifest" / "step_manifest.json").write_text("{}", encoding="utf-8")
        (named_outputs / "winner_candidate_metrics").write_text("{}", encoding="utf-8")
        _write_json(
            named_outputs / "rf_hpo_manifest" / "step_manifest.json",
            {
                "status": "success",
                "warnings": [],
                "tags": {"winner_run_id": "rf-best-run"},
            },
        )
        _write_json(
            named_outputs / "rf_train_manifest" / "step_manifest.json",
            {
                "status": "success",
                "warnings": ["Very small test split (8 rows); evaluation metrics are not stable."],
                "metrics": {"best_model_f1": 0.82},
            },
        )
        _write_json(
            named_outputs / "xgboost_hpo_manifest" / "step_manifest.json",
            {
                "status": "success",
                "warnings": [],
            },
        )
        _write_json(
            named_outputs / "xgboost_train_manifest" / "step_manifest.json",
            {
                "status": "success",
                "warnings": [],
                "metrics": {"best_model_f1": 0.81},
            },
        )

        report = inspect_downloaded_hpo_run(temp_dir)

        assert report["winner"]["model_name"] == "rf"
        assert report["validation"]["status"] == "passed"
        assert report["validation"]["drifted_column_share"] == 0.2857
        assert report["winner_outputs"]["model_output"].endswith("winner_model_output")
        assert report["winner_outputs"]["mlflow_model"].endswith("winner_mlflow_model")
        assert report["winner_outputs"]["train_config"].endswith("winner_train_config")
        assert report["families"]["rf"]["model_output"].endswith("rf_model_output")
        assert report["families"]["rf"]["mlflow_model"].endswith("rf_mlflow_model")
        assert report["families"]["rf"]["train_manifest"]["status"] == "success"
        assert report["families"]["rf"]["warnings"] == [
            "Very small test split (8 rows); evaluation metrics are not stable."
        ]
        assert report["families"]["xgboost"]["best_trial_run_id"] == "xgb-run"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
