"""
@meta
type: test
scope: unit
domain: hpo-winner
covers:
  - HPO winner artifact materialization from family-level outputs
  - Winner manifest persistence
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

import yaml


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_materialize_winner_artifacts_copies_selected_family_outputs() -> None:
    """
    @proves hpo.materialize-canonical-winner-outputs-winner-model-output-winner
    @proves hpo.materialize-winner-train-config-train-config-yaml-hpo
    """
    from src.materialize_hpo_winner import materialize_winner_artifacts

    temp_dir = _make_temp_dir()
    try:
        hpo_summary = temp_dir / "hpo_summary"
        hpo_summary.mkdir()
        (hpo_summary / "hpo_summary.json").write_text(
            json.dumps(
                {
                    "selection_policy": {
                        "primary_metric": "f1",
                        "secondary_metric": "roc_auc",
                        "family_priority": ["logreg", "rf", "xgboost"],
                        "final_fallback": "model_name",
                    },
                    "winner": {
                        "model_name": "rf",
                        "run_id": "rf-run",
                        "score": 0.82,
                        "tie_break_reason": "family_priority",
                        "tie_candidates": ["rf", "xgboost"],
                    }
                }
            ),
            encoding="utf-8",
        )
        rf_metrics = temp_dir / "rf_metrics"
        rf_metrics.mkdir()
        (rf_metrics / "candidate_metrics.json").write_text(
            json.dumps({"model_name": "rf", "run_id": "rf-run", "f1": 0.82}),
            encoding="utf-8",
        )
        rf_model_output = temp_dir / "rf_model_output"
        rf_model_output.mkdir()
        (rf_model_output / "rf_model.pkl").write_text("model", encoding="utf-8")
        rf_mlflow_model = temp_dir / "rf_mlflow_model"
        rf_mlflow_model.mkdir()
        (rf_mlflow_model / "MLmodel").write_text("mlflow", encoding="utf-8")
        rf_train_manifest = temp_dir / "rf_train_manifest"
        rf_train_manifest.mkdir()
        (rf_train_manifest / "step_manifest.json").write_text(
            json.dumps(
                {
                    "params": {
                        "class_weight": "balanced",
                        "random_state": 42,
                        "use_smote": True,
                    }
                }
            ),
            encoding="utf-8",
        )
        rf_hpo_manifest = temp_dir / "rf_hpo_manifest"
        rf_hpo_manifest.mkdir()
        (rf_hpo_manifest / "step_manifest.json").write_text(
            json.dumps(
                {
                    "params": {
                        "hyperparameters": {
                            "rf_n_estimators": "200",
                            "rf_max_depth": "4",
                            "rf_min_samples_split": "2",
                            "rf_min_samples_leaf": "2",
                            "rf_max_features": "sqrt",
                            "use_smote": "True",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        base_train_config = temp_dir / "train_base.yaml"
        base_train_config.write_text(
            yaml.safe_dump(
                {
                    "training": {
                        "experiment_name": "train-smoke",
                        "display_name": "train-smoke",
                        "models": ["logreg"],
                        "class_weight": "balanced",
                        "random_state": 42,
                        "use_smote": False,
                        "hyperparameters": {"logreg": {"C": 1.0}},
                    },
                    "promotion": {
                        "primary_metric": "f1",
                        "minimum_improvement": -1.0,
                        "minimum_candidate_score": 0.0,
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        output_root = temp_dir / "winner"
        args = type(
            "Args",
            (),
            {
                "hpo_summary": str(hpo_summary),
                "logreg_metrics": None,
                "rf_metrics": str(rf_metrics),
                "xgboost_metrics": None,
                "logreg_model_output": None,
                "rf_model_output": str(rf_model_output),
                "xgboost_model_output": None,
                "logreg_mlflow_model": None,
                "rf_mlflow_model": str(rf_mlflow_model),
                "xgboost_mlflow_model": None,
                "logreg_train_manifest": None,
                "rf_train_manifest": str(rf_train_manifest),
                "xgboost_train_manifest": None,
                "logreg_hpo_manifest": None,
                "rf_hpo_manifest": str(rf_hpo_manifest),
                "xgboost_hpo_manifest": None,
                "base_train_config": str(base_train_config),
                "winner_candidate_metrics": str(output_root / "winner_candidate_metrics"),
                "winner_model_output": str(output_root / "winner_model_output"),
                "winner_mlflow_model": str(output_root / "winner_mlflow_model"),
                "winner_train_manifest": str(output_root / "winner_train_manifest"),
                "winner_hpo_manifest": str(output_root / "winner_hpo_manifest"),
                "winner_train_config": str(output_root / "winner_train_config"),
                "winner_manifest": str(output_root / "winner_manifest"),
            },
        )()

        materialized = materialize_winner_artifacts(args)

        assert materialized["winner_model_name"] == "rf"
        assert (output_root / "winner_candidate_metrics" / "candidate_metrics.json").exists()
        assert (output_root / "winner_model_output" / "rf_model.pkl").exists()
        assert (output_root / "winner_mlflow_model" / "MLmodel").exists()
        assert (output_root / "winner_train_manifest" / "step_manifest.json").exists()
        assert (output_root / "winner_hpo_manifest" / "step_manifest.json").exists()
        train_config_path = output_root / "winner_train_config" / "train_config.yaml"
        assert train_config_path.exists()
        fixed_train_config = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
        assert fixed_train_config["training"]["models"] == ["rf"]
        assert fixed_train_config["training"]["use_smote"] is True
        assert fixed_train_config["training"]["hyperparameters"] == {
            "rf": {
                "n_estimators": 200,
                "max_depth": 4,
                "min_samples_split": 2,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            }
        }
        assert fixed_train_config["promotion"]["minimum_candidate_score"] == 0.0
        assert (output_root / "winner_manifest" / "step_manifest.json").exists()
        winner_manifest = json.loads(
            (output_root / "winner_manifest" / "step_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert (
            winner_manifest["step_specific"]["materialized_outputs"]["winner_train_config"]
            == train_config_path.as_posix()
        )
        assert winner_manifest["step_specific"]["selection_policy"]["secondary_metric"] == "roc_auc"
        assert winner_manifest["step_specific"]["tie_break_reason"] == "family_priority"
        assert winner_manifest["step_specific"]["tie_candidates"] == ["rf", "xgboost"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
