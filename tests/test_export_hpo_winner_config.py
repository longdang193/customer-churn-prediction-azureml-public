"""
@meta
type: test
scope: unit
domain: hpo-handoff
covers:
  - Exporting the winning HPO family into a fixed training config
  - Reusing downloaded HPO manifests instead of bespoke parsing logic
  - Root wrapper compatibility for the HPO winner export command
excludes:
  - Real Azure ML downloads
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


def test_export_winner_config_materializes_fixed_rf_config() -> None:
    """
    @proves hpo.treat-inspect-hpo-run-py-export-hpo-winner
    @proves hpo.analyze-completed-sweep-results-update-configs-train-yaml
    @proves fixed-train.accept-hpo-winner-train-config
    @proves fixed-train.preserve-train-config-identity
    @proves fixed-train.carry-manifest-lineage
    """
    from export_hpo_winner_config import export_winner_config

    temp_dir = _make_temp_dir()
    try:
        run_dir = temp_dir / "downloaded_hpo"
        named_outputs = run_dir / "named-outputs"
        (named_outputs / "hpo_summary").mkdir(parents=True)
        (named_outputs / "rf_hpo_manifest").mkdir(parents=True)
        (named_outputs / "rf_train_manifest").mkdir(parents=True)

        (named_outputs / "hpo_summary" / "hpo_summary").write_text(
            json.dumps(
                {
                    "primary_metric": "f1",
                    "winner": {
                        "model_name": "rf",
                        "run_id": "rf-run",
                        "score": 0.8571,
                    },
                    "candidate_results": [],
                    "family_bundle_artifacts": {},
                    "family_artifacts": {
                        "rf": {
                            "hpo_manifest": "rf_hpo_manifest/step_manifest.json",
                            "train_manifest": "rf_train_manifest/step_manifest.json",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        (named_outputs / "rf_hpo_manifest" / "step_manifest.json").write_text(
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
        (named_outputs / "rf_train_manifest" / "step_manifest.json").write_text(
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
        base_config = temp_dir / "train_base.yaml"
        base_config.write_text(
            yaml.safe_dump(
                {
                    "training": {
                        "experiment_name": "train-prod",
                        "display_name": "train-prod-run",
                        "models": ["xgboost"],
                        "class_weight": "balanced",
                        "random_state": 42,
                        "use_smote": False,
                        "hyperparameters": {"xgboost": {"n_estimators": 100}},
                    },
                    "promotion": {
                        "primary_metric": "f1",
                        "minimum_improvement": 0.0,
                        "minimum_candidate_score": 0.7,
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        output_path = temp_dir / "train_hpo_winner.yaml"

        exported = export_winner_config(
            run_dir=run_dir,
            output_config=output_path,
            base_config_path=base_config,
            experiment_name="train-smoke-hpo-winner-rf",
            display_name="train-smoke-hpo-winner-rf",
        )

        assert exported["training"]["models"] == ["rf"]
        assert exported["training"]["experiment_name"] == "train-smoke-hpo-winner-rf"
        assert exported["training"]["display_name"] == "train-smoke-hpo-winner-rf"
        assert exported["training"]["class_weight"] == "balanced"
        assert exported["training"]["random_state"] == 42
        assert exported["training"]["use_smote"] is True
        assert exported["training"]["hyperparameters"] == {
            "rf": {
                "n_estimators": 200,
                "max_depth": 4,
                "min_samples_split": 2,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            }
        }
        assert exported["promotion"]["minimum_candidate_score"] == 0.7
        assert exported["lineage"]["canonical_train_config"] == output_path.as_posix()
        persisted = yaml.safe_load(output_path.read_text(encoding="utf-8"))
        assert persisted == exported
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
