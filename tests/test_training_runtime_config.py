"""
@meta
type: test
scope: unit
domain: training-config
covers:
  - Canonical experiment-name ownership from configs/train.yaml
  - Promotion-threshold loading from configs/train.yaml
excludes:
  - Real training execution
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

from pathlib import Path
import shutil
import uuid


TEST_TEMP_ROOT = Path(__file__).resolve().parents[3] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_load_training_runtime_defaults_prefers_training_experiment_name() -> None:
    from src.config.runtime import load_training_runtime_defaults

    temp_dir = _make_temp_dir()
    try:
        train_config = temp_dir / "train.yaml"
        mlflow_config = temp_dir / "mlflow.yaml"
        train_config.write_text(
            "\n".join(
                [
                    "training:",
                    '  experiment_name: "train-owned-experiment"',
                    '  display_name: "train-display"',
                    '  class_weight: "balanced"',
                    "  random_state: 99",
                    '  use_smote: "false"',
                ]
            ),
            encoding="utf-8",
        )
        mlflow_config.write_text(
            "\n".join(
                [
                    "mlflow:",
                    '  experiment_name: "legacy-mlflow-experiment"',
                ]
            ),
            encoding="utf-8",
        )

        runtime = load_training_runtime_defaults(train_config)

        assert runtime.experiment_name == "train-owned-experiment"
        assert runtime.display_name == "train-display"
        assert runtime.class_weight == "balanced"
        assert runtime.random_state == 99
        assert runtime.use_smote is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_promotion_config_reads_thresholds_from_train_yaml() -> None:
    from src.config.runtime import load_promotion_config

    temp_dir = _make_temp_dir()
    try:
        train_config = temp_dir / "train.yaml"
        train_config.write_text(
            "\n".join(
                [
                    "training:",
                    '  experiment_name: "train-prod"',
                    "promotion:",
                    '  primary_metric: "roc_auc"',
                    "  minimum_improvement: 0.03",
                    "  minimum_candidate_score: 0.81",
                ]
            ),
            encoding="utf-8",
        )

        promotion_config = load_promotion_config(train_config)

        assert promotion_config.primary_metric == "roc_auc"
        assert promotion_config.minimum_improvement == 0.03
        assert promotion_config.minimum_candidate_score == 0.81
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
