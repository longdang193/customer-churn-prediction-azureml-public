"""
@meta
type: test
scope: unit
domain: training-cli
covers:
  - Training-level CLI override resolution from `--set`
  - Runtime precedence between CLI flags, parsed overrides, and config defaults
excludes:
  - Real model training
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import types
import uuid


TEST_TEMP_ROOT = Path(__file__).resolve().parents[3] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_train_main_applies_training_level_set_overrides(monkeypatch) -> None:
    fake_training_module = types.ModuleType("training")
    fake_training_module.determine_models_to_train = lambda *_args, **_kwargs: ["rf"]
    fake_training_module.is_hpo_mode = lambda _model_type: False
    fake_training_module.prepare_regular_hyperparams = lambda *_args, **_kwargs: {}
    fake_training_module.train_pipeline_stage = lambda **_kwargs: None

    fake_hyperparams_module = types.ModuleType("training.hyperparams")
    fake_hyperparams_module.parse_override_value = (
        lambda value: False if value == "false" else int(value) if value.isdigit() else value
    )

    monkeypatch.setitem(sys.modules, "training", fake_training_module)
    monkeypatch.setitem(sys.modules, "training.hyperparams", fake_hyperparams_module)

    import src.train as train_module

    temp_dir = _make_temp_dir()
    try:
        train_config = temp_dir / "train.yaml"
        train_config.write_text(
            "\n".join(
                [
                    "training:",
                    '  experiment_name: "train-prod"',
                    '  display_name: "train-display"',
                    '  class_weight: "balanced"',
                    "  random_state: 42",
                    '  use_smote: "true"',
                    "  models:",
                    '    - "rf"',
                    "  hyperparameters:",
                    "    rf:",
                    "      n_estimators: 100",
                ]
            ),
            encoding="utf-8",
        )

        captured: dict[str, object] = {}

        monkeypatch.setattr(train_module, "is_hpo_mode", lambda _model_type: False)
        monkeypatch.setattr(train_module, "prepare_regular_hyperparams", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(train_module, "determine_models_to_train", lambda *_args, **_kwargs: ["rf"])
        monkeypatch.setattr(
            train_module,
            "train_pipeline_stage",
            lambda **kwargs: captured.update(kwargs),
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "train.py",
                "--config",
                str(train_config),
                "--set",
                "use_smote=false",
                "--set",
                "random_state=7",
                "--set",
                "class_weight=custom",
            ],
        )

        train_module.main()

        assert captured["use_smote"] is False
        assert captured["random_state"] == 7
        assert captured["class_weight"] == "custom"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
