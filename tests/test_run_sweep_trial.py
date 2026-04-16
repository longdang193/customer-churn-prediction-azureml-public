"""
@meta
type: test
scope: unit
domain: hpo
covers:
  - Sweep trial CLI forwarding into train.py
  - Manifest and candidate-metrics output propagation for Azure ML HPO runs
excludes:
  - Real subprocess training execution
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import uuid


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_run_sweep_trial_forwards_manifest_and_metrics_outputs(monkeypatch) -> None:
    import src.run_sweep_trial as run_sweep_trial

    observed: dict[str, object] = {}
    temp_dir = _make_temp_dir()
    manifest_output = temp_dir / "forwarded-hpo-manifest"

    def fake_run(cli: list[str], check: bool) -> None:
        observed["cli"] = cli
        observed["check"] = check

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sweep_trial.py",
            "--data",
            "azureml://jobs/fake/outputs/processed_data",
            "--config",
            "configs/train.yaml",
            "--hpo-config",
            "configs/hpo_smoke.yaml",
            "--model-type",
            "rf",
            "--model-output",
            "azureml://jobs/fake/outputs/model_output",
            "--candidate-metrics-output",
            "azureml://jobs/fake/outputs/candidate_metrics",
            "--mlflow-model-output",
            "azureml://jobs/fake/outputs/mlflow_model",
            "--manifest-output",
            "azureml://jobs/fake/outputs/train_manifest",
            "--hpo-manifest-output",
            str(manifest_output),
            "--rf_n_estimators",
            "100",
        ],
    )

    try:
        run_sweep_trial.main()

        cli = observed["cli"]
        assert observed["check"] is True
        assert "--config" in cli
        assert "configs/train.yaml" in cli
        assert "--hpo-config" not in cli
        assert "--model-artifact-dir" in cli
        assert "azureml://jobs/fake/outputs/model_output" in cli
        assert "--candidate-metrics-output" in cli
        assert "azureml://jobs/fake/outputs/candidate_metrics" in cli
        assert "--mlflow-model-output" in cli
        assert "azureml://jobs/fake/outputs/mlflow_model" in cli
        assert "--manifest-output" in cli
        assert "azureml://jobs/fake/outputs/train_manifest" in cli
        assert "azureml://jobs/fake/outputs/hpo_manifest" not in cli
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_run_sweep_trial_writes_hpo_manifest_from_candidate_metrics(monkeypatch) -> None:
    import src.run_sweep_trial as run_sweep_trial

    temp_dir = _make_temp_dir()
    candidate_metrics_path = temp_dir / "candidate_metrics"
    candidate_metrics_path.mkdir()
    (candidate_metrics_path / "candidate_metrics.json").write_text(
        '{"model_name":"rf","run_id":"rf-run","f1":0.82,"roc_auc":0.9}',
        encoding="utf-8",
    )
    manifest_output = temp_dir / "hpo_manifest"

    def fake_run(cli: list[str], check: bool) -> None:
        assert check is True

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sweep_trial.py",
            "--data",
            "azureml://jobs/fake/outputs/processed_data",
            "--config",
            "configs/train.yaml",
            "--hpo-config",
            "configs/hpo_smoke.yaml",
            "--model-type",
            "rf",
            "--model-output",
            "azureml://jobs/fake/outputs/model_output",
            "--candidate-metrics-output",
            str(candidate_metrics_path),
            "--manifest-output",
            "azureml://jobs/fake/outputs/train_manifest",
            "--hpo-manifest-output",
            str(manifest_output),
            "--rf_n_estimators",
            "100",
        ],
    )

    try:
        run_sweep_trial.main()

        manifest_path = manifest_output / "step_manifest.json"
        assert manifest_path.exists()
        content = manifest_path.read_text(encoding="utf-8")
        assert '"step_name": "hpo_trial"' in content
        assert '"candidate_model_name": "rf"' in content
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
