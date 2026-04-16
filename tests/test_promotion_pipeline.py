"""
@meta
type: test
scope: unit
domain: promotion
covers:
  - Candidate-metrics artifact generation for the fixed training path
  - Promotion-step wiring in the fixed AML pipeline
excludes:
  - Real Azure ML submission
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

import json
import shutil
import types
import uuid
from pathlib import Path

import pytest


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_write_promotion_decision_from_files_persists_payload() -> None:
    from src.promotion.promote_model import write_promotion_decision_from_files

    temp_dir = _make_temp_dir()
    try:
        candidate_path = temp_dir / "candidate_metrics.json"
        baseline_path = temp_dir / "baseline_metrics.json"
        output_path = temp_dir / "promotion_decision.json"
        candidate_path.write_text(
            json.dumps({"model_name": "xgboost", "f1": 0.81, "roc_auc": 0.88}),
            encoding="utf-8",
        )
        baseline_path.write_text(
            json.dumps({"model_name": "approved", "f1": 0.75, "roc_auc": 0.82}),
            encoding="utf-8",
        )
        manifest_path = temp_dir / "promotion_manifest.json"

        decision = write_promotion_decision_from_files(
            candidate_metrics_path=candidate_path,
            baseline_metrics_path=baseline_path,
            output_path=output_path,
            manifest_output_path=manifest_path,
            primary_metric="f1",
            minimum_improvement=0.01,
            minimum_candidate_score=0.7,
        )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert decision.status == "promote"
        assert payload["candidate_score"] == 0.81
        assert payload["baseline_score"] == 0.75
        assert payload["improvement"] == pytest.approx(0.06)
        assert manifest["step_name"] == "promote_model"
        assert manifest["metrics"]["candidate_score"] == 0.81
        assert manifest["step_specific"]["decision_status"] == "promote"
        assert manifest["outputs"]["manifest_output_path"] == str(manifest_path)
        assert manifest["outputs"]["compatibility_manifest_output_path"] is None
        assert manifest["artifacts"]["step_manifest"] == str(manifest_path)
        assert manifest["artifacts"]["compatibility_step_manifest"] is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_promotion_decision_ignores_manifest_write_failure(monkeypatch) -> None:
    from src.promotion.promote_model import write_promotion_decision_from_files

    temp_dir = _make_temp_dir()
    try:
        candidate_path = temp_dir / "candidate_metrics.json"
        baseline_path = temp_dir / "baseline_metrics.json"
        output_path = temp_dir / "promotion_decision"
        candidate_path.write_text(
            json.dumps({"model_name": "xgboost", "f1": 0.81, "roc_auc": 0.88}),
            encoding="utf-8",
        )
        baseline_path.write_text(
            json.dumps({"model_name": "approved", "f1": 0.75, "roc_auc": 0.82}),
            encoding="utf-8",
        )
        manifest_path = temp_dir / "promotion_manifest.json"

        def fake_finalize_manifest(*args, **kwargs):
            raise FileNotFoundError("synthetic manifest path failure")

        monkeypatch.setattr("src.promotion.promote_model.finalize_manifest", fake_finalize_manifest)

        decision = write_promotion_decision_from_files(
            candidate_metrics_path=candidate_path,
            baseline_metrics_path=baseline_path,
            output_path=output_path,
            manifest_output_path=manifest_path,
            primary_metric="f1",
            minimum_improvement=0.01,
            minimum_candidate_score=0.7,
        )

        payload_path = output_path / "promotion_decision.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert decision.status == "promote"
        assert payload["candidate_score"] == 0.81
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_promotion_decision_writes_json_inside_folder_output() -> None:
    from src.promotion.promote_model import write_promotion_decision_from_files

    temp_dir = _make_temp_dir()
    try:
        candidate_path = temp_dir / "candidate_metrics.json"
        baseline_path = temp_dir / "baseline_metrics.json"
        output_dir = temp_dir / "promotion_decision"
        candidate_path.write_text(
            json.dumps({"model_name": "xgboost", "f1": 0.81, "roc_auc": 0.88}),
            encoding="utf-8",
        )
        baseline_path.write_text(
            json.dumps({"model_name": "approved", "f1": 0.75, "roc_auc": 0.82}),
            encoding="utf-8",
        )

        decision = write_promotion_decision_from_files(
            candidate_metrics_path=candidate_path,
            baseline_metrics_path=baseline_path,
            output_path=output_dir,
            primary_metric="f1",
            minimum_improvement=0.01,
            minimum_candidate_score=0.7,
        )

        payload_path = output_dir / "promotion_decision.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert decision.status == "promote"
        assert payload["status"] == "promote"
        assert payload_path.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_rejected_promotion_decision_keeps_manifest_successful() -> None:
    from src.promotion.promote_model import write_promotion_decision_from_files

    temp_dir = _make_temp_dir()
    try:
        candidate_path = temp_dir / "candidate_metrics.json"
        baseline_path = temp_dir / "baseline_metrics.json"
        output_path = temp_dir / "promotion_decision.json"
        manifest_path = temp_dir / "promotion_manifest.json"
        candidate_path.write_text(
            json.dumps({"model_name": "xgboost", "f1": 0.63, "roc_auc": 0.86}),
            encoding="utf-8",
        )
        baseline_path.write_text(
            json.dumps({"model_name": "approved", "f1": 0.0, "roc_auc": 0.0}),
            encoding="utf-8",
        )

        decision = write_promotion_decision_from_files(
            candidate_metrics_path=candidate_path,
            baseline_metrics_path=baseline_path,
            output_path=output_path,
            manifest_output_path=manifest_path,
            primary_metric="f1",
            minimum_improvement=0.0,
            minimum_candidate_score=0.7,
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert decision.status == "reject"
        assert manifest["status"] == "success"
        assert manifest["step_specific"]["decision_status"] == "reject"
        assert manifest["step_specific"]["reasons"] == ["candidate_below_minimum_score"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_define_pipeline_exposes_promotion_decision_output(monkeypatch) -> None:
    import run_pipeline

    def fake_pipeline(*, compute: str, description: str):
        def decorator(func):
            def wrapped(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped

        return decorator

    fake_dsl = types.SimpleNamespace(pipeline=fake_pipeline)
    fake_module = types.SimpleNamespace(dsl=fake_dsl)

    import sys

    monkeypatch.setitem(sys.modules, "azure.ai.ml", fake_module)

    class FakeOutputs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeJob:
        def __init__(self, **outputs):
            self.outputs = FakeOutputs(**outputs)

    components = {
        "validate_data": lambda **_: FakeJob(
            validation_report="validation-report",
            validation_summary="validation-summary",
            validation_manifest="validation-manifest",
        ),
        "data_prep": lambda **_: FakeJob(
            processed_data="processed-data",
            data_prep_manifest="data-prep-manifest",
        ),
        "train": lambda **_: FakeJob(
            model_output="model-output",
            mlflow_model="mlflow-model",
            train_manifest="train-manifest",
            parent_run_id="parent-run-id",
            candidate_metrics="candidate-metrics",
        ),
        "promote_model": lambda **_: FakeJob(
            promotion_decision="promotion-decision",
            promotion_manifest="promotion-manifest",
        ),
    }

    pipeline = run_pipeline.define_pipeline(
        components,
        compute_name="cpu-cluster",
    )
    outputs = pipeline(
        current_raw_data="current-data",
        reference_raw_data="reference-data",
        baseline_metrics="baseline-metrics",
        data_config="data-config",
        train_config="train-config",
    )

    assert outputs["promotion_decision"] == "promotion-decision"
    assert outputs["promotion_manifest"] == "promotion-manifest"
    assert outputs["validation_manifest"] == "validation-manifest"
    assert outputs["data_prep_manifest"] == "data-prep-manifest"
    assert outputs["train_manifest"] == "train-manifest"
    assert outputs["candidate_metrics"] == "candidate-metrics"
    assert outputs["mlflow_model"] == "mlflow-model"


def test_define_pipeline_can_gate_data_prep_on_validation(monkeypatch) -> None:
    import run_pipeline

    def fake_pipeline(*, compute: str, description: str):
        def decorator(func):
            def wrapped(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped

        return decorator

    fake_dsl = types.SimpleNamespace(pipeline=fake_pipeline)
    fake_module = types.SimpleNamespace(dsl=fake_dsl)

    import sys

    monkeypatch.setitem(sys.modules, "azure.ai.ml", fake_module)

    class FakeOutputs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeJob:
        def __init__(self, **outputs):
            self.outputs = FakeOutputs(**outputs)

    validate_job = FakeJob(
        validation_report="validation-report",
        validation_summary="validation-summary",
        validation_manifest="validation-manifest",
    )
    captured_data_prep_kwargs = {}

    def build_data_prep_job(**kwargs):
        captured_data_prep_kwargs.update(kwargs)
        return FakeJob(
            processed_data="processed-data",
            data_prep_manifest="data-prep-manifest",
        )

    train_job = FakeJob(
        model_output="model-output",
        mlflow_model="mlflow-model",
        train_manifest="train-manifest",
        parent_run_id="parent-run-id",
        candidate_metrics="candidate-metrics",
    )
    promote_job = FakeJob(
        promotion_decision="promotion-decision",
        promotion_manifest="promotion-manifest",
    )

    components = {
        "validate_data": lambda **_: validate_job,
        "data_prep": build_data_prep_job,
        "train": lambda **_: train_job,
        "promote_model": lambda **_: promote_job,
    }

    pipeline = run_pipeline.define_pipeline(
        components,
        compute_name="cpu-cluster",
        gate_validation_before_prep=True,
    )
    pipeline(
        current_raw_data="current-data",
        reference_raw_data="reference-data",
        baseline_metrics="baseline-metrics",
        data_config="data-config",
        train_config="train-config",
    )

    assert captured_data_prep_kwargs["validation_summary"] == "validation-summary"
