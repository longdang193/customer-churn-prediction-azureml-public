"""
@meta
type: test
scope: integration
domain: smoke-preflight
covers:
  - Local smoke preflight for prep, validation, and training
excludes:
  - Real Azure ML job submission
tags:
  - smoke
  - ci-safe
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def test_run_smoke_preflight_produces_local_stage_artifacts(monkeypatch) -> None:
    from src.smoke_preflight import run_smoke_preflight

    output_root = Path(__file__).resolve().parents[1] / ".tmp-tests" / "smoke-preflight"
    if output_root.exists():
        shutil.rmtree(output_root, ignore_errors=True)

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    try:
        summary = run_smoke_preflight(output_root=output_root, clean=True)

        assert summary["status"] == "passed"
        assert summary["validation_status"] == "passed"
        assert summary["trained_models"] == ["logreg"]
        assert (output_root / "prepared" / "X_train.csv").exists()
        assert (output_root / "validation" / "evidently_report.html").exists()
        assert (output_root / "candidate_metrics.json").exists()
        assert (output_root / "mlflow_model" / "MLmodel").exists()
        assert (output_root / "prepared" / "step_manifest.json").exists()
        assert (output_root / "validation" / "step_manifest.json").exists()
        assert (output_root / "model_artifact" / "step_manifest.json").exists()

        written_summary = json.loads(
            (output_root / "smoke_preflight_summary.json").read_text(encoding="utf-8")
        )
        assert written_summary["status"] == "passed"
        assert set(written_summary["step_manifests"]) == {"data_prep", "validate_data", "train"}
    finally:
        shutil.rmtree(output_root, ignore_errors=True)
