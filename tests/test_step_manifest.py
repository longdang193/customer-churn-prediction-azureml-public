"""
@meta
type: test
scope: unit
domain: manifests
covers:
  - Canonical manifest-path selection when declared AML outputs exist
  - Compatibility mirroring to fallback manifest locations
excludes:
  - Real MLflow artifact backend integration
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


def test_finalize_manifest_prefers_declared_output_path(monkeypatch) -> None:
    """
    @proves data-prep.emit-structured-step-manifest-json-artifact-validation-data
    """
    from src.utils import step_manifest

    monkeypatch.setattr(step_manifest, "_log_manifest_artifact", lambda _path: None)
    temp_dir = _make_temp_dir()
    try:
        manifest = step_manifest.build_step_manifest(step_name="train", stage_name="model_sweep")
        fallback_output = temp_dir / "model_artifact_dir" / "step_manifest.json"
        declared_output_dir = temp_dir / "train_manifest"

        canonical_path = step_manifest.finalize_manifest(
            manifest,
            output_path=fallback_output,
            mirror_output_path=declared_output_dir,
            status="success",
        )

        declared_path = declared_output_dir / "step_manifest.json"
        assert canonical_path == declared_path
        assert fallback_output.exists()
        assert declared_path.exists()
        assert json.loads(fallback_output.read_text(encoding="utf-8")) == json.loads(
            declared_path.read_text(encoding="utf-8")
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
