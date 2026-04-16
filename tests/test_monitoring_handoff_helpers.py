"""
@meta
type: test
scope: unit
domain: release-monitoring-orchestration
covers:
  - Shared monitoring handoff helper module boundaries
excludes:
  - Real Azure ML or Blob storage calls
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
    temp_dir = TEST_TEMP_ROOT / f"handoff-helpers-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_shared_handoff_helpers_resolve_release_targets() -> None:
    from src.monitoring.handoff_helpers import resolve_release_targets

    endpoint_name, deployment_name, model_name, model_version = resolve_release_targets(
        {
            "deployment": {
                "endpoint_name": "churn-endpoint",
                "deployment_name": "blue",
            },
            "registered_model": {
                "name": "churn-model",
                "version": "12",
            },
        }
    )

    assert endpoint_name == "churn-endpoint"
    assert deployment_name == "blue"
    assert model_name == "churn-model"
    assert model_version == "12"


def test_shared_handoff_helpers_collect_capture_paths_from_manifests() -> None:
    from src.monitoring.handoff_helpers import collect_capture_paths

    temp_dir = _make_temp_dir()
    try:
        manifest_a = temp_dir / "capture_manifest_a.json"
        manifest_b = temp_dir / "capture_manifest_b.json"
        _write_json(manifest_a, {"capture_path": "azureblob://account/container/a.jsonl"})
        _write_json(manifest_b, {"capture_path": "azureblob://account/container/b.jsonl"})

        capture_paths = collect_capture_paths([manifest_a, manifest_b])

        assert capture_paths == [
            "azureblob://account/container/a.jsonl",
            "azureblob://account/container/b.jsonl",
        ]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_runtime_owned_downloader_module_exposes_blob_parser() -> None:
    from src.monitoring.download_capture_blob import parse_capture_blob_path

    assert (
        parse_capture_blob_path(
            "azureblob://storage/container/monitoring/session/request-1.jsonl"
        )
        == "monitoring/session/request-1.jsonl"
    )
