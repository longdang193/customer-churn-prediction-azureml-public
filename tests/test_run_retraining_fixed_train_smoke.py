"""
@meta
type: test
scope: unit
domain: retraining-fixed-train-handoff
covers:
  - Validation-gated handoff from a retraining candidate into the fixed-train smoke path
  - Dry-run and submit-mode orchestration for the smoke fixed-train bridge
excludes:
  - Real Azure ML submission
  - Real deployment, release, or monitor execution
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import uuid

import pytest


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"retraining-fixed-train-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _release_record() -> dict[str, object]:
    return {
        "status": "succeeded",
        "deployment": {
            "endpoint_name": "churn-endpoint",
            "deployment_name": "blue",
        },
    }


def _candidate_manifest(current_data: str, reference_data: str) -> dict[str, object]:
    return {
        "release_record_path": "release_record.json",
        "decision_source": {
            "path": "retraining_decision.json",
            "kind": "retraining_decision",
        },
        "trigger": "retraining_candidate",
        "training_path_recommendation": "fixed_train",
        "current_data": {
            "raw": current_data,
            "normalized": current_data,
            "kind": "uri",
        },
        "reference_data": {
            "raw": reference_data,
            "normalized": reference_data,
            "kind": "azureml_asset",
        },
        "data_config_path": "configs/data_smoke_eval.yaml",
    }


def test_main_writes_dry_run_handoff_for_passed_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_fixed_train_smoke

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "handoff"
    try:
        release_record_path = temp_dir / "release_record.json"
        candidate_manifest_path = temp_dir / "retraining_candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(
            candidate_manifest_path,
            _candidate_manifest(
                "azureml://datastores/workspaceblobstore/paths/retraining/current/",
                "azureml:approved-reference:9",
            ),
        )
        _write_json(validation_summary_path, {"status": "passed"})

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_fixed_train_smoke.py",
                "--release-record",
                str(release_record_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_fixed_train_smoke.main()

        summary = json.loads(
            (output_dir / "retraining_fixed_train_summary.json").read_text(encoding="utf-8")
        )
        invocation = json.loads(
            (output_dir / "fixed_train_invocation.json").read_text(encoding="utf-8")
        )
        handoff = json.loads(
            (output_dir / "retraining_fixed_train_handoff.json").read_text(encoding="utf-8")
        )

        assert summary["status"] == "dry_run_ready"
        assert summary["validation_status"] == "passed"
        assert summary["submission"]["attempted"] is False
        assert invocation["arguments"]["train_config"] == "configs/train_smoke.yaml"
        assert invocation["arguments"]["current_data_override"].startswith("azureml://")
        assert invocation["arguments"]["reference_data_override"] == "azureml:approved-reference:9"
        assert "--current-data-override" in invocation["command"]
        assert "--reference-data-override" in invocation["command"]
        assert handoff["trigger"] == "retraining_candidate"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_blocks_when_validation_did_not_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_fixed_train_smoke

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "handoff"
    try:
        release_record_path = temp_dir / "release_record.json"
        candidate_manifest_path = temp_dir / "retraining_candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(
            candidate_manifest_path,
            _candidate_manifest(
                "azureml://datastores/workspaceblobstore/paths/retraining/current/",
                "azureml:approved-reference:9",
            ),
        )
        _write_json(validation_summary_path, {"status": "failed"})

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_fixed_train_smoke.py",
                "--release-record",
                str(release_record_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_fixed_train_smoke.main()

        summary = json.loads(
            (output_dir / "retraining_fixed_train_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "blocked_by_validation"
        assert summary["validation_status"] == "failed"
        assert not (output_dir / "fixed_train_invocation.json").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_submits_fixed_train_when_explicitly_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_fixed_train_smoke

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "handoff"
    commands: list[list[str]] = []
    try:
        release_record_path = temp_dir / "release_record.json"
        candidate_manifest_path = temp_dir / "retraining_candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(
            candidate_manifest_path,
            _candidate_manifest(
                "azureml://datastores/workspaceblobstore/paths/retraining/current/",
                "azureml:approved-reference:9",
            ),
        )
        _write_json(validation_summary_path, {"status": "passed"})

        def fake_run(
            command: list[str],
            *,
            check: bool,
            text: bool,
            capture_output: bool,
        ) -> subprocess.CompletedProcess[str]:
            assert check is True
            assert text is True
            assert capture_output is True
            commands.append(command)
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="OK Job submitted: train-job-123\n  View in Azure ML Studio: https://studio/job/123\n",
                stderr="",
            )

        monkeypatch.setattr(run_retraining_fixed_train_smoke.subprocess, "run", fake_run)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_fixed_train_smoke.py",
                "--release-record",
                str(release_record_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--output-dir",
                str(output_dir),
                "--submit",
            ],
        )

        run_retraining_fixed_train_smoke.main()

        summary = json.loads(
            (output_dir / "retraining_fixed_train_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "submitted"
        assert summary["submission"]["attempted"] is True
        assert summary["submission"]["job_name"] == "train-job-123"
        assert len(commands) == 1
        assert "--current-data-override" in commands[0]
        assert "--reference-data-override" in commands[0]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_writes_truthful_summary_when_submit_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_fixed_train_smoke

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "handoff"
    try:
        release_record_path = temp_dir / "release_record.json"
        candidate_manifest_path = temp_dir / "retraining_candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(
            candidate_manifest_path,
            _candidate_manifest(
                "azureml://datastores/workspaceblobstore/paths/retraining/current/",
                "azureml:approved-reference:9",
            ),
        )
        _write_json(validation_summary_path, {"status": "passed"})

        def fake_run(
            command: list[str],
            *,
            check: bool,
            text: bool,
            capture_output: bool,
        ) -> subprocess.CompletedProcess[str]:
            raise subprocess.CalledProcessError(
                1,
                command,
                output="",
                stderr="credential failure",
            )

        monkeypatch.setattr(run_retraining_fixed_train_smoke.subprocess, "run", fake_run)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_fixed_train_smoke.py",
                "--release-record",
                str(release_record_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--output-dir",
                str(output_dir),
                "--submit",
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_fixed_train_smoke.main()

        summary = json.loads(
            (output_dir / "retraining_fixed_train_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "submission_failed"
        assert summary["submission"]["attempted"] is True
        assert summary["submission"]["job_name"] is None
        assert summary["submission"]["stderr"] == "credential failure"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
