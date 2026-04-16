"""
@meta
type: test
scope: unit
domain: retraining-hpo-handoff
covers:
  - Validation-gated handoff from a retraining candidate into the HPO smoke path
  - Dry-run and submit-mode orchestration for the smoke HPO bridge
excludes:
  - Real Azure ML submission
  - Real HPO execution
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
    temp_dir = TEST_TEMP_ROOT / f"retraining-hpo-smoke-{uuid.uuid4().hex}"
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


def _candidate_manifest() -> dict[str, object]:
    return {
        "release_record_path": "release_record.json",
        "decision_source": {
            "path": "retraining_decision.json",
            "kind": "retraining_decision",
        },
        "trigger": "retraining_candidate",
        "training_path_recommendation": "model_sweep",
        "current_data": {
            "raw": "azureml://datastores/workspaceblobstore/paths/retraining/current/",
            "normalized": "azureml://datastores/workspaceblobstore/paths/retraining/current/",
            "kind": "uri",
        },
        "reference_data": {
            "raw": "azureml:approved-reference:9",
            "normalized": "azureml:approved-reference:9",
            "kind": "azureml_asset",
        },
        "data_config_path": "configs/data_smoke_eval.yaml",
    }


def _selection(selected_path: str) -> dict[str, object]:
    return {
        "release_record_path": "release_record.json",
        "decision_source": {
            "path": "retraining_decision.json",
            "kind": "retraining_decision",
        },
        "candidate_manifest_path": "retraining_candidate_manifest.json",
        "validation_summary_path": "validation_summary.json",
        "trigger": "retraining_candidate",
        "policy_version": 1,
        "reason_codes": ["prediction_class_balance_exceeded"],
        "selected_path": selected_path,
        "validation_status": "passed",
        "invoke_selected_path": False,
        "current_data": _candidate_manifest()["current_data"],
        "reference_data": _candidate_manifest()["reference_data"],
        "downstream": {
            "entrypoint": "run_hpo_pipeline.py" if selected_path == "model_sweep" else None,
            "data_config_path": "configs/data_smoke_eval.yaml",
            "train_config_path": None,
            "hpo_config_path": "configs/hpo_smoke.yaml" if selected_path == "model_sweep" else None,
            "summary_path": None,
        },
    }


def test_main_writes_dry_run_handoff_for_passed_model_sweep_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_hpo_smoke

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "handoff"
    try:
        release_record_path = temp_dir / "release_record.json"
        selection_path = temp_dir / "retraining_path_selection.json"
        candidate_manifest_path = temp_dir / "retraining_candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(selection_path, _selection("model_sweep"))
        _write_json(candidate_manifest_path, _candidate_manifest())
        _write_json(validation_summary_path, {"status": "passed"})

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_hpo_smoke.py",
                "--release-record",
                str(release_record_path),
                "--selection",
                str(selection_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_hpo_smoke.main()

        summary = json.loads(
            (output_dir / "retraining_hpo_smoke_summary.json").read_text(encoding="utf-8")
        )
        invocation = json.loads((output_dir / "hpo_invocation.json").read_text(encoding="utf-8"))
        handoff = json.loads(
            (output_dir / "retraining_hpo_smoke_handoff.json").read_text(encoding="utf-8")
        )

        assert summary["status"] == "dry_run_ready"
        assert summary["validation_status"] == "passed"
        assert summary["submission"]["attempted"] is False
        assert invocation["arguments"]["hpo_config"] == "configs/hpo_smoke.yaml"
        assert invocation["arguments"]["current_data_override"].startswith("azureml://")
        assert invocation["arguments"]["reference_data_override"] == "azureml:approved-reference:9"
        assert "--current-data-override" in invocation["command"]
        assert "--reference-data-override" in invocation["command"]
        assert handoff["selected_path"] == "model_sweep"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_blocks_when_validation_did_not_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_hpo_smoke

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "handoff"
    try:
        release_record_path = temp_dir / "release_record.json"
        selection_path = temp_dir / "retraining_path_selection.json"
        candidate_manifest_path = temp_dir / "retraining_candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(selection_path, _selection("model_sweep"))
        _write_json(candidate_manifest_path, _candidate_manifest())
        _write_json(validation_summary_path, {"status": "failed"})

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_hpo_smoke.py",
                "--release-record",
                str(release_record_path),
                "--selection",
                str(selection_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_hpo_smoke.main()

        summary = json.loads(
            (output_dir / "retraining_hpo_smoke_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "blocked_by_validation"
        assert summary["validation_status"] == "failed"
        assert not (output_dir / "hpo_invocation.json").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_blocks_when_selected_path_is_not_model_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_hpo_smoke

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "handoff"
    try:
        release_record_path = temp_dir / "release_record.json"
        selection_path = temp_dir / "retraining_path_selection.json"
        candidate_manifest_path = temp_dir / "retraining_candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(selection_path, _selection("fixed_train"))
        _write_json(candidate_manifest_path, _candidate_manifest())
        _write_json(validation_summary_path, {"status": "passed"})

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_hpo_smoke.py",
                "--release-record",
                str(release_record_path),
                "--selection",
                str(selection_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_hpo_smoke.main()

        summary = json.loads(
            (output_dir / "retraining_hpo_smoke_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "blocked_by_selection"
        assert summary["selected_path"] == "fixed_train"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_submits_hpo_when_explicitly_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_hpo_smoke

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "handoff"
    commands: list[list[str]] = []
    try:
        release_record_path = temp_dir / "release_record.json"
        selection_path = temp_dir / "retraining_path_selection.json"
        candidate_manifest_path = temp_dir / "retraining_candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(selection_path, _selection("model_sweep"))
        _write_json(candidate_manifest_path, _candidate_manifest())
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
                stdout="OK Submitted hpo-pipeline sweep: hpo-job-123\n  View in Azure ML Studio: https://studio/hpo/123\n",
                stderr="",
            )

        monkeypatch.setattr(run_retraining_hpo_smoke.subprocess, "run", fake_run)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_hpo_smoke.py",
                "--release-record",
                str(release_record_path),
                "--selection",
                str(selection_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--output-dir",
                str(output_dir),
                "--submit",
            ],
        )

        run_retraining_hpo_smoke.main()

        summary = json.loads(
            (output_dir / "retraining_hpo_smoke_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "submitted"
        assert summary["submission"]["attempted"] is True
        assert summary["submission"]["job_name"] == "hpo-job-123"
        assert len(commands) == 1
        assert "--current-data-override" in commands[0]
        assert "--reference-data-override" in commands[0]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_writes_truthful_summary_when_submit_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_hpo_smoke

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "handoff"
    try:
        release_record_path = temp_dir / "release_record.json"
        selection_path = temp_dir / "retraining_path_selection.json"
        candidate_manifest_path = temp_dir / "retraining_candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(selection_path, _selection("model_sweep"))
        _write_json(candidate_manifest_path, _candidate_manifest())
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

        monkeypatch.setattr(run_retraining_hpo_smoke.subprocess, "run", fake_run)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_hpo_smoke.py",
                "--release-record",
                str(release_record_path),
                "--selection",
                str(selection_path),
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
            run_retraining_hpo_smoke.main()

        summary = json.loads(
            (output_dir / "retraining_hpo_smoke_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "submission_failed"
        assert summary["submission"]["attempted"] is True
        assert summary["submission"]["job_name"] is None
        assert summary["submission"]["stderr"] == "credential failure"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
