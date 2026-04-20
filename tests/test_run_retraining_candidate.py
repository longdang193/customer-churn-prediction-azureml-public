"""
@meta
type: test
scope: unit
domain: retraining-handoff
covers:
  - Monitor decision to retraining candidate freeze bridge
  - Optional validate_data handoff reuse from the new candidate bridge
  - Root wrapper compatibility for the retraining candidate command
excludes:
  - Real Azure ML calls
  - Real Blob storage or remote dataset resolution
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import pytest


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"retraining-candidate-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _release_record() -> dict[str, object]:
    return {
        "status": "succeeded",
        "deployment": {
            "endpoint_name": "churn-endpoint",
            "deployment_name": "blue",
            "repo_owned_scoring_status": "repo_owned_scoring_proven",
        },
    }


def test_main_opens_candidate_manifest_for_retraining_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    @proves online-deploy.provide-enough-release-monitor-provenance-later-retraining-candidate
    @proves monitor.support-thin-post-monitor-bridge-freezes-explicit-current
    """
    import run_retraining_candidate

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "candidate-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        current_data = temp_dir / "current"
        reference_data = temp_dir / "reference"

        _write_json(release_record_path, _release_record())
        _write_json(
            decision_path,
            {
                "trigger": "retraining_candidate",
                "recommended_training_path": "fixed_train",
                "policy_version": 1,
                "reason_codes": ["prediction_class_balance_exceeded"],
                "recommendation_summary": {
                    "recommended_action": "Freeze the dataset window and validate before retraining.",
                    "next_step": "freeze_candidate_then_validate",
                },
            },
        )

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_candidate.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                str(current_data),
                "--reference-data",
                str(reference_data),
                "--data-config",
                "configs/data.yaml",
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_candidate.main()

        summary = json.loads((output_dir / "candidate_summary.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (output_dir / "retraining_candidate_manifest.json").read_text(encoding="utf-8")
        )
        handoff = json.loads((output_dir / "validation_handoff.json").read_text(encoding="utf-8"))

        assert summary["status"] == "candidate_opened"
        assert summary["trigger"] == "retraining_candidate"
        assert summary["recommendation_summary"]["recommended_action"].startswith(
            "Freeze the dataset window"
        )
        assert manifest["training_path_recommendation"] == "fixed_train"
        assert manifest["recommendation_summary"]["next_step"] == "freeze_candidate_then_validate"
        assert manifest["current_data"]["kind"] == "local_path"
        assert manifest["reference_data"]["kind"] == "local_path"
        assert handoff["run_validation"] is False
        assert handoff["current_data"]["raw"] == str(current_data)
        assert handoff["reference_data"]["raw"] == str(reference_data)
        assert handoff["command"][0] == "python"
        assert handoff["command"][1] == "src/validate_data.py"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_does_not_open_candidate_for_no_retraining_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_candidate

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "candidate-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        monitor_summary_path = temp_dir / "monitor_summary.json"
        current_data = "azureml:churn-current:5"
        reference_data = "azureml:churn-reference:3"

        _write_json(release_record_path, _release_record())
        _write_json(
            monitor_summary_path,
            {
                "monitor_status": "limited_but_healthy",
                "retraining_policy": {
                    "trigger": "no_retraining_signal",
                    "recommended_training_path": None,
                    "policy_version": 1,
                    "reason_codes": ["release_evidence_healthy"],
                },
            },
        )

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_candidate.py",
                "--release-record",
                str(release_record_path),
                "--monitor-summary",
                str(monitor_summary_path),
                "--current-data",
                current_data,
                "--reference-data",
                reference_data,
                "--data-config",
                "configs/data.yaml",
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_candidate.main()

        summary = json.loads((output_dir / "candidate_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "no_candidate_opened"
        assert summary["trigger"] == "no_retraining_signal"
        assert summary["recommendation_summary"] is None
        assert not (output_dir / "retraining_candidate_manifest.json").exists()
        assert not (output_dir / "validation_handoff.json").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_runs_validation_for_open_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_candidate

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "candidate-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        current_data = temp_dir / "current"
        reference_data = temp_dir / "reference"

        _write_json(release_record_path, _release_record())
        _write_json(
            decision_path,
            {
                "trigger": "retraining_candidate",
                "recommended_training_path": "fixed_train",
                "policy_version": 1,
                "reason_codes": ["prediction_class_balance_exceeded"],
                "recommendation_summary": {
                    "recommended_action": "Freeze the dataset window and validate before retraining.",
                    "next_step": "freeze_candidate_then_validate",
                },
            },
        )
        _write_csv(
            reference_data / "ref.csv",
            [
                "Age,Balance,Exited",
                "35,1000,0",
                "42,1500,1",
            ],
        )
        _write_csv(
            current_data / "cur.csv",
            [
                "Age,Balance,Exited",
                "36,1100,0",
                "43,1400,1",
            ],
        )

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_candidate.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                str(current_data),
                "--reference-data",
                str(reference_data),
                "--data-config",
                "configs/data.yaml",
                "--output-dir",
                str(output_dir),
                "--run-validation",
            ],
        )

        run_retraining_candidate.main()

        summary = json.loads((output_dir / "candidate_summary.json").read_text(encoding="utf-8"))
        handoff = json.loads((output_dir / "validation_handoff.json").read_text(encoding="utf-8"))
        validation_summary = json.loads(
            (output_dir / "validation" / "validation_summary.json").read_text(encoding="utf-8")
        )

        assert summary["status"] == "candidate_opened"
        assert summary["recommendation_summary"]["next_step"] == "freeze_candidate_then_validate"
        assert summary["validation"]["status"] == "passed"
        assert handoff["run_validation"] is True
        assert validation_summary["status"] == "passed"
        assert (output_dir / "validation" / "validation_manifest" / "step_manifest.json").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_writes_investigation_summary_without_opening_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_candidate

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "candidate-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"

        _write_json(release_record_path, _release_record())
        _write_json(
            decision_path,
            {
                "trigger": "investigate_before_retraining",
                "recommended_training_path": None,
                "policy_version": 1,
                "reason_codes": ["capture_expected_but_unretrieved"],
            },
        )

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_candidate.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                "azureml://datastores/workspaceblobstore/paths/retraining/current/",
                "--reference-data",
                "azureml:approved-reference:9",
                "--data-config",
                "configs/data.yaml",
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_candidate.main()

        summary = json.loads((output_dir / "candidate_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "investigation_required"
        assert summary["trigger"] == "investigate_before_retraining"
        assert summary["recommendation_summary"] is None
        assert not (output_dir / "retraining_candidate_manifest.json").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
