"""
@meta
type: test
scope: unit
domain: retraining-path-selection
covers:
  - Post-validation retraining path selection
  - Fixed-train invocation through the selector
  - Prepared-only HPO handoff semantics
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
import uuid

import pytest


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"retraining-path-selection-{uuid.uuid4().hex}"
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


def _candidate_manifest(training_path_recommendation: str) -> dict[str, object]:
    return {
        "release_record_path": "release_record.json",
        "decision_source": {
            "path": "retraining_decision.json",
            "kind": "retraining_decision",
        },
        "trigger": "retraining_candidate",
        "reason_codes": ["prediction_class_balance_exceeded"],
        "training_path_recommendation": training_path_recommendation,
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


def _retraining_decision(training_path: str | None) -> dict[str, object]:
    return {
        "trigger": "retraining_candidate",
        "recommended_training_path": training_path,
        "policy_version": 1,
        "reason_codes": ["prediction_class_balance_exceeded"],
        "next_step": "freeze_retraining_candidate",
    }


def test_main_blocks_when_validation_not_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_path_selection

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "selection"
    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        candidate_manifest_path = temp_dir / "retraining_candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision("fixed_train"))
        _write_json(candidate_manifest_path, _candidate_manifest("fixed_train"))
        _write_json(validation_summary_path, {"status": "failed"})

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_path_selection.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_path_selection.main()

        summary = json.loads(
            (output_dir / "retraining_path_selection_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "blocked"
        assert summary["validation_status"] == "failed"
        assert summary["selected_path"] == "fixed_train"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_uses_candidate_manifest_recommendation_when_decision_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    @proves fixed-train.accept-path-selection-artifact
    """
    import run_retraining_path_selection

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "selection"
    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        candidate_manifest_path = temp_dir / "retraining_candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision(None))
        _write_json(candidate_manifest_path, _candidate_manifest("fixed_train"))
        _write_json(validation_summary_path, {"status": "passed"})

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_path_selection.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_path_selection.main()

        selection = json.loads(
            (output_dir / "retraining_path_selection.json").read_text(encoding="utf-8")
        )
        summary = json.loads(
            (output_dir / "retraining_path_selection_summary.json").read_text(encoding="utf-8")
        )
        assert selection["selected_path"] == "fixed_train"
        assert summary["status"] == "dry_run_ready"
        assert summary["selected_path"] == "fixed_train"
        assert summary["reason_codes"] == ["prediction_class_balance_exceeded"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_writes_dry_run_fixed_train_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_path_selection

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "selection"
    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        candidate_manifest_path = temp_dir / "retraining_candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision("fixed_train"))
        _write_json(candidate_manifest_path, _candidate_manifest("fixed_train"))
        _write_json(validation_summary_path, {"status": "passed"})

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_path_selection.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_path_selection.main()

        selection = json.loads(
            (output_dir / "retraining_path_selection.json").read_text(encoding="utf-8")
        )
        summary = json.loads(
            (output_dir / "retraining_path_selection_summary.json").read_text(encoding="utf-8")
        )

        assert selection["selected_path"] == "fixed_train"
        assert selection["downstream"]["entrypoint"] == "run_retraining_fixed_train_smoke.py"
        assert selection["invoke_selected_path"] is False
        assert summary["status"] == "dry_run_ready"
        assert summary["selected_path"] == "fixed_train"
        assert summary["downstream_invoked"] is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_invokes_fixed_train_handoff_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    @proves fixed-train.path-selection-invokes-bridges
    @proves online-deploy.provide-enough-release-monitor-provenance-later-post-validation
    @proves monitor.support-third-thin-bridge-selects-next-retraining-path
    """
    import run_retraining_path_selection

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "selection"
    calls: list[dict[str, object]] = []
    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        candidate_manifest_path = temp_dir / "retraining_candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision("fixed_train"))
        _write_json(candidate_manifest_path, _candidate_manifest("fixed_train"))
        _write_json(validation_summary_path, {"status": "passed"})

        def fake_fixed_train_main(argv: list[str]) -> Path:
            calls.append({"argv": argv})
            selected_path_output = output_dir / "selected-path"
            selected_path_output.mkdir(parents=True, exist_ok=True)
            summary_path = selected_path_output / "retraining_fixed_train_summary.json"
            _write_json(
                summary_path,
                {
                    "status": "dry_run_ready",
                    "validation_status": "passed",
                },
            )
            return summary_path

        monkeypatch.setattr(
            run_retraining_path_selection,
            "_invoke_fixed_train_handoff",
            fake_fixed_train_main,
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_path_selection.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--output-dir",
                str(output_dir),
                "--invoke-selected-path",
            ],
        )

        run_retraining_path_selection.main()

        summary = json.loads(
            (output_dir / "retraining_path_selection_summary.json").read_text(encoding="utf-8")
        )
        assert len(calls) == 1
        assert summary["status"] == "selected_and_invoked"
        assert summary["selected_path"] == "fixed_train"
        assert summary["downstream_invoked"] is True
        assert Path(str(summary["downstream_summary_path"])).as_posix().endswith(
            "selected-path/retraining_fixed_train_summary.json"
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_prepares_hpo_handoff_without_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    @proves fixed-train.path-selection-invokes-bridges
    @proves online-deploy.provide-enough-release-monitor-provenance-later-post-validation
    @proves monitor.support-third-thin-bridge-selects-next-retraining-path
    """
    import run_retraining_path_selection

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "selection"
    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        candidate_manifest_path = temp_dir / "retraining_candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision("model_sweep"))
        _write_json(candidate_manifest_path, _candidate_manifest("model_sweep"))
        _write_json(validation_summary_path, {"status": "passed"})

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_path_selection.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--output-dir",
                str(output_dir),
                "--hpo-config",
                "configs/hpo_smoke.yaml",
            ],
        )

        run_retraining_path_selection.main()

        selection = json.loads(
            (output_dir / "retraining_path_selection.json").read_text(encoding="utf-8")
        )
        summary = json.loads(
            (output_dir / "retraining_path_selection_summary.json").read_text(encoding="utf-8")
        )
        hpo_handoff = json.loads(
            (output_dir / "retraining_hpo_handoff.json").read_text(encoding="utf-8")
        )

        assert selection["selected_path"] == "model_sweep"
        assert selection["downstream"]["entrypoint"] == "run_hpo_pipeline.py"
        assert summary["status"] == "prepared_hpo_handoff"
        assert summary["downstream_invoked"] is False
        assert hpo_handoff["hpo_config_path"] == "configs/hpo_smoke.yaml"
        assert hpo_handoff["current_data"]["raw"].startswith("azureml://")
        assert not (output_dir / "selected_path_invocation.json").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
