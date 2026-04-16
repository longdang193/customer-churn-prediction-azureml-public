"""
@meta
type: test
scope: unit
domain: retraining-loop
covers:
  - Phase-1 retraining loop stop modes
  - Truthful orchestration over candidate, path-selection, and selected-bridge surfaces
excludes:
  - Real Azure ML submission
  - Real training or HPO execution
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
    temp_dir = TEST_TEMP_ROOT / f"retraining-loop-{uuid.uuid4().hex}"
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


def _retraining_decision(trigger: str = "retraining_candidate") -> dict[str, object]:
    return {
        "trigger": trigger,
        "recommended_training_path": "fixed_train" if trigger == "retraining_candidate" else None,
        "policy_version": 1,
        "reason_codes": ["prediction_class_balance_exceeded"]
        if trigger == "retraining_candidate"
        else [trigger],
        "next_step": "freeze_retraining_candidate",
    }


def test_main_stops_after_candidate_for_freeze_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_loop

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "loop-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision())

        def fake_candidate(*, output_dir: Path, **_: object) -> dict[str, Path]:
            candidate_summary_path = output_dir / "candidate_summary.json"
            candidate_manifest_path = output_dir / "retraining_candidate_manifest.json"
            _write_json(candidate_summary_path, {"status": "candidate_opened"})
            _write_json(
                candidate_manifest_path,
                {"training_path_recommendation": "fixed_train"},
            )
            return {
                "summary_path": candidate_summary_path,
                "candidate_manifest_path": candidate_manifest_path,
                "validation_summary_path": output_dir / "validation" / "validation_summary.json",
            }

        monkeypatch.setattr(run_retraining_loop, "_invoke_candidate_bridge", fake_candidate)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_loop.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                "azureml:current:1",
                "--reference-data",
                "azureml:reference:1",
                "--data-config",
                "configs/data.yaml",
                "--mode",
                "freeze_only",
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_loop.main()

        summary = json.loads((output_dir / "retraining_loop_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "succeeded"
        assert summary["mode"] == "freeze_only"
        assert summary["final_stage"] == "candidate"
        assert summary["selected_bridge"] is None
        assert Path(str(summary["candidate"]["summary_path"])).name == "candidate_summary.json"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_stops_after_path_selection_for_validate_and_select_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_loop

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "loop-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision())

        def fake_candidate(*, output_dir: Path, **_: object) -> dict[str, Path]:
            candidate_summary_path = output_dir / "candidate_summary.json"
            candidate_manifest_path = output_dir / "retraining_candidate_manifest.json"
            validation_summary_path = output_dir / "validation" / "validation_summary.json"
            _write_json(candidate_summary_path, {"status": "candidate_opened"})
            _write_json(
                candidate_manifest_path,
                {"training_path_recommendation": "fixed_train"},
            )
            _write_json(validation_summary_path, {"status": "passed"})
            return {
                "summary_path": candidate_summary_path,
                "candidate_manifest_path": candidate_manifest_path,
                "validation_summary_path": validation_summary_path,
            }

        def fake_path_selection(*, output_dir: Path, **_: object) -> dict[str, Path]:
            selection_path = output_dir / "retraining_path_selection.json"
            selection_summary_path = output_dir / "retraining_path_selection_summary.json"
            _write_json(selection_path, {"selected_path": "fixed_train"})
            _write_json(
                selection_summary_path,
                {
                    "status": "dry_run_ready",
                    "selected_path": "fixed_train",
                },
            )
            return {
                "selection_path": selection_path,
                "summary_path": selection_summary_path,
                "selected_path": "fixed_train",
            }

        monkeypatch.setattr(run_retraining_loop, "_invoke_candidate_bridge", fake_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_path_selection_bridge", fake_path_selection)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_loop.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                "azureml:current:1",
                "--reference-data",
                "azureml:reference:1",
                "--data-config",
                "configs/data.yaml",
                "--mode",
                "validate_and_select_path",
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_loop.main()

        summary = json.loads((output_dir / "retraining_loop_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "succeeded"
        assert summary["final_stage"] == "path_selection"
        assert summary["path_selection"]["selected_path"] == "fixed_train"
        assert summary["selected_bridge"] is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_submits_fixed_train_selected_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_loop

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "loop-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision())

        def fake_candidate(*, output_dir: Path, **_: object) -> dict[str, Path]:
            candidate_summary_path = output_dir / "candidate_summary.json"
            candidate_manifest_path = output_dir / "retraining_candidate_manifest.json"
            validation_summary_path = output_dir / "validation" / "validation_summary.json"
            _write_json(candidate_summary_path, {"status": "candidate_opened"})
            _write_json(candidate_manifest_path, {"training_path_recommendation": "fixed_train"})
            _write_json(validation_summary_path, {"status": "passed"})
            return {
                "summary_path": candidate_summary_path,
                "candidate_manifest_path": candidate_manifest_path,
                "validation_summary_path": validation_summary_path,
            }

        def fake_path_selection(*, output_dir: Path, **_: object) -> dict[str, Path]:
            selection_path = output_dir / "retraining_path_selection.json"
            selection_summary_path = output_dir / "retraining_path_selection_summary.json"
            _write_json(selection_path, {"selected_path": "fixed_train"})
            _write_json(selection_summary_path, {"status": "dry_run_ready", "selected_path": "fixed_train"})
            return {
                "selection_path": selection_path,
                "summary_path": selection_summary_path,
                "selected_path": "fixed_train",
            }

        def fake_fixed_train(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            bridge_summary_path = output_dir / "retraining_fixed_train_summary.json"
            _write_json(
                bridge_summary_path,
                {
                    "status": "submitted",
                    "submission": {"job_name": "fixed-train-job"},
                },
            )
            return {
                "summary_path": bridge_summary_path,
                "job_name": "fixed-train-job",
            }

        monkeypatch.setattr(run_retraining_loop, "_invoke_candidate_bridge", fake_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_path_selection_bridge", fake_path_selection)
        monkeypatch.setattr(run_retraining_loop, "_invoke_fixed_train_bridge", fake_fixed_train)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_loop.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                "azureml:current:1",
                "--reference-data",
                "azureml:reference:1",
                "--data-config",
                "configs/data.yaml",
                "--mode",
                "submit_selected_path",
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_loop.main()

        summary = json.loads((output_dir / "retraining_loop_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "succeeded"
        assert summary["final_stage"] == "selected_bridge"
        assert summary["selected_bridge"] == "run_retraining_fixed_train_smoke.py"
        assert summary["submitted_job_name"] == "fixed-train-job"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_submits_hpo_selected_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_loop

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "loop-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision())

        def fake_candidate(*, output_dir: Path, **_: object) -> dict[str, Path]:
            candidate_summary_path = output_dir / "candidate_summary.json"
            candidate_manifest_path = output_dir / "retraining_candidate_manifest.json"
            validation_summary_path = output_dir / "validation" / "validation_summary.json"
            _write_json(candidate_summary_path, {"status": "candidate_opened"})
            _write_json(candidate_manifest_path, {"training_path_recommendation": "model_sweep"})
            _write_json(validation_summary_path, {"status": "passed"})
            return {
                "summary_path": candidate_summary_path,
                "candidate_manifest_path": candidate_manifest_path,
                "validation_summary_path": validation_summary_path,
            }

        def fake_path_selection(*, output_dir: Path, **_: object) -> dict[str, Path]:
            selection_path = output_dir / "retraining_path_selection.json"
            selection_summary_path = output_dir / "retraining_path_selection_summary.json"
            _write_json(selection_path, {"selected_path": "model_sweep"})
            _write_json(selection_summary_path, {"status": "prepared_hpo_handoff", "selected_path": "model_sweep"})
            return {
                "selection_path": selection_path,
                "summary_path": selection_summary_path,
                "selected_path": "model_sweep",
            }

        def fake_hpo(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            bridge_summary_path = output_dir / "retraining_hpo_smoke_summary.json"
            _write_json(
                bridge_summary_path,
                {
                    "status": "submitted",
                    "submission": {"job_name": "hpo-job"},
                },
            )
            return {
                "summary_path": bridge_summary_path,
                "job_name": "hpo-job",
            }

        monkeypatch.setattr(run_retraining_loop, "_invoke_candidate_bridge", fake_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_path_selection_bridge", fake_path_selection)
        monkeypatch.setattr(run_retraining_loop, "_invoke_hpo_bridge", fake_hpo)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_loop.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                "azureml:current:1",
                "--reference-data",
                "azureml:reference:1",
                "--data-config",
                "configs/data.yaml",
                "--mode",
                "submit_selected_path",
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_loop.main()

        summary = json.loads((output_dir / "retraining_loop_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "succeeded"
        assert summary["final_stage"] == "selected_bridge"
        assert summary["selected_bridge"] == "run_retraining_hpo_smoke.py"
        assert summary["submitted_job_name"] == "hpo-job"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_blocks_before_candidate_when_monitor_does_not_open_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_loop

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "loop-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision(trigger="no_retraining_signal"))

        def should_not_run(**_: object) -> dict[str, Path]:
            raise AssertionError("candidate bridge should not run for blocked monitor decisions")

        monkeypatch.setattr(run_retraining_loop, "_invoke_candidate_bridge", should_not_run)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_loop.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                "azureml:current:1",
                "--reference-data",
                "azureml:reference:1",
                "--data-config",
                "configs/data.yaml",
                "--mode",
                "submit_selected_path",
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_loop.main()

        summary = json.loads((output_dir / "retraining_loop_summary.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (output_dir / "retraining_loop_manifest" / "step_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert summary["status"] == "blocked"
        assert summary["final_stage"] == "monitor_gate"
        assert summary["candidate"] is None
        assert (
            manifest["outputs"]["retraining_loop_report"]["path"].endswith(
                "retraining_loop_report.md"
            )
        )
        assert manifest["inputs"]["current_data"]["value"] == "azureml:current:1"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_continues_to_release_after_fixed_train_when_promoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_loop

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "loop-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision())

        def fake_candidate(*, output_dir: Path, **_: object) -> dict[str, Path]:
            candidate_summary_path = output_dir / "candidate_summary.json"
            candidate_manifest_path = output_dir / "retraining_candidate_manifest.json"
            validation_summary_path = output_dir / "validation" / "validation_summary.json"
            _write_json(candidate_summary_path, {"status": "candidate_opened"})
            _write_json(candidate_manifest_path, {"training_path_recommendation": "fixed_train"})
            _write_json(validation_summary_path, {"status": "passed"})
            return {
                "summary_path": candidate_summary_path,
                "candidate_manifest_path": candidate_manifest_path,
                "validation_summary_path": validation_summary_path,
            }

        def fake_path_selection(*, output_dir: Path, **_: object) -> dict[str, Path]:
            selection_path = output_dir / "retraining_path_selection.json"
            selection_summary_path = output_dir / "retraining_path_selection_summary.json"
            _write_json(selection_path, {"selected_path": "fixed_train"})
            _write_json(selection_summary_path, {"status": "dry_run_ready", "selected_path": "fixed_train"})
            return {
                "selection_path": selection_path,
                "summary_path": selection_summary_path,
                "selected_path": "fixed_train",
            }

        def fake_fixed_train(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            bridge_summary_path = output_dir / "retraining_fixed_train_summary.json"
            _write_json(
                bridge_summary_path,
                {
                    "status": "submitted",
                    "submission": {"job_name": "fixed-train-job"},
                },
            )
            return {
                "summary_path": bridge_summary_path,
                "status": "submitted",
                "job_name": "fixed-train-job",
            }

        def fake_release_candidate(*, selected_path: str, submitted_job_name: str | None, **_: object) -> dict[str, object]:
            assert selected_path == "fixed_train"
            assert submitted_job_name == "fixed-train-job"
            return {
                "eligible": True,
                "promotion_status": "promote",
                "job_name": "fixed-train-job",
                "release_train_config_path": "configs/train_smoke.yaml",
            }

        def fake_release(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            release_dir = output_dir / "release" / "fixed-train-job"
            release_record_out = release_dir / "release_record.json"
            _write_json(release_record_out, {"status": "succeeded"})
            return {
                "status": "succeeded",
                "release_record_path": release_record_out,
            }

        monkeypatch.setattr(run_retraining_loop, "_invoke_candidate_bridge", fake_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_path_selection_bridge", fake_path_selection)
        monkeypatch.setattr(run_retraining_loop, "_invoke_fixed_train_bridge", fake_fixed_train)
        monkeypatch.setattr(run_retraining_loop, "_resolve_release_candidate", fake_release_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_release_bridge", fake_release)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_loop.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                "azureml:current:1",
                "--reference-data",
                "azureml:reference:1",
                "--data-config",
                "configs/data.yaml",
                "--mode",
                "submit_selected_path",
                "--release-mode",
                "after_promotion",
                "--release-config",
                "config.env",
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_loop.main()

        summary = json.loads((output_dir / "retraining_loop_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "succeeded"
        assert summary["final_stage"] == "release"
        assert summary["release_mode"] == "after_promotion"
        assert summary["release_attempted"] is True
        assert summary["release_status"] == "succeeded"
        assert summary["release_candidate_job_name"] == "fixed-train-job"
        assert str(summary["continued_release_record_path"]).endswith("release_record.json")
        assert summary["monitor_handoff_attempted"] is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_wait_for_release_candidate_job_polls_until_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_loop

    observed: list[str] = []
    statuses = iter(["Running", "Completed"])

    class FakeJobs:
        def get(self, job_name: str):
            observed.append(job_name)
            return type("FakeJob", (), {"status": next(statuses)})()

    fake_client = type("FakeClient", (), {"jobs": FakeJobs()})()
    monkeypatch.setattr(run_retraining_loop.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(run_retraining_loop.time, "monotonic", lambda: 0.0)

    status = run_retraining_loop._wait_for_release_candidate_job(
        ml_client=fake_client,
        job_name="fixed-train-job",
        timeout_seconds=60,
        poll_interval_seconds=0,
    )

    assert status == "completed"
    assert observed == ["fixed-train-job", "fixed-train-job"]


def test_main_blocks_at_promotion_gate_before_release(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_loop

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "loop-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision())

        def fake_candidate(*, output_dir: Path, **_: object) -> dict[str, Path]:
            candidate_summary_path = output_dir / "candidate_summary.json"
            candidate_manifest_path = output_dir / "retraining_candidate_manifest.json"
            validation_summary_path = output_dir / "validation" / "validation_summary.json"
            _write_json(candidate_summary_path, {"status": "candidate_opened"})
            _write_json(candidate_manifest_path, {"training_path_recommendation": "fixed_train"})
            _write_json(validation_summary_path, {"status": "passed"})
            return {
                "summary_path": candidate_summary_path,
                "candidate_manifest_path": candidate_manifest_path,
                "validation_summary_path": validation_summary_path,
            }

        def fake_path_selection(*, output_dir: Path, **_: object) -> dict[str, Path]:
            selection_path = output_dir / "retraining_path_selection.json"
            selection_summary_path = output_dir / "retraining_path_selection_summary.json"
            _write_json(selection_path, {"selected_path": "fixed_train"})
            _write_json(selection_summary_path, {"status": "dry_run_ready", "selected_path": "fixed_train"})
            return {
                "selection_path": selection_path,
                "summary_path": selection_summary_path,
                "selected_path": "fixed_train",
            }

        def fake_fixed_train(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            bridge_summary_path = output_dir / "retraining_fixed_train_summary.json"
            _write_json(
                bridge_summary_path,
                {
                    "status": "submitted",
                    "submission": {"job_name": "fixed-train-job"},
                },
            )
            return {
                "summary_path": bridge_summary_path,
                "status": "submitted",
                "job_name": "fixed-train-job",
            }

        def fake_release_candidate(**_: object) -> dict[str, object]:
            return {
                "eligible": False,
                "promotion_status": "hold",
                "job_name": "fixed-train-job",
                "release_train_config_path": "configs/train_smoke.yaml",
            }

        def should_not_release(**_: object) -> dict[str, Path | str | None]:
            raise AssertionError("release bridge should not run when promotion is not promotable")

        monkeypatch.setattr(run_retraining_loop, "_invoke_candidate_bridge", fake_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_path_selection_bridge", fake_path_selection)
        monkeypatch.setattr(run_retraining_loop, "_invoke_fixed_train_bridge", fake_fixed_train)
        monkeypatch.setattr(run_retraining_loop, "_resolve_release_candidate", fake_release_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_release_bridge", should_not_release)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_loop.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                "azureml:current:1",
                "--reference-data",
                "azureml:reference:1",
                "--data-config",
                "configs/data.yaml",
                "--mode",
                "submit_selected_path",
                "--release-mode",
                "after_promotion",
                "--release-config",
                "config.env",
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_loop.main()

        summary = json.loads((output_dir / "retraining_loop_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "blocked"
        assert summary["final_stage"] == "promotion_gate"
        assert summary["release_attempted"] is False
        assert summary["promotion_status"] == "hold"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_blocks_at_validation_before_path_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_loop

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "loop-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision())

        def fake_candidate(*, output_dir: Path, **_: object) -> dict[str, Path]:
            candidate_summary_path = output_dir / "candidate_summary.json"
            candidate_manifest_path = output_dir / "retraining_candidate_manifest.json"
            validation_summary_path = output_dir / "validation" / "validation_summary.json"
            _write_json(candidate_summary_path, {"status": "candidate_opened"})
            _write_json(candidate_manifest_path, {"training_path_recommendation": "fixed_train"})
            _write_json(validation_summary_path, {"status": "failed"})
            return {
                "summary_path": candidate_summary_path,
                "candidate_manifest_path": candidate_manifest_path,
                "validation_summary_path": validation_summary_path,
            }

        def should_not_select_path(**_: object) -> dict[str, Path]:
            raise AssertionError("path selection should not run when validation did not pass")

        monkeypatch.setattr(run_retraining_loop, "_invoke_candidate_bridge", fake_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_path_selection_bridge", should_not_select_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_loop.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                "azureml:current:1",
                "--reference-data",
                "azureml:reference:1",
                "--data-config",
                "configs/data.yaml",
                "--mode",
                "submit_selected_path",
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_loop.main()

        summary = json.loads((output_dir / "retraining_loop_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "blocked"
        assert summary["final_stage"] == "validation"
        assert summary["path_selection"] is None
        assert summary["selected_bridge"] is None
        assert summary["submitted_job_name"] is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_continues_to_monitor_handoff_after_release(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_loop

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "loop-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision())

        def fake_candidate(*, output_dir: Path, **_: object) -> dict[str, Path]:
            candidate_summary_path = output_dir / "candidate_summary.json"
            candidate_manifest_path = output_dir / "retraining_candidate_manifest.json"
            validation_summary_path = output_dir / "validation" / "validation_summary.json"
            _write_json(candidate_summary_path, {"status": "candidate_opened"})
            _write_json(candidate_manifest_path, {"training_path_recommendation": "fixed_train"})
            _write_json(validation_summary_path, {"status": "passed"})
            return {
                "summary_path": candidate_summary_path,
                "candidate_manifest_path": candidate_manifest_path,
                "validation_summary_path": validation_summary_path,
            }

        def fake_path_selection(*, output_dir: Path, **_: object) -> dict[str, Path]:
            selection_path = output_dir / "retraining_path_selection.json"
            selection_summary_path = output_dir / "retraining_path_selection_summary.json"
            _write_json(selection_path, {"selected_path": "fixed_train"})
            _write_json(selection_summary_path, {"status": "dry_run_ready", "selected_path": "fixed_train"})
            return {
                "selection_path": selection_path,
                "summary_path": selection_summary_path,
                "selected_path": "fixed_train",
            }

        def fake_fixed_train(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            bridge_summary_path = output_dir / "retraining_fixed_train_summary.json"
            _write_json(
                bridge_summary_path,
                {
                    "status": "submitted",
                    "submission": {"job_name": "fixed-train-job"},
                },
            )
            return {
                "summary_path": bridge_summary_path,
                "status": "submitted",
                "job_name": "fixed-train-job",
            }

        def fake_release_candidate(**_: object) -> dict[str, object]:
            return {
                "eligible": True,
                "promotion_status": "promote",
                "job_name": "fixed-train-job",
                "release_train_config_path": "configs/train_smoke.yaml",
            }

        def fake_release(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            release_dir = output_dir / "release" / "fixed-train-job"
            release_record_out = release_dir / "release_record.json"
            _write_json(release_record_out, {"status": "succeeded"})
            return {
                "status": "succeeded",
                "release_record_path": release_record_out,
            }

        def fake_monitor_handoff(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            handoff_dir = output_dir / "monitor-handoff"
            handoff_summary_path = handoff_dir / "handoff_summary.json"
            _write_json(
                handoff_summary_path,
                {
                    "status": "succeeded",
                    "handoff": {"status": "capture_backed_monitoring_ready"},
                },
            )
            return {
                "status": "capture_backed_monitoring_ready",
                "summary_path": handoff_summary_path,
            }

        monkeypatch.setattr(run_retraining_loop, "_invoke_candidate_bridge", fake_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_path_selection_bridge", fake_path_selection)
        monkeypatch.setattr(run_retraining_loop, "_invoke_fixed_train_bridge", fake_fixed_train)
        monkeypatch.setattr(run_retraining_loop, "_resolve_release_candidate", fake_release_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_release_bridge", fake_release)
        monkeypatch.setattr(run_retraining_loop, "_invoke_monitor_handoff_bridge", fake_monitor_handoff)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_loop.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                "azureml:current:1",
                "--reference-data",
                "azureml:reference:1",
                "--data-config",
                "configs/data.yaml",
                "--mode",
                "submit_selected_path",
                "--release-mode",
                "after_release_monitor_handoff",
                "--release-config",
                "config.env",
                "--probe-request",
                "sample-data.json",
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_loop.main()

        summary = json.loads((output_dir / "retraining_loop_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "succeeded"
        assert summary["final_stage"] == "monitor_handoff"
        assert summary["release_attempted"] is True
        assert summary["monitor_handoff_attempted"] is True
        assert summary["monitor_handoff_status"] == "capture_backed_monitoring_ready"
        assert str(summary["monitor_handoff_summary_path"]).endswith("handoff_summary.json")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_stops_at_release_when_release_continuation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_loop

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "loop-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision())

        def fake_candidate(*, output_dir: Path, **_: object) -> dict[str, Path]:
            candidate_summary_path = output_dir / "candidate_summary.json"
            candidate_manifest_path = output_dir / "retraining_candidate_manifest.json"
            validation_summary_path = output_dir / "validation" / "validation_summary.json"
            _write_json(candidate_summary_path, {"status": "candidate_opened"})
            _write_json(candidate_manifest_path, {"training_path_recommendation": "fixed_train"})
            _write_json(validation_summary_path, {"status": "passed"})
            return {
                "summary_path": candidate_summary_path,
                "candidate_manifest_path": candidate_manifest_path,
                "validation_summary_path": validation_summary_path,
            }

        def fake_path_selection(*, output_dir: Path, **_: object) -> dict[str, Path]:
            selection_path = output_dir / "retraining_path_selection.json"
            selection_summary_path = output_dir / "retraining_path_selection_summary.json"
            _write_json(selection_path, {"selected_path": "fixed_train"})
            _write_json(selection_summary_path, {"status": "dry_run_ready", "selected_path": "fixed_train"})
            return {
                "selection_path": selection_path,
                "summary_path": selection_summary_path,
                "selected_path": "fixed_train",
            }

        def fake_fixed_train(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            bridge_summary_path = output_dir / "retraining_fixed_train_summary.json"
            _write_json(
                bridge_summary_path,
                {
                    "status": "submitted",
                    "submission": {"job_name": "fixed-train-job"},
                },
            )
            return {
                "summary_path": bridge_summary_path,
                "status": "submitted",
                "job_name": "fixed-train-job",
            }

        def fake_release_candidate(**_: object) -> dict[str, object]:
            return {
                "eligible": True,
                "promotion_status": "promote",
                "job_name": "fixed-train-job",
                "release_train_config_path": "configs/train_smoke.yaml",
            }

        def fake_release(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            release_dir = output_dir / "release" / "fixed-train-job"
            release_record_out = release_dir / "release_record.json"
            _write_json(release_record_out, {"status": "failed"})
            return {
                "status": "failed",
                "release_record_path": release_record_out,
            }

        def should_not_handoff(**_: object) -> dict[str, Path | str | None]:
            raise AssertionError("monitor handoff should not run when release failed")

        monkeypatch.setattr(run_retraining_loop, "_invoke_candidate_bridge", fake_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_path_selection_bridge", fake_path_selection)
        monkeypatch.setattr(run_retraining_loop, "_invoke_fixed_train_bridge", fake_fixed_train)
        monkeypatch.setattr(run_retraining_loop, "_resolve_release_candidate", fake_release_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_release_bridge", fake_release)
        monkeypatch.setattr(run_retraining_loop, "_invoke_monitor_handoff_bridge", should_not_handoff)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_loop.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                "azureml:current:1",
                "--reference-data",
                "azureml:reference:1",
                "--data-config",
                "configs/data.yaml",
                "--mode",
                "submit_selected_path",
                "--release-mode",
                "after_release_monitor_handoff",
                "--release-config",
                "config.env",
                "--probe-request",
                "sample-data.json",
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_loop.main()

        summary = json.loads((output_dir / "retraining_loop_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "failed"
        assert summary["final_stage"] == "release"
        assert summary["release_attempted"] is True
        assert summary["release_status"] == "failed"
        assert summary["monitor_handoff_attempted"] is False
        assert str(summary["continued_release_record_path"]).endswith("release_record.json")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_monitor_handoff_failure_preserves_release_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_loop

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "loop-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision())

        def fake_candidate(*, output_dir: Path, **_: object) -> dict[str, Path]:
            candidate_summary_path = output_dir / "candidate_summary.json"
            candidate_manifest_path = output_dir / "retraining_candidate_manifest.json"
            validation_summary_path = output_dir / "validation" / "validation_summary.json"
            _write_json(candidate_summary_path, {"status": "candidate_opened"})
            _write_json(candidate_manifest_path, {"training_path_recommendation": "fixed_train"})
            _write_json(validation_summary_path, {"status": "passed"})
            return {
                "summary_path": candidate_summary_path,
                "candidate_manifest_path": candidate_manifest_path,
                "validation_summary_path": validation_summary_path,
            }

        def fake_path_selection(*, output_dir: Path, **_: object) -> dict[str, Path]:
            selection_path = output_dir / "retraining_path_selection.json"
            selection_summary_path = output_dir / "retraining_path_selection_summary.json"
            _write_json(selection_path, {"selected_path": "fixed_train"})
            _write_json(selection_summary_path, {"status": "dry_run_ready", "selected_path": "fixed_train"})
            return {
                "selection_path": selection_path,
                "summary_path": selection_summary_path,
                "selected_path": "fixed_train",
            }

        def fake_fixed_train(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            bridge_summary_path = output_dir / "retraining_fixed_train_summary.json"
            _write_json(
                bridge_summary_path,
                {
                    "status": "submitted",
                    "submission": {"job_name": "fixed-train-job"},
                },
            )
            return {
                "summary_path": bridge_summary_path,
                "status": "submitted",
                "job_name": "fixed-train-job",
            }

        def fake_release_candidate(**_: object) -> dict[str, object]:
            return {
                "eligible": True,
                "promotion_status": "promote",
                "job_name": "fixed-train-job",
                "release_train_config_path": "configs/train_smoke.yaml",
            }

        def fake_release(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            release_dir = output_dir / "release" / "fixed-train-job"
            release_record_out = release_dir / "release_record.json"
            _write_json(release_record_out, {"status": "succeeded"})
            return {
                "status": "succeeded",
                "release_record_path": release_record_out,
            }

        def fake_monitor_handoff(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            handoff_dir = output_dir / "monitor-handoff"
            handoff_summary_path = handoff_dir / "handoff_summary.json"
            _write_json(
                handoff_summary_path,
                {
                    "status": "failed",
                    "handoff": {"status": "blocked"},
                },
            )
            return {
                "status": "failed",
                "summary_path": handoff_summary_path,
            }

        monkeypatch.setattr(run_retraining_loop, "_invoke_candidate_bridge", fake_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_path_selection_bridge", fake_path_selection)
        monkeypatch.setattr(run_retraining_loop, "_invoke_fixed_train_bridge", fake_fixed_train)
        monkeypatch.setattr(run_retraining_loop, "_resolve_release_candidate", fake_release_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_release_bridge", fake_release)
        monkeypatch.setattr(run_retraining_loop, "_invoke_monitor_handoff_bridge", fake_monitor_handoff)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_loop.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                "azureml:current:1",
                "--reference-data",
                "azureml:reference:1",
                "--data-config",
                "configs/data.yaml",
                "--mode",
                "submit_selected_path",
                "--release-mode",
                "after_release_monitor_handoff",
                "--release-config",
                "config.env",
                "--probe-request",
                "sample-data.json",
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_loop.main()

        summary = json.loads((output_dir / "retraining_loop_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "failed"
        assert summary["final_stage"] == "monitor_handoff"
        assert summary["release_attempted"] is True
        assert summary["release_status"] == "succeeded"
        assert summary["monitor_handoff_attempted"] is True
        assert summary["monitor_handoff_status"] == "failed"
        assert str(summary["continued_release_record_path"]).endswith("release_record.json")
        assert str(summary["monitor_handoff_summary_path"]).endswith("handoff_summary.json")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_model_sweep_release_continues_via_hpo_to_fixed_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_loop

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "loop-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision())

        def fake_candidate(*, output_dir: Path, **_: object) -> dict[str, Path]:
            candidate_summary_path = output_dir / "candidate_summary.json"
            candidate_manifest_path = output_dir / "retraining_candidate_manifest.json"
            validation_summary_path = output_dir / "validation" / "validation_summary.json"
            _write_json(candidate_summary_path, {"status": "candidate_opened"})
            _write_json(candidate_manifest_path, {"training_path_recommendation": "model_sweep"})
            _write_json(validation_summary_path, {"status": "passed"})
            return {
                "summary_path": candidate_summary_path,
                "candidate_manifest_path": candidate_manifest_path,
                "validation_summary_path": validation_summary_path,
            }

        def fake_path_selection(*, output_dir: Path, **_: object) -> dict[str, Path]:
            selection_path = output_dir / "retraining_path_selection.json"
            selection_summary_path = output_dir / "retraining_path_selection_summary.json"
            _write_json(selection_path, {"selected_path": "model_sweep"})
            _write_json(selection_summary_path, {"status": "prepared_hpo_handoff", "selected_path": "model_sweep"})
            return {
                "selection_path": selection_path,
                "summary_path": selection_summary_path,
                "selected_path": "model_sweep",
            }

        def fake_hpo(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            bridge_summary_path = output_dir / "retraining_hpo_smoke_summary.json"
            _write_json(
                bridge_summary_path,
                {
                    "status": "submitted",
                    "submission": {"job_name": "hpo-job"},
                },
            )
            return {
                "summary_path": bridge_summary_path,
                "status": "submitted",
                "job_name": "hpo-job",
            }

        def fake_hpo_to_fixed(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            summary_path = output_dir / "retraining_hpo_to_fixed_train_summary.json"
            _write_json(
                summary_path,
                {
                    "status": "submitted",
                    "submitted_job_name": "continued-fixed-job",
                    "exported_train_config_path": "configs/train_hpo_winner.yaml",
                },
            )
            return {
                "summary_path": summary_path,
                "status": "submitted",
                "job_name": "continued-fixed-job",
                "train_config_path": "configs/train_hpo_winner.yaml",
            }

        def fake_release_candidate(*, selected_path: str, submitted_job_name: str | None, **_: object) -> dict[str, object]:
            assert selected_path == "model_sweep"
            assert submitted_job_name == "continued-fixed-job"
            return {
                "eligible": True,
                "promotion_status": "promote",
                "job_name": "continued-fixed-job",
                "release_train_config_path": "configs/train_hpo_winner.yaml",
            }

        def fake_release(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            release_dir = output_dir / "release" / "continued-fixed-job"
            release_record_out = release_dir / "release_record.json"
            _write_json(release_record_out, {"status": "succeeded"})
            return {
                "status": "succeeded",
                "release_record_path": release_record_out,
            }

        monkeypatch.setattr(run_retraining_loop, "_invoke_candidate_bridge", fake_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_path_selection_bridge", fake_path_selection)
        monkeypatch.setattr(run_retraining_loop, "_invoke_hpo_bridge", fake_hpo)
        monkeypatch.setattr(run_retraining_loop, "_invoke_hpo_to_fixed_train_bridge", fake_hpo_to_fixed)
        monkeypatch.setattr(run_retraining_loop, "_resolve_release_candidate", fake_release_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_release_bridge", fake_release)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_loop.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                "azureml:current:1",
                "--reference-data",
                "azureml:reference:1",
                "--data-config",
                "configs/data.yaml",
                "--mode",
                "submit_selected_path",
                "--release-mode",
                "after_promotion",
                "--release-config",
                "config.env",
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_loop.main()

        summary = json.loads((output_dir / "retraining_loop_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "succeeded"
        assert summary["selected_bridge"] == "run_retraining_hpo_smoke.py"
        assert summary["continuation_bridge"] == "run_retraining_hpo_to_fixed_train.py"
        assert summary["release_candidate_job_name"] == "continued-fixed-job"
        assert summary["release_status"] == "succeeded"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_model_sweep_resume_continuation_reaches_monitor_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_loop

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "loop-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        resume_summary_path = temp_dir / "retraining_hpo_to_fixed_train_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision())
        _write_json(
            resume_summary_path,
            {
                "status": "submitted",
                "selected_path": "model_sweep",
                "submitted_job_name": "resumed-fixed-job",
                "exported_train_config_path": "configs/train_hpo_winner.yaml",
                "hpo_smoke_summary_path": str(temp_dir / "retraining_hpo_smoke_summary.json"),
            },
        )

        def fake_candidate(*, output_dir: Path, **_: object) -> dict[str, Path]:
            candidate_summary_path = output_dir / "candidate_summary.json"
            candidate_manifest_path = output_dir / "retraining_candidate_manifest.json"
            validation_summary_path = output_dir / "validation" / "validation_summary.json"
            _write_json(candidate_summary_path, {"status": "candidate_opened"})
            _write_json(candidate_manifest_path, {"training_path_recommendation": "model_sweep"})
            _write_json(validation_summary_path, {"status": "passed"})
            return {
                "summary_path": candidate_summary_path,
                "candidate_manifest_path": candidate_manifest_path,
                "validation_summary_path": validation_summary_path,
            }

        def fake_path_selection(*, output_dir: Path, **_: object) -> dict[str, Path]:
            selection_path = output_dir / "retraining_path_selection.json"
            selection_summary_path = output_dir / "retraining_path_selection_summary.json"
            _write_json(selection_path, {"selected_path": "model_sweep"})
            _write_json(
                selection_summary_path,
                {"status": "prepared_hpo_handoff", "selected_path": "model_sweep"},
            )
            return {
                "selection_path": selection_path,
                "summary_path": selection_summary_path,
                "selected_path": "model_sweep",
            }

        def fail_hpo_bridge(**_: object) -> dict[str, object]:
            raise AssertionError("HPO bridge should not run when resuming continuation evidence")

        def fail_hpo_to_fixed(**_: object) -> dict[str, object]:
            raise AssertionError(
                "HPO continuation bridge should not run when resuming continuation evidence"
            )

        def fake_release_candidate(*, selected_path: str, submitted_job_name: str | None, **_: object) -> dict[str, object]:
            assert selected_path == "model_sweep"
            assert submitted_job_name == "resumed-fixed-job"
            return {
                "eligible": True,
                "promotion_status": "promote",
                "job_name": "resumed-fixed-job",
                "release_train_config_path": "configs/train_hpo_winner.yaml",
            }

        def fake_release(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            release_dir = output_dir / "release" / "resumed-fixed-job"
            release_record_out = release_dir / "release_record.json"
            _write_json(release_record_out, {"status": "succeeded"})
            return {
                "status": "succeeded",
                "release_record_path": release_record_out,
            }

        def fake_monitor_handoff(*, output_dir: Path, **_: object) -> dict[str, Path | str | None]:
            handoff_dir = output_dir / "monitor-handoff"
            handoff_summary_path = handoff_dir / "handoff_summary.json"
            _write_json(
                handoff_summary_path,
                {
                    "status": "capture_backed_monitoring_ready",
                    "handoff": {"status": "capture_backed_monitoring_ready"},
                },
            )
            return {
                "status": "capture_backed_monitoring_ready",
                "summary_path": handoff_summary_path,
            }

        monkeypatch.setattr(run_retraining_loop, "_invoke_candidate_bridge", fake_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_path_selection_bridge", fake_path_selection)
        monkeypatch.setattr(run_retraining_loop, "_invoke_hpo_bridge", fail_hpo_bridge)
        monkeypatch.setattr(run_retraining_loop, "_invoke_hpo_to_fixed_train_bridge", fail_hpo_to_fixed)
        monkeypatch.setattr(run_retraining_loop, "_resolve_release_candidate", fake_release_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_release_bridge", fake_release)
        monkeypatch.setattr(run_retraining_loop, "_invoke_monitor_handoff_bridge", fake_monitor_handoff)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_loop.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                "azureml:current:1",
                "--reference-data",
                "azureml:reference:1",
                "--data-config",
                "configs/data.yaml",
                "--mode",
                "submit_selected_path",
                "--release-mode",
                "after_release_monitor_handoff",
                "--release-config",
                "config.env",
                "--resume-continuation-summary",
                str(resume_summary_path),
                "--probe-request",
                "sample-data.json",
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_loop.main()

        summary = json.loads((output_dir / "retraining_loop_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "succeeded"
        assert summary["final_stage"] == "monitor_handoff"
        assert summary["selected_bridge"] == "run_retraining_hpo_smoke.py"
        assert summary["selected_bridge_status"] == "resumed"
        assert summary["continuation_bridge"] == "run_retraining_hpo_to_fixed_train.py"
        assert summary["continuation_bridge_status"] == "resumed"
        assert summary["release_candidate_job_name"] == "resumed-fixed-job"
        assert summary["monitor_handoff_status"] == "capture_backed_monitoring_ready"
        assert summary["resumed_from_continuation_summary_path"] == str(resume_summary_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_resume_continuation_blocks_when_summary_is_not_submitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_loop

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "loop-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        decision_path = temp_dir / "retraining_decision.json"
        resume_summary_path = temp_dir / "retraining_hpo_to_fixed_train_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(decision_path, _retraining_decision())
        _write_json(
            resume_summary_path,
            {
                "status": "blocked_by_winner_inconsistency",
                "selected_path": "model_sweep",
                "submitted_job_name": "resumed-fixed-job",
                "exported_train_config_path": "configs/train_hpo_winner.yaml",
            },
        )

        def fake_candidate(*, output_dir: Path, **_: object) -> dict[str, Path]:
            candidate_summary_path = output_dir / "candidate_summary.json"
            candidate_manifest_path = output_dir / "retraining_candidate_manifest.json"
            validation_summary_path = output_dir / "validation" / "validation_summary.json"
            _write_json(candidate_summary_path, {"status": "candidate_opened"})
            _write_json(candidate_manifest_path, {"training_path_recommendation": "model_sweep"})
            _write_json(validation_summary_path, {"status": "passed"})
            return {
                "summary_path": candidate_summary_path,
                "candidate_manifest_path": candidate_manifest_path,
                "validation_summary_path": validation_summary_path,
            }

        def fake_path_selection(*, output_dir: Path, **_: object) -> dict[str, Path]:
            selection_path = output_dir / "retraining_path_selection.json"
            selection_summary_path = output_dir / "retraining_path_selection_summary.json"
            _write_json(selection_path, {"selected_path": "model_sweep"})
            _write_json(
                selection_summary_path,
                {"status": "prepared_hpo_handoff", "selected_path": "model_sweep"},
            )
            return {
                "selection_path": selection_path,
                "summary_path": selection_summary_path,
                "selected_path": "model_sweep",
            }

        def fail_release_candidate(**_: object) -> dict[str, object]:
            raise AssertionError("Release gating must not run for invalid resumed continuation evidence")

        monkeypatch.setattr(run_retraining_loop, "_invoke_candidate_bridge", fake_candidate)
        monkeypatch.setattr(run_retraining_loop, "_invoke_path_selection_bridge", fake_path_selection)
        monkeypatch.setattr(run_retraining_loop, "_resolve_release_candidate", fail_release_candidate)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_loop.py",
                "--release-record",
                str(release_record_path),
                "--retraining-decision",
                str(decision_path),
                "--current-data",
                "azureml:current:1",
                "--reference-data",
                "azureml:reference:1",
                "--data-config",
                "configs/data.yaml",
                "--mode",
                "submit_selected_path",
                "--release-mode",
                "after_release_monitor_handoff",
                "--release-config",
                "config.env",
                "--resume-continuation-summary",
                str(resume_summary_path),
                "--probe-request",
                "sample-data.json",
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit, match="status='submitted'"):
            run_retraining_loop.main()

        assert not (output_dir / "release").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
