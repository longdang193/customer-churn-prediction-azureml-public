"""
@meta
type: test
scope: unit
domain: retraining-hpo-continuation
covers:
  - HPO winner consistency checks before fixed-train continuation
  - Reuse or export of the effective fixed-train config
  - Truthful downstream fixed-train handoff summary propagation
excludes:
  - Real Azure ML submission
  - Real HPO download
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
    temp_dir = TEST_TEMP_ROOT / f"retraining-hpo-to-fixed-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _release_record() -> dict[str, object]:
    return {
        "status": "succeeded",
        "deployment": {
            "endpoint_name": "churn-endpoint",
            "deployment_name": "blue",
        },
    }


def _selection(selected_path: str = "model_sweep") -> dict[str, object]:
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


def _hpo_smoke_summary(
    *,
    status: str = "submitted",
    job_name: str | None = "hpo-parent-123",
) -> dict[str, object]:
    return {
        "status": status,
        "validation_status": "passed",
        "selected_path": "model_sweep",
        "release_record_path": "release_record.json",
        "selection_path": "retraining_path_selection.json",
        "candidate_manifest_path": "retraining_candidate_manifest.json",
        "validation_summary_path": "validation_summary.json",
        "invocation_path": "hpo_invocation.json",
        "submission": {
            "attempted": status == "submitted",
            "job_name": job_name,
            "studio_url": "https://studio/hpo" if job_name else None,
            "stderr": "",
        },
    }


def _write_minimal_hpo_run(run_dir: Path, *, winner_family: str = "logreg") -> None:
    _write_json(
        run_dir / "named-outputs" / "hpo_summary" / "hpo_summary.json",
        {
            "primary_metric": "f1",
            "winner": {
                "model_name": winner_family,
                "run_id": "winner-run-1",
                "score": 0.88,
                "tie_break_reason": "primary_metric",
                "tie_candidates": [winner_family],
            },
            "candidate_results": [
                {
                    "model_name": winner_family,
                    "run_id": "winner-run-1",
                    "metrics": {
                        "f1": 0.88,
                        "roc_auc": 0.87,
                    },
                }
            ],
            "family_artifacts": {},
            "family_bundle_artifacts": {},
        },
    )
    _write_json(
        run_dir / "named-outputs" / "winner_manifest" / "step_manifest.json",
        {
            "inputs": {"winner_family": winner_family},
            "tags": {"winner_model": winner_family},
            "step_specific": {
                "materialized_outputs": {
                    "winner_model_name": winner_family,
                    "winner_train_config": "winner_train_config/train_config.yaml",
                }
            },
        },
    )
    _write_text(
        run_dir / "named-outputs" / "winner_train_config" / "train_config.yaml",
        "\n".join(
            [
                "training:",
                "  models:",
                f"  - {winner_family}",
                "  experiment_name: train-prod",
                "promotion:",
                "  primary_metric: f1",
                "",
            ]
        ),
    )
    _write_json(
        run_dir / "named-outputs" / f"{winner_family}_hpo_manifest" / "step_manifest.json",
        {
            "params": {
                "hyperparameters": {
                    f"{winner_family}_alpha": "0.1",
                }
            }
        },
    )
    _write_json(
        run_dir / "named-outputs" / f"{winner_family}_train_manifest" / "step_manifest.json",
        {
            "params": {
                "class_weight": "balanced",
                "random_state": 42,
                "use_smote": True,
            }
        },
    )


def test_main_blocks_when_hpo_smoke_summary_not_submitted(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_hpo_to_fixed_train

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "continuation"
    try:
        release_record_path = temp_dir / "release_record.json"
        selection_path = temp_dir / "selection.json"
        candidate_manifest_path = temp_dir / "candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        hpo_smoke_summary_path = temp_dir / "hpo_smoke_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(selection_path, _selection())
        _write_json(candidate_manifest_path, _candidate_manifest())
        _write_json(validation_summary_path, {"status": "passed"})
        _write_json(hpo_smoke_summary_path, _hpo_smoke_summary(status="dry_run_ready", job_name=None))

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_hpo_to_fixed_train.py",
                "--release-record",
                str(release_record_path),
                "--selection",
                str(selection_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--hpo-smoke-summary",
                str(hpo_smoke_summary_path),
                "--hpo-run-dir",
                str(temp_dir / "downloaded-hpo"),
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_hpo_to_fixed_train.main()

        summary = json.loads(
            (output_dir / "retraining_hpo_to_fixed_train_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "blocked_by_hpo_status"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_blocks_when_selected_path_is_not_model_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_hpo_to_fixed_train

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "continuation"
    try:
        release_record_path = temp_dir / "release_record.json"
        selection_path = temp_dir / "selection.json"
        candidate_manifest_path = temp_dir / "candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        hpo_smoke_summary_path = temp_dir / "hpo_smoke_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(selection_path, _selection(selected_path="fixed_train"))
        _write_json(candidate_manifest_path, _candidate_manifest())
        _write_json(validation_summary_path, {"status": "passed"})
        _write_json(hpo_smoke_summary_path, _hpo_smoke_summary())

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_hpo_to_fixed_train.py",
                "--release-record",
                str(release_record_path),
                "--selection",
                str(selection_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--hpo-smoke-summary",
                str(hpo_smoke_summary_path),
                "--hpo-run-dir",
                str(temp_dir / "downloaded-hpo"),
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_hpo_to_fixed_train.main()

        summary = json.loads(
            (output_dir / "retraining_hpo_to_fixed_train_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "blocked_by_selection"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_blocks_when_winner_identity_is_inconsistent(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_retraining_hpo_to_fixed_train

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "continuation"
    run_dir = temp_dir / "downloaded-hpo"
    try:
        release_record_path = temp_dir / "release_record.json"
        selection_path = temp_dir / "selection.json"
        candidate_manifest_path = temp_dir / "candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        hpo_smoke_summary_path = temp_dir / "hpo_smoke_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(selection_path, _selection())
        _write_json(candidate_manifest_path, _candidate_manifest())
        _write_json(validation_summary_path, {"status": "passed"})
        _write_json(hpo_smoke_summary_path, _hpo_smoke_summary())
        _write_minimal_hpo_run(run_dir, winner_family="logreg")
        _write_text(
            run_dir / "named-outputs" / "winner_train_config" / "train_config.yaml",
            "\n".join(
                [
                    "training:",
                    "  models:",
                    "  - rf",
                    "",
                ]
            ),
        )

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_hpo_to_fixed_train.py",
                "--release-record",
                str(release_record_path),
                "--selection",
                str(selection_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--hpo-smoke-summary",
                str(hpo_smoke_summary_path),
                "--hpo-run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_hpo_to_fixed_train.main()

        summary = json.loads(
            (output_dir / "retraining_hpo_to_fixed_train_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "blocked_by_winner_inconsistency"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_blocks_when_hpo_summary_winner_disagrees_with_winner_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_hpo_to_fixed_train

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "continuation"
    run_dir = temp_dir / "downloaded-hpo"
    try:
        release_record_path = temp_dir / "release_record.json"
        selection_path = temp_dir / "selection.json"
        candidate_manifest_path = temp_dir / "candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        hpo_smoke_summary_path = temp_dir / "hpo_smoke_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(selection_path, _selection())
        _write_json(candidate_manifest_path, _candidate_manifest())
        _write_json(validation_summary_path, {"status": "passed"})
        _write_json(hpo_smoke_summary_path, _hpo_smoke_summary())
        _write_minimal_hpo_run(run_dir, winner_family="logreg")
        _write_json(
            run_dir / "named-outputs" / "winner_manifest" / "step_manifest.json",
            {
                "inputs": {"winner_family": "rf"},
                "tags": {"winner_model": "rf"},
                "step_specific": {
                    "materialized_outputs": {
                        "winner_model_name": "rf",
                        "winner_train_config": "winner_train_config/train_config.yaml",
                    }
                },
            },
        )

        def should_not_continue(**_: object) -> dict[str, object]:
            raise AssertionError("fixed-train bridge should not run on winner mismatch")

        monkeypatch.setattr(run_retraining_hpo_to_fixed_train, "_invoke_fixed_train_bridge", should_not_continue)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_hpo_to_fixed_train.py",
                "--release-record",
                str(release_record_path),
                "--selection",
                str(selection_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--hpo-smoke-summary",
                str(hpo_smoke_summary_path),
                "--hpo-run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_hpo_to_fixed_train.main()

        summary = json.loads(
            (output_dir / "retraining_hpo_to_fixed_train_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "blocked_by_winner_inconsistency"
        assert summary["submitted_job_name"] is None
        assert summary["downstream_summary_path"] is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_blocks_when_required_winner_manifest_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_hpo_to_fixed_train

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "continuation"
    run_dir = temp_dir / "downloaded-hpo"
    try:
        release_record_path = temp_dir / "release_record.json"
        selection_path = temp_dir / "selection.json"
        candidate_manifest_path = temp_dir / "candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        hpo_smoke_summary_path = temp_dir / "hpo_smoke_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(selection_path, _selection())
        _write_json(candidate_manifest_path, _candidate_manifest())
        _write_json(validation_summary_path, {"status": "passed"})
        _write_json(hpo_smoke_summary_path, _hpo_smoke_summary())
        _write_minimal_hpo_run(run_dir, winner_family="logreg")
        (run_dir / "named-outputs" / "winner_manifest" / "step_manifest.json").unlink()

        def should_not_continue(**_: object) -> dict[str, object]:
            raise AssertionError("fixed-train bridge should not run when winner manifest is missing")

        monkeypatch.setattr(run_retraining_hpo_to_fixed_train, "_invoke_fixed_train_bridge", should_not_continue)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_hpo_to_fixed_train.py",
                "--release-record",
                str(release_record_path),
                "--selection",
                str(selection_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--hpo-smoke-summary",
                str(hpo_smoke_summary_path),
                "--hpo-run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_hpo_to_fixed_train.main()

        summary = json.loads(
            (output_dir / "retraining_hpo_to_fixed_train_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "blocked_by_winner_inconsistency"
        assert summary["submitted_job_name"] is None
        assert summary["downstream_summary_path"] is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_reuses_existing_winner_train_config_and_invokes_fixed_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    @proves fixed-train.accept-hpo-to-fixed-continuation
    @proves fixed-train.accept-hpo-winner-train-config
    """
    import run_retraining_hpo_to_fixed_train

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "continuation"
    run_dir = temp_dir / "downloaded-hpo"
    calls: list[dict[str, object]] = []
    try:
        release_record_path = temp_dir / "release_record.json"
        selection_path = temp_dir / "selection.json"
        candidate_manifest_path = temp_dir / "candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        hpo_smoke_summary_path = temp_dir / "hpo_smoke_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(selection_path, _selection())
        _write_json(candidate_manifest_path, _candidate_manifest())
        _write_json(validation_summary_path, {"status": "passed"})
        _write_json(hpo_smoke_summary_path, _hpo_smoke_summary())
        _write_minimal_hpo_run(run_dir, winner_family="logreg")

        def fail_if_exported(**_: object) -> dict[str, object]:
            raise AssertionError("winner config should have been reused")

        def fake_fixed_train(
            *,
            release_record_path: Path,
            candidate_manifest_path: Path,
            validation_summary_path: Path,
            data_config_path: str,
            train_config_path: Path,
            output_dir: Path,
            submit: bool,
        ) -> dict[str, object]:
            calls.append(
                {
                    "release_record_path": release_record_path,
                    "candidate_manifest_path": candidate_manifest_path,
                    "validation_summary_path": validation_summary_path,
                    "data_config_path": data_config_path,
                    "train_config_path": train_config_path,
                    "submit": submit,
                }
            )
            summary_path = output_dir / "retraining_fixed_train_summary.json"
            _write_json(
                summary_path,
                {
                    "status": "dry_run_ready",
                    "submission": {"attempted": False, "job_name": None},
                },
            )
            return {
                "summary_path": summary_path,
                "status": "dry_run_ready",
                "job_name": None,
            }

        monkeypatch.setattr(run_retraining_hpo_to_fixed_train, "export_winner_config", fail_if_exported)
        monkeypatch.setattr(run_retraining_hpo_to_fixed_train, "_invoke_fixed_train_bridge", fake_fixed_train)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_hpo_to_fixed_train.py",
                "--release-record",
                str(release_record_path),
                "--selection",
                str(selection_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--hpo-smoke-summary",
                str(hpo_smoke_summary_path),
                "--hpo-run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        run_retraining_hpo_to_fixed_train.main()

        summary = json.loads(
            (output_dir / "retraining_hpo_to_fixed_train_summary.json").read_text(encoding="utf-8")
        )
        exported = json.loads(
            (output_dir / "retraining_exported_train_config.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "dry_run_ready"
        assert exported["source_kind"] == "reused_winner_train_config"
        assert len(calls) == 1
        assert Path(str(calls[0]["train_config_path"])).exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_downloads_hpo_outputs_and_exports_when_winner_config_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    @proves fixed-train.accept-hpo-to-fixed-continuation
    @proves hpo.export-selected-hpo-winner-fixed-train-config-yaml
    """
    import run_retraining_hpo_to_fixed_train

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "continuation"
    run_dir = temp_dir / "downloaded-hpo"
    calls: dict[str, object] = {}
    try:
        release_record_path = temp_dir / "release_record.json"
        selection_path = temp_dir / "selection.json"
        candidate_manifest_path = temp_dir / "candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        hpo_smoke_summary_path = temp_dir / "hpo_smoke_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(selection_path, _selection())
        _write_json(candidate_manifest_path, _candidate_manifest())
        _write_json(validation_summary_path, {"status": "passed"})
        _write_json(hpo_smoke_summary_path, _hpo_smoke_summary(job_name="hungry_screw_h500178g5p"))
        _write_minimal_hpo_run(run_dir, winner_family="logreg")
        (run_dir / "named-outputs" / "winner_train_config" / "train_config.yaml").unlink()

        def fake_download_hpo_run(
            *,
            job_name: str,
            resource_group: str,
            workspace_name: str,
            download_path: Path,
        ) -> Path:
            calls["download"] = {
                "job_name": job_name,
                "resource_group": resource_group,
                "workspace_name": workspace_name,
                "download_path": download_path,
            }
            return run_dir

        def fake_wait_for_hpo_job_completion(
            *,
            job_name: str,
            resource_group: str,
            workspace_name: str,
            timeout_seconds: int = 0,
            poll_interval_seconds: int = 0,
        ) -> str:
            calls["wait"] = {
                "job_name": job_name,
                "resource_group": resource_group,
                "workspace_name": workspace_name,
                "timeout_seconds": timeout_seconds,
                "poll_interval_seconds": poll_interval_seconds,
            }
            return "completed"

        def fake_export_winner_config(
            *,
            run_dir: Path,
            output_config: Path,
            base_config_path: Path,
            experiment_name: str | None = None,
            display_name: str | None = None,
        ) -> dict[str, object]:
            calls["export"] = {
                "run_dir": run_dir,
                "output_config": output_config,
                "base_config_path": base_config_path,
                "experiment_name": experiment_name,
                "display_name": display_name,
            }
            _write_text(
                output_config,
                "\n".join(
                    [
                        "training:",
                        "  models:",
                        "  - logreg",
                        "",
                    ]
                ),
            )
            return {"training": {"models": ["logreg"]}}

        def fake_fixed_train(
            *,
            release_record_path: Path,
            candidate_manifest_path: Path,
            validation_summary_path: Path,
            data_config_path: str,
            train_config_path: Path,
            output_dir: Path,
            submit: bool,
        ) -> dict[str, object]:
            calls["fixed_train"] = {
                "train_config_path": train_config_path,
                "submit": submit,
            }
            summary_path = output_dir / "retraining_fixed_train_summary.json"
            _write_json(
                summary_path,
                {
                    "status": "submitted",
                    "submission": {"attempted": True, "job_name": "fixed-train-job-123"},
                },
            )
            return {
                "summary_path": summary_path,
                "status": "submitted",
                "job_name": "fixed-train-job-123",
            }

        monkeypatch.setattr(
            run_retraining_hpo_to_fixed_train,
            "_wait_for_hpo_job_completion",
            fake_wait_for_hpo_job_completion,
        )
        monkeypatch.setattr(run_retraining_hpo_to_fixed_train, "download_hpo_run", fake_download_hpo_run)
        monkeypatch.setattr(run_retraining_hpo_to_fixed_train, "export_winner_config", fake_export_winner_config)
        monkeypatch.setattr(run_retraining_hpo_to_fixed_train, "_invoke_fixed_train_bridge", fake_fixed_train)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_hpo_to_fixed_train.py",
                "--release-record",
                str(release_record_path),
                "--selection",
                str(selection_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--hpo-smoke-summary",
                str(hpo_smoke_summary_path),
                "--hpo-job-name",
                "hungry_screw_h500178g5p",
                "--resource-group",
                "rg-churn-ml-project",
                "--workspace-name",
                "churn-ml-workspace",
                "--output-dir",
                str(output_dir),
                "--submit",
            ],
        )

        run_retraining_hpo_to_fixed_train.main()

        summary = json.loads(
            (output_dir / "retraining_hpo_to_fixed_train_summary.json").read_text(encoding="utf-8")
        )
        exported = json.loads(
            (output_dir / "retraining_exported_train_config.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "submitted"
        assert summary["submitted_job_name"] == "fixed-train-job-123"
        assert exported["source_kind"] == "exported_from_hpo"
        assert "wait" in calls
        assert "download" in calls
        assert "export" in calls
        assert "fixed_train" in calls
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_blocks_when_remote_hpo_job_does_not_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_retraining_hpo_to_fixed_train

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "continuation"
    calls: dict[str, object] = {}
    try:
        release_record_path = temp_dir / "release_record.json"
        selection_path = temp_dir / "selection.json"
        candidate_manifest_path = temp_dir / "candidate_manifest.json"
        validation_summary_path = temp_dir / "validation_summary.json"
        hpo_smoke_summary_path = temp_dir / "hpo_smoke_summary.json"
        _write_json(release_record_path, _release_record())
        _write_json(selection_path, _selection())
        _write_json(candidate_manifest_path, _candidate_manifest())
        _write_json(validation_summary_path, {"status": "passed"})
        _write_json(hpo_smoke_summary_path, _hpo_smoke_summary(job_name="blue_fowl_zzg53xbv34"))

        def fake_wait_for_hpo_job_completion(
            *,
            job_name: str,
            resource_group: str,
            workspace_name: str,
            timeout_seconds: int = 0,
            poll_interval_seconds: int = 0,
        ) -> str:
            calls["wait"] = {
                "job_name": job_name,
                "resource_group": resource_group,
                "workspace_name": workspace_name,
            }
            return "running"

        def should_not_download_hpo_run(**_: object) -> Path:
            raise AssertionError("HPO download should not run before the remote job completes")

        def should_not_continue(**_: object) -> dict[str, object]:
            raise AssertionError("fixed-train bridge should not run when HPO parent is still running")

        monkeypatch.setattr(
            run_retraining_hpo_to_fixed_train,
            "_wait_for_hpo_job_completion",
            fake_wait_for_hpo_job_completion,
        )
        monkeypatch.setattr(run_retraining_hpo_to_fixed_train, "download_hpo_run", should_not_download_hpo_run)
        monkeypatch.setattr(run_retraining_hpo_to_fixed_train, "_invoke_fixed_train_bridge", should_not_continue)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_retraining_hpo_to_fixed_train.py",
                "--release-record",
                str(release_record_path),
                "--selection",
                str(selection_path),
                "--candidate-manifest",
                str(candidate_manifest_path),
                "--validation-summary",
                str(validation_summary_path),
                "--hpo-smoke-summary",
                str(hpo_smoke_summary_path),
                "--hpo-job-name",
                "blue_fowl_zzg53xbv34",
                "--resource-group",
                "rg-churn-ml-project",
                "--workspace-name",
                "churn-ml-workspace",
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit):
            run_retraining_hpo_to_fixed_train.main()

        summary = json.loads(
            (output_dir / "retraining_hpo_to_fixed_train_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "blocked_by_hpo_status"
        assert summary["submitted_job_name"] is None
        assert "wait" in calls
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
