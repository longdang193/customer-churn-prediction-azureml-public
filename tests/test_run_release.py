"""
@meta
type: test
scope: unit
domain: release-orchestration
covers:
  - Release entrypoint deployment config handoff
excludes:
  - Real Azure ML registry or endpoint calls
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
    temp_dir = TEST_TEMP_ROOT / f"release-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


class _RegisteredModel:
    name = "churn-model"
    version = "10"
    path = "azureml://jobs/job-123/outputs/mlflow_model/paths/"
    type = "mlflow_model"
    tags = {"lineage_status": "validated"}


def _patch_common_successful_release(
    monkeypatch,
    run_release: object,
    temp_dir: Path,
    *,
    extra_args: list[str] | None = None,
) -> Path:
    config_path = temp_dir / "config.env"
    config_path.write_text("AZURE_WORKSPACE_NAME=workspace\n", encoding="utf-8")
    (temp_dir / "smoke-request.json").write_text("{}", encoding="utf-8")

    argv = [
        "run_release.py",
        "--job-name",
        "job-123",
        "--config",
        str(config_path),
        "--data-config",
        "configs/data_smoke_eval.yaml",
        "--train-config",
        "configs/train_smoke.yaml",
    ]
    argv.extend(extra_args or [])
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(run_release, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run_release,
        "get_release_config",
        lambda _config_path: {
            "model_name": "churn-model",
            "endpoint_name": "churn-endpoint",
            "deployment_name": "blue",
            "instance_type": "Standard_D2as_v4",
            "instance_count": "1",
            "smoke_payload": "smoke-request.json",
        },
    )
    monkeypatch.setattr(
        run_release,
        "get_data_asset_config",
        lambda _config_path: {"data_asset_name": "current", "data_asset_version": "1"},
    )
    monkeypatch.setattr(
        run_release,
        "get_reference_data_asset_config",
        lambda _config_path: {"data_asset_name": "reference", "data_asset_version": "2"},
    )
    monkeypatch.setattr(run_release, "load_asset_manifest", lambda: {})
    monkeypatch.setattr(
        run_release,
        "build_asset_lineage_tags",
        lambda **kwargs: {
            "data_config": str(kwargs["data_config_path"]),
            "train_config": str(kwargs["train_config_path"]),
            "lineage_status": "validated",
        },
    )
    monkeypatch.setattr(run_release, "get_ml_client", lambda _config_path: object())

    def fake_download_json_output(_ml_client, _job_name, output_name, _target_dir):
        if output_name == "promotion_decision":
            return {"status": "promote", "primary_metric": "f1"}
        if output_name == "candidate_metrics":
            return {"model_name": "rf", "f1": 0.8, "roc_auc": 0.85}
        raise AssertionError(output_name)

    def fake_optional_json_output(_ml_client, _job_name, output_name, _target_dir):
        if output_name == "train_manifest":
            return {
                "config": {"config_paths": {"train_config": "configs/train_smoke.yaml"}},
                "tags": {"best_model": "rf"},
            }
        if output_name == "validation_manifest":
            return {
                "config": {"config_paths": {"data_config": "configs/data_smoke_eval.yaml"}},
            }
        return None

    monkeypatch.setattr(run_release, "_download_json_output", fake_download_json_output)
    monkeypatch.setattr(
        run_release,
        "_download_optional_json_output",
        fake_optional_json_output,
        raising=False,
    )
    monkeypatch.setattr(run_release, "ensure_promotable_decision", lambda _decision: None)
    return config_path


def test_deployment_metadata_from_result_preserves_repo_owned_scoring_fields() -> None:
    import run_release

    metadata = run_release._deployment_metadata_from_result(
        {
            "smoke_test_response": "[0]",
            "deployment_state": "Succeeded",
            "recovery_used": False,
            "finalization_timed_out": False,
            "traffic_updated": True,
            "smoke_invoked": True,
            "inference_capture_enabled": False,
            "inference_capture_mode": "release_evidence_only",
            "inference_capture_status": "disabled",
            "inference_capture_warnings": [],
            "inference_capture_output_path": None,
            "repo_owned_scoring_expected": True,
            "repo_owned_scoring_observed": False,
            "repo_owned_scoring_status": "generated_runtime_still_in_control",
            "repo_owned_scoring_log_markers": [],
            "repo_owned_scoring_warnings": [
                "Azure deployment logs did not show repo-owned scorer proof markers after canary invoke."
            ],
        },
        instance_type="Standard_D2as_v4",
        instance_count=1,
        smoke_payload="sample-data.json",
    )

    assert metadata["repo_owned_scoring_expected"] is True
    assert metadata["repo_owned_scoring_observed"] is False
    assert metadata["repo_owned_scoring_status"] == "generated_runtime_still_in_control"
    assert metadata["repo_owned_scoring_log_markers"] == []
    assert metadata["repo_owned_scoring_warnings"] == [
        "Azure deployment logs did not show repo-owned scorer proof markers after canary invoke."
    ]


def test_main_uses_configured_smoke_payload_for_deploy(monkeypatch) -> None:
    import run_release

    captured: dict[str, object] = {}
    temp_dir = _make_temp_dir()
    monkeypatch.chdir(temp_dir)
    smoke_payload_path = temp_dir / "smoke-request.json"

    try:
        config_path = temp_dir / "config.env"
        config_path.write_text("AZURE_WORKSPACE_NAME=workspace\n", encoding="utf-8")
        smoke_payload_path.write_text("{}", encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_release.py",
                "--job-name",
                "job-123",
                "--config",
                str(config_path),
                "--deploy",
            ],
        )
        monkeypatch.setattr(run_release, "load_dotenv", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            run_release,
            "get_release_config",
            lambda _config_path: {
                "model_name": "churn-model",
                "endpoint_name": "churn-endpoint",
                "deployment_name": "blue",
                "instance_type": "Standard_D2as_v4",
                "instance_count": "1",
                "smoke_payload": "smoke-request.json",
            },
        )
        monkeypatch.setattr(
            run_release,
            "get_data_asset_config",
            lambda _config_path: {"data_asset_name": "current", "data_asset_version": "1"},
        )
        monkeypatch.setattr(
            run_release,
            "get_reference_data_asset_config",
            lambda _config_path: {"data_asset_name": "reference", "data_asset_version": "2"},
        )
        monkeypatch.setattr(run_release, "load_asset_manifest", lambda: {})
        monkeypatch.setattr(
            run_release,
            "build_asset_lineage_tags",
            lambda **_kwargs: {"data_asset": "current"},
        )
        monkeypatch.setattr(
            run_release,
            "_download_optional_json_output",
            lambda *_args, **_kwargs: None,
            raising=False,
        )
        monkeypatch.setattr(run_release, "get_ml_client", lambda _config_path: object())
        monkeypatch.setattr(
            run_release,
            "find_reusable_registered_model",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            run_release,
            "_download_json_output",
            lambda *_args: {"status": "promote", "primary_metric": "f1"},
        )
        monkeypatch.setattr(run_release, "ensure_promotable_decision", lambda _decision: None)

        class RegisteredModel:
            name = "churn-model"
            version = "7"

        monkeypatch.setattr(
            run_release,
            "register_promoted_model",
            lambda *_args, **_kwargs: RegisteredModel(),
        )

        def fake_deploy_registered_model(*_args, **kwargs):
            captured["sample_data_path"] = kwargs["sample_data_path"]
            return {
                "endpoint_name": kwargs["endpoint_name"],
                "deployment_name": kwargs["deployment_name"],
                "deployment_state": "Succeeded",
                "recovery_used": False,
                "finalization_timed_out": False,
                "traffic_updated": True,
                "smoke_invoked": True,
                "smoke_test_response": "ok",
                "payload_summary": {
                    "path": str(kwargs["sample_data_path"]),
                    "format": "input_data_2d_array",
                    "row_count": 1,
                    "feature_count": 10,
                    "validation_status": "passed",
                    },
                }

        monkeypatch.setattr(run_release, "deploy_registered_model", fake_deploy_registered_model)

        run_release.main()

        assert captured["sample_data_path"] == smoke_payload_path.resolve()
        record_path = temp_dir / ".release-artifacts" / "job-123" / "release_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["canary_inference"]["payload"]["validation_status"] == "passed"
        assert record["canary_inference"]["payload"]["feature_count"] == 10
        assert record["canary_inference"]["endpoint"]["endpoint_name"] == "churn-endpoint"
        assert record["canary_inference"]["model"]["version"] == "7"
        assert record["canary_inference"]["response"]["preview"] == "ok"
        assert record["monitoring_handoff"]["status"] == "ready_for_basic_monitoring_handoff"
        assert record["monitoring_handoff"]["canary_invoked"] is True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_blocks_lineage_mismatch_before_registration(monkeypatch) -> None:
    import run_release

    temp_dir = _make_temp_dir()
    register_called = False
    deploy_called = False
    monkeypatch.chdir(temp_dir)

    try:
        config_path = temp_dir / "config.env"
        config_path.write_text("AZURE_WORKSPACE_NAME=workspace\n", encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_release.py",
                "--job-name",
                "job-123",
                "--config",
                str(config_path),
                "--data-config",
                "configs/data_smoke_eval.yaml",
                "--train-config",
                "configs/train_smoke_hpo_winner_rf.yaml",
                "--deploy",
            ],
        )
        monkeypatch.setattr(run_release, "load_dotenv", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            run_release,
            "get_release_config",
            lambda _config_path: {
                "model_name": "churn-model",
                "endpoint_name": "churn-endpoint",
                "deployment_name": "blue",
                "instance_type": "Standard_D2as_v4",
                "instance_count": "1",
                "smoke_payload": "smoke-request.json",
            },
        )
        monkeypatch.setattr(
            run_release,
            "get_data_asset_config",
            lambda _config_path: {"data_asset_name": "current", "data_asset_version": "1"},
        )
        monkeypatch.setattr(
            run_release,
            "get_reference_data_asset_config",
            lambda _config_path: {"data_asset_name": "reference", "data_asset_version": "2"},
        )
        monkeypatch.setattr(run_release, "load_asset_manifest", lambda: {})
        monkeypatch.setattr(
            run_release,
            "build_asset_lineage_tags",
            lambda **kwargs: {
                "data_config": str(kwargs["data_config_path"]),
                "train_config": str(kwargs["train_config_path"]),
            },
        )
        monkeypatch.setattr(run_release, "get_ml_client", lambda _config_path: object())

        def fake_download_json_output(_ml_client, _job_name, output_name, _target_dir):
            if output_name == "promotion_decision":
                return {"status": "promote", "primary_metric": "f1"}
            if output_name == "candidate_metrics":
                return {"model_name": "logreg", "f1": 0.75, "roc_auc": 0.8125}
            raise AssertionError(output_name)

        def fake_optional_json_output(_ml_client, _job_name, output_name, _target_dir):
            if output_name == "train_manifest":
                return {
                    "config": {
                        "config_paths": {
                            "train_config": "/mnt/azureml/wd/INPUT_config/train_smoke.yaml",
                        },
                    },
                    "tags": {"best_model": "logreg"},
                }
            if output_name == "validation_manifest":
                return {"config": {"config_paths": {"data_config": "configs/data_smoke_eval.yaml"}}}
            return None

        def fake_register_promoted_model(*_args, **_kwargs):
            nonlocal register_called
            register_called = True
            raise AssertionError("registration should not run after lineage mismatch")

        def fake_deploy_registered_model(*_args, **_kwargs):
            nonlocal deploy_called
            deploy_called = True
            raise AssertionError("deployment should not run after lineage mismatch")

        monkeypatch.setattr(run_release, "_download_json_output", fake_download_json_output)
        monkeypatch.setattr(
            run_release,
            "_download_optional_json_output",
            fake_optional_json_output,
            raising=False,
        )
        monkeypatch.setattr(run_release, "ensure_promotable_decision", lambda _decision: None)
        monkeypatch.setattr(run_release, "register_promoted_model", fake_register_promoted_model)
        monkeypatch.setattr(run_release, "deploy_registered_model", fake_deploy_registered_model)

        with pytest.raises(SystemExit, match="Release lineage validation failed"):
            run_release.main()

        assert register_called is False
        assert deploy_called is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_accepts_canonical_train_config_even_when_runtime_filename_is_generic(
    monkeypatch,
) -> None:
    import run_release

    temp_dir = _make_temp_dir()
    monkeypatch.chdir(temp_dir)

    try:
        _patch_common_successful_release(
            monkeypatch,
            run_release,
            temp_dir,
            extra_args=[
                "--data-config",
                "configs/data_smoke_eval.yaml",
                "--train-config",
                "configs/train_smoke_hpo_winner_rf.yaml",
            ],
        )
        monkeypatch.setattr(
            run_release,
            "find_reusable_registered_model",
            lambda *_args, **_kwargs: _RegisteredModel(),
            raising=False,
        )

        def fake_optional_json_output(_ml_client, _job_name, output_name, _target_dir):
            if output_name == "train_manifest":
                return {
                    "config": {
                        "config_paths": {
                            "train_config": "/mnt/azureml/wd/INPUT_config/train_config.yaml",
                            "canonical_train_config": "configs/train_smoke_hpo_winner_rf.yaml",
                        },
                    },
                    "tags": {"best_model": "rf"},
                }
            if output_name == "validation_manifest":
                return {
                    "config": {"config_paths": {"data_config": "configs/data_smoke_eval.yaml"}}
                }
            return None

        monkeypatch.setattr(
            run_release,
            "_download_optional_json_output",
            fake_optional_json_output,
            raising=False,
        )

        run_release.main()

        record_path = temp_dir / ".release-artifacts" / "job-123" / "release_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["status"] == "succeeded"
        assert record["lineage"]["effective_lineage"]["train_config"] == (
            "configs/train_smoke_hpo_winner_rf.yaml"
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_reuses_matching_registered_model_by_default(monkeypatch) -> None:
    import run_release

    temp_dir = _make_temp_dir()
    monkeypatch.chdir(temp_dir)

    try:
        _patch_common_successful_release(monkeypatch, run_release, temp_dir)
        monkeypatch.setattr(
            run_release,
            "find_reusable_registered_model",
            lambda *_args, **_kwargs: _RegisteredModel(),
            raising=False,
        )

        def fail_registration(*_args, **_kwargs):
            raise AssertionError("matching release should reuse existing model")

        monkeypatch.setattr(run_release, "register_promoted_model", fail_registration)

        run_release.main()

        record_path = temp_dir / ".release-artifacts" / "job-123" / "release_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["status"] == "succeeded"
        assert record["model_resolution"] == "reused"
        assert record["registered_model"]["version"] == "10"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_force_reregister_skips_matching_reuse(monkeypatch) -> None:
    import run_release

    temp_dir = _make_temp_dir()
    register_called = False
    monkeypatch.chdir(temp_dir)

    try:
        _patch_common_successful_release(
            monkeypatch,
            run_release,
            temp_dir,
            extra_args=["--force-reregister"],
        )
        monkeypatch.setattr(
            run_release,
            "find_reusable_registered_model",
            lambda *_args, **_kwargs: _RegisteredModel(),
            raising=False,
        )

        def fake_register_promoted_model(*_args, **_kwargs):
            nonlocal register_called
            register_called = True
            return _RegisteredModel()

        monkeypatch.setattr(run_release, "register_promoted_model", fake_register_promoted_model)

        run_release.main()

        record_path = temp_dir / ".release-artifacts" / "job-123" / "release_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert register_called is True
        assert record["status"] == "succeeded"
        assert record["model_resolution"] == "registered"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_writes_failed_release_record_when_deployment_fails_after_registration(
    monkeypatch,
) -> None:
    import run_release

    temp_dir = _make_temp_dir()
    monkeypatch.chdir(temp_dir)

    try:
        _patch_common_successful_release(
            monkeypatch,
            run_release,
            temp_dir,
            extra_args=["--deploy"],
        )
        monkeypatch.setattr(
            run_release,
            "find_reusable_registered_model",
            lambda *_args, **_kwargs: None,
            raising=False,
        )
        monkeypatch.setattr(
            run_release,
            "register_promoted_model",
            lambda *_args, **_kwargs: _RegisteredModel(),
        )

        def fail_deployment(*_args, **_kwargs):
            raise RuntimeError("quota exceeded")

        monkeypatch.setattr(run_release, "deploy_registered_model", fail_deployment)

        with pytest.raises(RuntimeError, match="quota exceeded"):
            run_release.main()

        record_path = temp_dir / ".release-artifacts" / "job-123" / "release_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["status"] == "failed"
        assert record["model_resolution"] == "registered"
        assert record["failure"]["failure_stage"] == "deployment"
        assert record["failure"]["error_type"] == "RuntimeError"
        assert record["failure"]["error_message"] == "quota exceeded"
        assert record["registered_model"]["version"] == "10"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_preserves_structured_deployment_metadata_for_payload_validation_failure(
    monkeypatch,
) -> None:
    import run_release

    temp_dir = _make_temp_dir()
    monkeypatch.chdir(temp_dir)

    try:
        _patch_common_successful_release(
            monkeypatch,
            run_release,
            temp_dir,
            extra_args=["--deploy"],
        )
        monkeypatch.setattr(
            run_release,
            "find_reusable_registered_model",
            lambda *_args, **_kwargs: _RegisteredModel(),
            raising=False,
        )
        monkeypatch.setattr(
            run_release,
            "deploy_registered_model",
            lambda *_args, **_kwargs: {
                "endpoint_name": "churn-endpoint",
                "deployment_name": "blue",
                "deployment_state": "Succeeded",
                "recovery_used": False,
                "finalization_timed_out": False,
                "traffic_updated": True,
                "smoke_invoked": False,
                "smoke_test_response": None,
                "failure": {
                    "failure_stage": "deployment",
                    "error_type": "ValueError",
                    "error_message": "Endpoint payload input_data row 0 expected 10 features, got 3.",
                },
            },
        )

        with pytest.raises(SystemExit, match="Release deployment did not reach a successful terminal state"):
            run_release.main()

        record_path = temp_dir / ".release-artifacts" / "job-123" / "release_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["status"] == "failed"
        assert record["deployment"]["deployment_state"] == "Succeeded"
        assert record["deployment"]["traffic_updated"] is True
        assert record["deployment"]["smoke_invoked"] is False
        assert record["failure"]["error_type"] == "ValueError"
        assert record["monitoring_handoff"]["status"] == "canary_failed_after_deploy"
        assert record["monitoring_handoff"]["canary_invoked"] is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_default_release_payload_fixture_still_fails_local_validation_for_wrong_feature_count() -> None:
    from src.inference.payloads import DEFAULT_ENDPOINT_FEATURE_COUNT, validate_endpoint_payload_file

    payload_path = Path("sample-data-invalid-feature-count.json")

    with pytest.raises(
        ValueError,
        match=f"expected {DEFAULT_ENDPOINT_FEATURE_COUNT} features, got 9",
    ):
        validate_endpoint_payload_file(payload_path)


def test_main_writes_failed_release_record_for_deployment_finalization_timeout(
    monkeypatch,
) -> None:
    import run_release

    temp_dir = _make_temp_dir()
    monkeypatch.chdir(temp_dir)

    try:
        _patch_common_successful_release(
            monkeypatch,
            run_release,
            temp_dir,
            extra_args=["--deploy"],
        )
        monkeypatch.setattr(
            run_release,
            "find_reusable_registered_model",
            lambda *_args, **_kwargs: _RegisteredModel(),
            raising=False,
        )

        monkeypatch.setattr(
            run_release,
            "deploy_registered_model",
            lambda *_args, **_kwargs: {
                "endpoint_name": "churn-endpoint",
                "deployment_name": "blue",
                "deployment_state": "Creating",
                "recovery_used": True,
                "finalization_timed_out": True,
                "traffic_updated": False,
                "smoke_invoked": False,
                "smoke_test_response": None,
                "failure": {
                    "failure_stage": "deployment_finalization_timeout",
                    "error_type": "TimeoutError",
                    "error_message": "local release wait expired before Azure became terminal",
                },
            },
        )

        with pytest.raises(SystemExit, match="Release deployment did not reach a successful terminal state"):
            run_release.main()

        record_path = temp_dir / ".release-artifacts" / "job-123" / "release_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["status"] == "failed"
        assert record["model_resolution"] == "reused"
        assert record["deployment"]["deployment_state"] == "Creating"
        assert record["deployment"]["finalization_timed_out"] is True
        assert record["failure"]["failure_stage"] == "deployment_finalization_timeout"
        assert "canary_inference" not in record
        assert record["monitoring_handoff"]["status"] == "deploy_incomplete_or_timed_out"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_writes_successful_release_record_for_delayed_deployment_success(
    monkeypatch,
) -> None:
    import run_release

    temp_dir = _make_temp_dir()
    monkeypatch.chdir(temp_dir)

    try:
        _patch_common_successful_release(
            monkeypatch,
            run_release,
            temp_dir,
            extra_args=["--deploy"],
        )
        monkeypatch.setattr(
            run_release,
            "find_reusable_registered_model",
            lambda *_args, **_kwargs: _RegisteredModel(),
            raising=False,
        )

        monkeypatch.setattr(
            run_release,
            "deploy_registered_model",
            lambda *_args, **_kwargs: {
                "endpoint_name": "churn-endpoint",
                "deployment_name": "blue",
                "deployment_state": "Succeeded",
                "recovery_used": True,
                "finalization_timed_out": False,
                "traffic_updated": True,
                "smoke_invoked": True,
                "smoke_test_response": "ok",
                "payload_summary": {
                    "path": "smoke-request.json",
                    "format": "input_data_2d_array",
                    "row_count": 1,
                    "feature_count": 10,
                    "validation_status": "passed",
                },
            },
        )

        run_release.main()

        record_path = temp_dir / ".release-artifacts" / "job-123" / "release_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["status"] == "succeeded"
        assert record["deployment"]["deployment_state"] == "Succeeded"
        assert record["deployment"]["recovery_used"] is True
        assert record["deployment"]["traffic_updated"] is True
        assert record["deployment"]["smoke_invoked"] is True
        assert record["canary_inference"]["payload"]["validation_status"] == "passed"
        assert record["monitoring_handoff"]["status"] == "ready_for_basic_monitoring_handoff"
        assert record["monitoring_handoff"]["evidence_level"] == "release_evidence_only"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
