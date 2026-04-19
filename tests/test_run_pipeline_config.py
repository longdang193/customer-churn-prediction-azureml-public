"""
@meta
type: test
scope: unit
domain: training-orchestration
covers:
  - Environment-driven compute selection for the fixed training pipeline
  - Reference/current data asset resolution for validation-aware execution
features:
  - model-training-pipeline
capabilities:
  - fixed-train.submit-pipeline-job
excludes:
  - Real Azure ML job submission
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pytest


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_get_pipeline_runtime_settings_reads_reference_assets_and_compute() -> None:
    from run_pipeline import get_pipeline_runtime_settings

    temp_dir = _make_temp_dir()
    try:
        config_env = temp_dir / "config.env"
        config_env.write_text(
            "\n".join(
                [
                    'AZURE_SUBSCRIPTION_ID="sub-1"',
                    'AZURE_RESOURCE_GROUP="rg-1"',
                    'AZURE_WORKSPACE_NAME="ws-1"',
                    'AZURE_PIPELINE_COMPUTE="cpu-train"',
                    'DATA_ASSET_FULL="churn-current"',
                    'DATA_VERSION="7"',
                    'DATA_REFERENCE_ASSET_FULL="churn-baseline"',
                    'DATA_REFERENCE_VERSION="3"',
                ]
            ),
            encoding="utf-8",
        )

        runtime = get_pipeline_runtime_settings(str(config_env))

        assert runtime.compute_name == "cpu-train"
        assert runtime.current_data_asset_name == "churn-current"
        assert runtime.current_data_asset_version == "7"
        assert runtime.reference_data_asset_name == "churn-baseline"
        assert runtime.reference_data_asset_version == "3"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_resolve_pipeline_data_inputs_uses_explicit_overrides() -> None:
    """
    @proves fixed-train.support-fixed-dataset-overrides
    """
    from run_pipeline import get_pipeline_runtime_settings, resolve_pipeline_data_inputs

    current_input, reference_input, cleanup_dirs = resolve_pipeline_data_inputs(
        current_data_override="azureml://datastores/workspaceblobstore/paths/retraining/current/",
        reference_data_override="azureml:approved-reference:9",
        runtime=get_pipeline_runtime_settings("config.env"),
    )

    assert current_input.type == "uri_folder"
    assert current_input.path == "azureml://datastores/workspaceblobstore/paths/retraining/current/"
    assert reference_input.type == "uri_folder"
    assert reference_input.path == "azureml:approved-reference:9"
    assert cleanup_dirs == []


def test_resolve_pipeline_data_inputs_stages_local_csv_overrides() -> None:
    from run_pipeline import get_pipeline_runtime_settings, resolve_pipeline_data_inputs

    temp_dir = _make_temp_dir()
    try:
        current_csv = temp_dir / "current.csv"
        reference_csv = temp_dir / "reference.csv"
        current_csv.write_text("a,b\n1,2\n", encoding="utf-8")
        reference_csv.write_text("a,b\n3,4\n", encoding="utf-8")

        current_input, reference_input, cleanup_dirs = resolve_pipeline_data_inputs(
            current_data_override=str(current_csv),
            reference_data_override=str(reference_csv),
            runtime=get_pipeline_runtime_settings("config.env"),
        )

        assert current_input.type == "uri_folder"
        assert reference_input.type == "uri_folder"
        assert len(cleanup_dirs) == 2
        assert Path(current_input.path).is_dir()
        assert Path(reference_input.path).is_dir()
        assert (Path(current_input.path) / current_csv.name).exists()
        assert (Path(reference_input.path) / reference_csv.name).exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        for cleanup_dir in locals().get("cleanup_dirs", []):
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def test_resolve_pipeline_data_inputs_requires_both_overrides() -> None:
    from run_pipeline import get_pipeline_runtime_settings, resolve_pipeline_data_inputs

    with pytest.raises(ValueError):
        resolve_pipeline_data_inputs(
            current_data_override="azureml://datastores/workspaceblobstore/paths/retraining/current/",
            reference_data_override=None,
            runtime=get_pipeline_runtime_settings("config.env"),
        )


def test_get_pipeline_runtime_settings_falls_back_to_training_asset_defaults() -> None:
    from run_pipeline import get_pipeline_runtime_settings

    temp_dir = _make_temp_dir()
    try:
        config_env = temp_dir / "config.env"
        config_env.write_text(
            "\n".join(
                [
                    'AZURE_SUBSCRIPTION_ID="sub-1"',
                    'AZURE_RESOURCE_GROUP="rg-1"',
                    'AZURE_WORKSPACE_NAME="ws-1"',
                    'AZURE_COMPUTE_CLUSTER_NAME="cpu-cluster"',
                    'DATA_ASSET_FULL="churn-current"',
                    'DATA_VERSION="9"',
                ]
            ),
            encoding="utf-8",
        )

        runtime = get_pipeline_runtime_settings(str(config_env))

        assert runtime.compute_name == "cpu-cluster"
        assert runtime.reference_data_asset_name == "churn-current"
        assert runtime.reference_data_asset_version == "9"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_get_pipeline_runtime_settings_uses_asset_manifest_defaults(
    monkeypatch,
) -> None:
    from run_pipeline import get_pipeline_runtime_settings

    for key in (
        "DATA_ASSET_FULL",
        "DATA_VERSION",
        "DATA_REFERENCE_ASSET_FULL",
        "DATA_REFERENCE_VERSION",
    ):
        monkeypatch.delenv(key, raising=False)

    temp_dir = _make_temp_dir()
    try:
        config_env = temp_dir / "config.env"
        config_env.write_text(
            "\n".join(
                [
                    'AZURE_SUBSCRIPTION_ID="sub-1"',
                    'AZURE_RESOURCE_GROUP="rg-1"',
                    'AZURE_WORKSPACE_NAME="ws-1"',
                    'AZURE_COMPUTE_CLUSTER_NAME="cpu-cluster"',
                ]
            ),
            encoding="utf-8",
        )

        runtime = get_pipeline_runtime_settings(str(config_env))

        assert runtime.current_data_asset_name == "churn-data"
        assert runtime.current_data_asset_version == "1"
        assert runtime.reference_data_asset_name == "churn-data"
        assert runtime.reference_data_asset_version == "1"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_get_pipeline_runtime_settings_uses_smoke_asset_for_smoke_data_config(
    monkeypatch,
) -> None:
    from run_pipeline import get_pipeline_runtime_settings

    for key in (
        "DATA_ASSET_FULL",
        "DATA_VERSION",
        "DATA_REFERENCE_ASSET_FULL",
        "DATA_REFERENCE_VERSION",
    ):
        monkeypatch.delenv(key, raising=False)

    temp_dir = _make_temp_dir()
    try:
        config_env = temp_dir / "config.env"
        config_env.write_text(
            "\n".join(
                [
                    'AZURE_SUBSCRIPTION_ID="sub-1"',
                    'AZURE_RESOURCE_GROUP="rg-1"',
                    'AZURE_WORKSPACE_NAME="ws-1"',
                    'AZURE_COMPUTE_CLUSTER_NAME="cpu-cluster"',
                ]
            ),
            encoding="utf-8",
        )

        runtime = get_pipeline_runtime_settings(
            str(config_env),
            data_config_path=Path("configs/data_smoke.yaml"),
        )

        assert runtime.current_data_asset_name == "churn-data-smoke"
        assert runtime.current_data_asset_version == "2"
        assert runtime.reference_data_asset_name == "churn-data-smoke"
        assert runtime.reference_data_asset_version == "2"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_get_pipeline_runtime_settings_uses_named_smoke_variant_asset(
    monkeypatch,
) -> None:
    from run_pipeline import get_pipeline_runtime_settings

    for key in (
        "DATA_ASSET_FULL",
        "DATA_VERSION",
        "DATA_REFERENCE_ASSET_FULL",
        "DATA_REFERENCE_VERSION",
    ):
        monkeypatch.delenv(key, raising=False)

    temp_dir = _make_temp_dir()
    try:
        config_env = temp_dir / "config.env"
        config_env.write_text(
            "\n".join(
                [
                    'AZURE_SUBSCRIPTION_ID="sub-1"',
                    'AZURE_RESOURCE_GROUP="rg-1"',
                    'AZURE_WORKSPACE_NAME="ws-1"',
                    'AZURE_COMPUTE_CLUSTER_NAME="cpu-cluster"',
                ]
            ),
            encoding="utf-8",
        )

        runtime = get_pipeline_runtime_settings(
            str(config_env),
            data_config_path=Path("configs/data_smoke_eval.yaml"),
        )

        assert runtime.current_data_asset_name == "churn-data-smoke-eval"
        assert runtime.current_data_asset_version == "1"
        assert runtime.reference_data_asset_name == "churn-data-smoke"
        assert runtime.reference_data_asset_version == "2"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_get_pipeline_runtime_settings_ignores_ambient_data_assets_for_smoke_variant(
    monkeypatch,
) -> None:
    from run_pipeline import get_pipeline_runtime_settings

    monkeypatch.setenv("DATA_ASSET_FULL", "churn-data-smoke")
    monkeypatch.setenv("DATA_VERSION", "2")
    monkeypatch.setenv("DATA_REFERENCE_ASSET_FULL", "churn-data-smoke")
    monkeypatch.setenv("DATA_REFERENCE_VERSION", "2")

    temp_dir = _make_temp_dir()
    try:
        config_env = temp_dir / "config.env"
        config_env.write_text(
            "\n".join(
                [
                    'AZURE_SUBSCRIPTION_ID="sub-1"',
                    'AZURE_RESOURCE_GROUP="rg-1"',
                    'AZURE_WORKSPACE_NAME="ws-1"',
                    'AZURE_COMPUTE_CLUSTER_NAME="cpu-cluster"',
                ]
            ),
            encoding="utf-8",
        )

        runtime = get_pipeline_runtime_settings(
            str(config_env),
            data_config_path=Path("configs/data_smoke_eval.yaml"),
        )

        assert runtime.current_data_asset_name == "churn-data-smoke-eval"
        assert runtime.current_data_asset_version == "1"
        assert runtime.reference_data_asset_name == "churn-data-smoke"
        assert runtime.reference_data_asset_version == "2"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_release_config_uses_asset_manifest_defaults(monkeypatch) -> None:
    from src.config.runtime import get_release_config

    for key in (
        "AML_DEPLOY_MODEL_NAME",
        "MODEL_NAME",
        "AML_ONLINE_ENDPOINT_NAME",
        "ENDPOINT_NAME",
        "AML_ONLINE_DEPLOYMENT_NAME",
        "DEPLOYMENT_NAME",
        "AML_ONLINE_INSTANCE_TYPE",
        "AML_ONLINE_INSTANCE_COUNT",
        "AML_ONLINE_SMOKE_PAYLOAD",
    ):
        monkeypatch.delenv(key, raising=False)

    temp_dir = _make_temp_dir()
    try:
        config_env = temp_dir / "config.env"
        config_env.write_text(
            "\n".join(
                [
                    'AZURE_SUBSCRIPTION_ID="sub-1"',
                    'AZURE_RESOURCE_GROUP="rg-1"',
                    'AZURE_WORKSPACE_NAME="ws-1"',
                ]
            ),
            encoding="utf-8",
        )

        release_config = get_release_config(str(config_env))

        assert release_config == {
            "model_name": "churn-prediction-model",
            "endpoint_name": "churn-endpoint",
            "deployment_name": "churn-deployment",
            "instance_type": "Standard_D2as_v4",
            "instance_count": "1",
            "smoke_payload": "sample-data.json",
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_release_config_allows_smoke_payload_override(monkeypatch) -> None:
    from src.config.runtime import get_release_config

    monkeypatch.setenv("AML_ONLINE_SMOKE_PAYLOAD", "samples/custom-payload.json")

    temp_dir = _make_temp_dir()
    try:
        config_env = temp_dir / "config.env"
        config_env.write_text(
            "\n".join(
                [
                    'AZURE_SUBSCRIPTION_ID="sub-1"',
                    'AZURE_RESOURCE_GROUP="rg-1"',
                    'AZURE_WORKSPACE_NAME="ws-1"',
                ]
            ),
            encoding="utf-8",
        )

        release_config = get_release_config(str(config_env))

        assert release_config["smoke_payload"] == "samples/custom-payload.json"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_environment_asset_config_builds_image_from_acr_name(monkeypatch) -> None:
    from src.config.runtime import get_environment_asset_config

    monkeypatch.delenv("AZURE_ACR_NAME", raising=False)

    temp_dir = _make_temp_dir()
    try:
        config_env = temp_dir / "config.env"
        config_env.write_text(
            "\n".join(
                [
                    'AZURE_SUBSCRIPTION_ID="sub-1"',
                    'AZURE_RESOURCE_GROUP="rg-1"',
                    'AZURE_WORKSPACE_NAME="ws-1"',
                    'AZURE_ACR_NAME="churnmlacr"',
                ]
            ),
            encoding="utf-8",
        )

        environment_config = get_environment_asset_config(str(config_env))

        assert environment_config == {
            "name": "bank-churn-env",
            "version": "1",
            "image_repository": "bank-churn",
            "image_tag": "1",
            "image": "churnmlacr.azurecr.io/bank-churn:1",
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_installs_noise_filters_and_submits_job_quietly(monkeypatch) -> None:
    """
    @proves fixed-train.submit-pipeline-job
    @proves fixed-train.use-quiet-submission
    """
    import run_pipeline

    captured: dict[str, object] = {}

    monkeypatch.setattr(run_pipeline, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run_pipeline,
        "get_release_config",
        lambda: {"model_name": "model-name"},
    )
    monkeypatch.setattr(
        run_pipeline,
        "get_pipeline_runtime_settings",
        lambda *_args, **_kwargs: run_pipeline.PipelineRuntimeSettings(
            compute_name="cpu-cluster",
            current_data_asset_name="current",
            current_data_asset_version="1",
            reference_data_asset_name="reference",
            reference_data_asset_version="2",
        ),
    )
    monkeypatch.setattr(
        run_pipeline,
        "load_pipeline_components",
        lambda _path: {"train": object(), "data_prep": object(), "validate_data": object(), "promote_model": object()},
    )
    monkeypatch.setattr(run_pipeline, "get_pipeline_validation_gate", lambda _path: False)
    monkeypatch.setattr(
        run_pipeline,
        "define_pipeline",
        lambda _components, _compute_name, gate_validation_before_prep=False: (
            lambda **kwargs: type(
                "FakePipelineJob",
                (),
                {"settings": type("Settings", (), {"force_rerun": False})(), "tags": {}, "kwargs": kwargs},
            )()
        ),
    )
    monkeypatch.setattr(run_pipeline, "get_pipeline_metadata", lambda _path: {})
    monkeypatch.setattr(run_pipeline, "build_local_file_input", lambda path: f"local::{path}")
    monkeypatch.setattr(
        run_pipeline,
        "build_pipeline_lineage_tags",
        lambda *_args, **_kwargs: {"data_asset": "current"},
    )
    monkeypatch.setattr(
        run_pipeline,
        "build_asset_input",
        lambda name, version: f"asset::{name}:{version}",
    )
    monkeypatch.setattr(
        run_pipeline,
        "write_registry_backed_baseline_file",
        lambda **kwargs: Path(kwargs["output_path"]).write_text("{}", encoding="utf-8"),
    )
    temp_dir = _make_temp_dir()
    monkeypatch.setattr(run_pipeline, "mkdtemp", lambda prefix: str(temp_dir))

    class FakeMlClient:
        jobs = object()

    monkeypatch.setattr(run_pipeline, "get_ml_client", lambda: FakeMlClient())
    monkeypatch.setattr(
        run_pipeline,
        "install_azure_console_noise_filters",
        lambda: captured.setdefault("filters_installed", True),
    )

    def fake_submit_job_quietly(jobs_client, job):
        captured["jobs_client"] = jobs_client
        captured["job"] = job
        return type(
            "ReturnedJob",
            (),
            {"name": "submitted-job", "studio_url": "https://studio/job"},
        )()

    monkeypatch.setattr(run_pipeline, "submit_job_quietly", fake_submit_job_quietly)
    monkeypatch.setattr("sys.argv", ["run_pipeline.py"])

    try:
        run_pipeline.main()

        assert captured["filters_installed"] is True
        assert captured["jobs_client"] is FakeMlClient.jobs
        assert getattr(captured["job"], "settings").force_rerun is True
        assert getattr(captured["job"], "tags")["data_asset"] == "current"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_registry_backed_baseline_file_uses_latest_approved_model_tags() -> None:
    from run_pipeline import write_registry_backed_baseline_file

    temp_dir = _make_temp_dir()
    try:
        output_path = temp_dir / "baseline.json"

        class FakeModels:
            def list(self, *, name: str):
                assert name == "churn-model"
                return [
                    {
                        "name": "churn-model",
                        "version": "1",
                        "tags": {"approval_status": "approved", "f1": "0.7", "roc_auc": "0.8"},
                    },
                    {
                        "name": "churn-model",
                        "version": "2",
                        "tags": {"approval_status": "approved", "primary_metric": "f1", "f1": "0.8", "roc_auc": "0.86"},
                    },
                ]

        class FakeClient:
            models = FakeModels()

        payload = write_registry_backed_baseline_file(
            ml_client=FakeClient(),
            model_name="churn-model",
            output_path=output_path,
        )

        assert payload["model_name"] == "churn-model"
        assert payload["model_version"] == "2"
        assert payload["f1"] == 0.8
        assert output_path.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_resolve_train_config_path_defaults_to_production_config() -> None:
    from run_pipeline import resolve_train_config_path

    assert resolve_train_config_path() == Path("configs/train.yaml")


def test_resolve_train_config_path_accepts_smoke_config() -> None:
    """
    @proves fixed-train.accept-train-config-path
    """
    from run_pipeline import resolve_train_config_path

    assert resolve_train_config_path("configs/train_smoke.yaml") == Path("configs/train_smoke.yaml")


def test_resolve_data_config_path_defaults_and_accepts_smoke_config() -> None:
    """
    @proves fixed-train.accept-data-config-path
    """
    from run_pipeline import resolve_data_config_path

    assert resolve_data_config_path() == Path("configs/data.yaml")
    assert resolve_data_config_path("configs/data_smoke.yaml") == Path("configs/data_smoke.yaml")


def test_build_pipeline_lineage_tags_includes_selected_assets_and_configs() -> None:
    """
    @proves fixed-train.attach-lineage-tags
    """
    from run_pipeline import PipelineRuntimeSettings, build_pipeline_lineage_tags

    runtime = PipelineRuntimeSettings(
        compute_name="cpu-cluster",
        current_data_asset_name="churn-current",
        current_data_asset_version="5",
        reference_data_asset_name="churn-reference",
        reference_data_asset_version="4",
    )

    tags = build_pipeline_lineage_tags(
        runtime,
        data_config_path=Path("configs/data_smoke.yaml"),
        train_config_path=Path("configs/train_smoke.yaml"),
    )

    assert tags["data_asset"] == "churn-current"
    assert tags["data_version"] == "5"
    assert tags["reference_data_asset"] == "churn-reference"
    assert tags["reference_data_version"] == "4"
    assert tags["data_config"] == "configs/data_smoke.yaml"
    assert tags["train_config"] == "configs/train_smoke.yaml"
    assert tags["data_prep_component"] == "data_prep:1"


def test_pipeline_components_accept_policy_config_inputs() -> None:
    """
    @proves fixed-train.share-validation-prep-contracts
    @proves fixed-train.share-training-artifact-vocabulary
    @proves fixed-train.emit-release-artifacts
    @proves fixed-train.emit-declared-manifest-folders
    @proves fixed-train.canonical-manifest-paths
    @proves fixed-train.execute-promotion-utility
    """
    import yaml

    project_root = Path(__file__).resolve().parents[1]
    validate_component = yaml.safe_load(
        (project_root / "aml" / "components" / "validate_data.yaml").read_text(encoding="utf-8")
    )
    data_prep_component = yaml.safe_load(
        (project_root / "aml" / "components" / "data_prep.yaml").read_text(encoding="utf-8")
    )
    train_component = yaml.safe_load((project_root / "aml" / "components" / "train.yaml").read_text(encoding="utf-8"))
    promote_component = yaml.safe_load(
        (project_root / "aml" / "components" / "promote_model.yaml").read_text(encoding="utf-8")
    )

    assert validate_component["inputs"]["config"]["type"] == "uri_file"
    assert "--config ${{inputs.config}}" in validate_component["command"]
    assert validate_component["outputs"]["validation_manifest"]["type"] == "uri_folder"
    assert "--manifest-output ${{outputs.validation_manifest}}" in validate_component["command"]
    assert data_prep_component["inputs"]["config"]["type"] == "uri_file"
    assert "--config ${{inputs.config}}" in data_prep_component["command"]
    assert data_prep_component["inputs"]["validation_summary"]["type"] == "uri_folder"
    assert data_prep_component["inputs"]["validation_summary"]["optional"] is True
    assert "$[[--validation-summary ${{inputs.validation_summary}}]]" in data_prep_component["command"]
    assert data_prep_component["outputs"]["data_prep_manifest"]["type"] == "uri_folder"
    assert "--manifest-output ${{outputs.data_prep_manifest}}" in data_prep_component["command"]
    assert train_component["inputs"]["config"]["type"] == "uri_file"
    assert "--config ${{inputs.config}}" in train_component["command"]
    assert train_component["outputs"]["parent_run_id"]["type"] == "uri_folder"
    assert train_component["outputs"]["candidate_metrics"]["type"] == "uri_folder"
    assert train_component["outputs"]["train_manifest"]["type"] == "uri_folder"
    assert "--manifest-output ${{outputs.train_manifest}}" in train_component["command"]
    assert promote_component["inputs"]["config"]["type"] == "uri_file"
    assert promote_component["inputs"]["candidate_metrics"]["type"] == "uri_folder"
    assert "--config ${{inputs.config}}" in promote_component["command"]
    assert promote_component["outputs"]["promotion_manifest"]["type"] == "uri_folder"
    assert "--manifest-output ${{outputs.promotion_manifest}}" in promote_component["command"]


def test_build_submission_messages_are_ascii_safe() -> None:
    from run_pipeline import build_submission_messages

    messages = build_submission_messages(
        job_name="bold_eagle_nlg5yg49qw",
        studio_url="https://ml.azure.com/runs/bold_eagle_nlg5yg49qw",
    )

    assert messages == [
        "OK Job submitted: bold_eagle_nlg5yg49qw",
        "  View in Azure ML Studio: https://ml.azure.com/runs/bold_eagle_nlg5yg49qw",
    ]
    assert all(message.isascii() for message in messages)


def test_get_pipeline_metadata_reads_train_config_fields() -> None:
    """
    @proves fixed-train.apply-train-metadata
    """
    from run_pipeline import get_pipeline_metadata

    temp_dir = _make_temp_dir()
    try:
        train_config = temp_dir / "train.yaml"
        train_config.write_text(
            "\n".join(
                [
                    "training:",
                    '  experiment_name: "train-owned-experiment"',
                    '  display_name: "train-display"',
                ]
            ),
            encoding="utf-8",
        )

        metadata = get_pipeline_metadata(train_config)

        assert metadata["experiment_name"] == "train-owned-experiment"
        assert metadata["display_name"] == "train-display"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
