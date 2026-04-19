"""
@meta
type: test
scope: unit
domain: asset-management
covers:
  - Lightweight asset manifest loading
  - Component and environment lineage tag helpers
  - Asset/config lineage tag construction
excludes:
  - Real Azure ML asset or registry operations
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_asset_manifest_reads_canonical_asset_names() -> None:
    from src.config.assets import load_asset_manifest

    manifest = load_asset_manifest(PROJECT_ROOT / "configs" / "assets.yaml")

    assert manifest["data_assets"]["production"]["name"] == "churn-data"
    assert manifest["data_assets"]["smoke"]["name"] == "churn-data-smoke"
    assert manifest["model"]["name"] == "churn-prediction-model"


def test_component_and_environment_identity_helpers() -> None:
    from src.config.assets import component_identity, environment_identity, load_asset_manifest

    manifest = load_asset_manifest(PROJECT_ROOT / "configs" / "assets.yaml")

    assert component_identity(manifest, "data_prep") == "data_prep:1"
    assert component_identity(manifest, "train") == "train_model:1"
    assert environment_identity(manifest) == "bank-churn-env:1"


def test_environment_image_defaults_come_from_asset_manifest() -> None:
    from src.config.assets import environment_image_defaults, load_asset_manifest

    manifest = load_asset_manifest(PROJECT_ROOT / "configs" / "assets.yaml")

    assert environment_image_defaults(manifest) == {
        "image_repository": "bank-churn",
        "image_tag": "1",
    }


def test_build_asset_lineage_tags_includes_configs_components_and_git_fallback() -> None:
    """
    @proves fixed-train.attach-lineage-tags
    @proves online-deploy.stamp-approved-registered-models-lightweight-data-config-component
    """
    from src.config.assets import build_asset_lineage_tags, load_asset_manifest

    tags = build_asset_lineage_tags(
        current_data_asset_name="churn-current",
        current_data_asset_version="7",
        reference_data_asset_name="churn-reference",
        reference_data_asset_version="3",
        data_config_path="configs\\data_smoke.yaml",
        train_config_path="configs\\train_smoke.yaml",
        manifest=load_asset_manifest(PROJECT_ROOT / "configs" / "assets.yaml"),
        git_commit="test-sha",
    )

    assert tags["data_asset"] == "churn-current"
    assert tags["data_version"] == "7"
    assert tags["reference_data_asset"] == "churn-reference"
    assert tags["reference_data_version"] == "3"
    assert tags["data_config"] == "configs/data_smoke.yaml"
    assert tags["train_config"] == "configs/train_smoke.yaml"
    assert tags["validate_component"] == "validate_data:1"
    assert tags["data_prep_component"] == "data_prep:1"
    assert tags["environment"] == "bank-churn-env:1"
    assert tags["git_commit"] == "test-sha"


def test_load_asset_manifest_falls_back_when_manifest_is_missing() -> None:
    from src.config.assets import component_identity, load_asset_manifest

    manifest = load_asset_manifest(PROJECT_ROOT / "configs" / "does-not-exist.yaml")

    assert component_identity(manifest, "promote") == "promote_model:1"


def test_deployment_inference_capture_settings_come_from_asset_manifest() -> None:
    """
    @proves online-deploy.invoke-endpoint-deployment-smoke-payload-path-configs-assets
    """
    from src.config.assets import (
        deployment_inference_capture_settings,
        load_asset_manifest,
    )

    manifest = load_asset_manifest(PROJECT_ROOT / "configs" / "assets.yaml")

    settings = deployment_inference_capture_settings(manifest)

    assert settings.enabled is False
    assert settings.mode == "workspaceblobstore_jsonl"
    assert settings.sample_rate == 1.0
    assert settings.max_rows_per_request == 5
    assert settings.capture_inputs is True
    assert settings.capture_outputs is True
    assert settings.redact_inputs is False
    assert settings.output_path == "monitoring/inference_capture"
    assert (
        settings.storage_connection_string_env
        == "INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING"
    )
    assert settings.storage_container_env == "INFERENCE_CAPTURE_STORAGE_CONTAINER"


def test_repo_owned_online_inference_capture_settings_downgrades_collector_mode() -> None:
    from src.config.assets import repo_owned_online_inference_capture_settings

    settings = repo_owned_online_inference_capture_settings(
        {
            "deployment": {
                "inference_capture": {
                    "enabled": True,
                    "mode": "azureml_data_collector",
                    "inputs_name": "model_inputs",
                    "outputs_name": "model_outputs",
                }
            }
        }
    )

    assert settings.enabled is False
    assert settings.mode == "release_evidence_only"
    assert (
        settings.storage_connection_string_env
        == "INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING"
    )
    assert settings.storage_container_env == "INFERENCE_CAPTURE_STORAGE_CONTAINER"
