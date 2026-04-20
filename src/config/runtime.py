"""Central runtime configuration ownership for Azure, training, and promotion.

@meta
name: runtime
type: module
domain: config
responsibility:
  - Provide config behavior for `src/config/runtime.py`.
inputs: []
outputs: []
tags:
  - config
capabilities:
  - fixed-train.apply-train-metadata
  - fixed-train.expose-promotion-thresholds
lifecycle:
  status: active
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, cast

from dotenv import dotenv_values

from utils.config_loader import get_config_value, load_config
from utils.env_loader import get_env_var, load_env_file
from utils.type_utils import parse_bool

from .assets import load_asset_manifest


DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_TRAIN_CONFIG = DEFAULT_CONFIGS_DIR / "train.yaml"
DEFAULT_MLFLOW_CONFIG = DEFAULT_CONFIGS_DIR / "mlflow.yaml"
DEFAULT_ASSET_CONFIG = DEFAULT_CONFIGS_DIR / "assets.yaml"


@dataclass(frozen=True)
class TrainingRuntimeDefaults:
    experiment_name: str
    display_name: str | None
    class_weight: str
    random_state: int
    use_smote: bool


@dataclass(frozen=True)
class PromotionConfig:
    primary_metric: str
    minimum_improvement: float
    minimum_candidate_score: float


def _load_env_overrides(config_path: str | None) -> dict[str, str]:
    if not config_path:
        return {}

    values = dotenv_values(config_path)
    return {
        key: str(value)
        for key, value in values.items()
        if value is not None
    }


def _resolve_env_var(
    key: str,
    *,
    config_path: str | None,
    default: str | None = None,
    required: bool = False,
) -> str | None:
    overrides = _load_env_overrides(config_path)
    if key in overrides:
        return overrides[key]

    load_env_file(config_path)
    return get_env_var(key, default=default, required=required)


def _require_env_var(key: str, *, config_path: str | None) -> str:
    value = _resolve_env_var(key, config_path=config_path, required=True)
    return cast(str, value)


def _has_env_value(key: str, *, config_path: str | None) -> bool:
    overrides = _load_env_overrides(config_path)
    if key in overrides:
        return True

    load_env_file(config_path)
    return get_env_var(key) is not None


def _load_yaml_mapping(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        return {}
    return load_config(str(config_path)) or {}


def _asset_manifest_value(*keys: str, default: str) -> str:
    value: object = load_asset_manifest(DEFAULT_ASSET_CONFIG)
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return str(value) if value is not None else default


def _resolve_train_config_path(config_path: Path | str | None) -> Path:
    return Path(config_path) if config_path else DEFAULT_TRAIN_CONFIG


def load_azure_config(config_path: str | None = None) -> dict[str, str]:
    """Load Azure ML workspace coordinates from env-backed config."""
    return {
        "subscription_id": _require_env_var("AZURE_SUBSCRIPTION_ID", config_path=config_path),
        "resource_group": _require_env_var("AZURE_RESOURCE_GROUP", config_path=config_path),
        "workspace_name": _require_env_var("AZURE_WORKSPACE_NAME", config_path=config_path),
    }


def get_data_asset_config(config_path: str | None = None) -> dict[str, str]:
    """Resolve the current training data asset coordinates."""
    default_name = _asset_manifest_value(
        "data_assets",
        "production",
        "name",
        default="churn-data",
    )
    default_version = _asset_manifest_value(
        "data_assets",
        "production",
        "version",
        default="1",
    )
    return {
        "data_asset_name": cast(
            str,
            _resolve_env_var(
                "DATA_ASSET_FULL",
                config_path=config_path,
                default=default_name,
            ),
        ),
        "data_asset_version": cast(
            str,
            _resolve_env_var(
                "DATA_VERSION",
                config_path=config_path,
                default=default_version,
            ),
        ),
    }


def get_reference_data_asset_config(config_path: str | None = None) -> dict[str, str]:
    """Resolve the optional reference asset, falling back to the current asset."""
    current_asset = get_data_asset_config(config_path)
    current_asset_overridden = _has_env_value(
        "DATA_ASSET_FULL",
        config_path=config_path,
    ) or _has_env_value("DATA_VERSION", config_path=config_path)
    reference_asset_overridden = _has_env_value(
        "DATA_REFERENCE_ASSET_FULL",
        config_path=config_path,
    ) or _has_env_value("DATA_REFERENCE_VERSION", config_path=config_path)
    default_name = _asset_manifest_value(
        "data_assets",
        "reference",
        "name",
        default=current_asset["data_asset_name"],
    )
    default_version = _asset_manifest_value(
        "data_assets",
        "reference",
        "version",
        default=current_asset["data_asset_version"],
    )
    if current_asset_overridden and not reference_asset_overridden:
        default_name = current_asset["data_asset_name"]
        default_version = current_asset["data_asset_version"]

    reference_asset_name = _resolve_env_var(
        "DATA_REFERENCE_ASSET_FULL",
        config_path=config_path,
        default=default_name,
    )
    reference_asset_version = _resolve_env_var(
        "DATA_REFERENCE_VERSION",
        config_path=config_path,
        default=default_version,
    )
    return {
        "data_asset_name": cast(str, reference_asset_name),
        "data_asset_version": cast(str, reference_asset_version),
    }


def get_pipeline_compute_name(config_path: str | None = None) -> str:
    """Resolve the Azure ML compute target used by orchestration scripts."""
    return (
        _resolve_env_var("AZURE_PIPELINE_COMPUTE", config_path=config_path)
        or _resolve_env_var("AZURE_COMPUTE_CLUSTER_NAME", config_path=config_path)
        or "cpu-cluster"
    )


def get_release_config(config_path: str | None = None) -> dict[str, str]:
    """Resolve model-registration and endpoint-deployment settings."""
    default_model_name = _asset_manifest_value(
        "model",
        "name",
        default="churn-prediction-model",
    )
    default_endpoint_name = _asset_manifest_value(
        "deployment",
        "endpoint_name",
        default="churn-endpoint",
    )
    default_deployment_name = _asset_manifest_value(
        "deployment",
        "deployment_name",
        default="blue",
    )
    default_instance_type = _asset_manifest_value(
        "deployment",
        "instance_type",
        default="Standard_D2as_v4",
    )
    default_instance_count = _asset_manifest_value(
        "deployment",
        "instance_count",
        default="1",
    )
    default_smoke_payload = _asset_manifest_value(
        "deployment",
        "smoke_payload",
        default="sample-data.json",
    )
    return {
        "model_name": (
            _resolve_env_var("AML_DEPLOY_MODEL_NAME", config_path=config_path)
            or _resolve_env_var("MODEL_NAME", config_path=config_path)
            or default_model_name
        ),
        "endpoint_name": (
            _resolve_env_var("AML_ONLINE_ENDPOINT_NAME", config_path=config_path)
            or _resolve_env_var("ENDPOINT_NAME", config_path=config_path)
            or default_endpoint_name
        ),
        "deployment_name": (
            _resolve_env_var("AML_ONLINE_DEPLOYMENT_NAME", config_path=config_path)
            or _resolve_env_var("DEPLOYMENT_NAME", config_path=config_path)
            or default_deployment_name
        ),
        "instance_type": (
            _resolve_env_var("AML_ONLINE_INSTANCE_TYPE", config_path=config_path)
            or default_instance_type
        ),
        "instance_count": (
            _resolve_env_var("AML_ONLINE_INSTANCE_COUNT", config_path=config_path)
            or default_instance_count
        ),
        "smoke_payload": (
            _resolve_env_var("AML_ONLINE_SMOKE_PAYLOAD", config_path=config_path)
            or default_smoke_payload
        ),
    }


def get_environment_asset_config(config_path: str | None = None) -> dict[str, str]:
    """Resolve the AML environment asset identity and container image URI."""
    manifest = load_asset_manifest(DEFAULT_ASSET_CONFIG)
    raw_environment_config = manifest.get("environment", {})
    environment_config = (
        cast(Mapping[str, object], raw_environment_config)
        if isinstance(raw_environment_config, dict)
        else {}
    )
    default_name = str(environment_config.get("name", "bank-churn-env"))
    default_version = str(environment_config.get("version", "1"))
    default_repository = str(environment_config.get("image_repository", "bank-churn"))
    default_tag = str(environment_config.get("image_tag", "1"))

    environment_name = default_name
    environment_version = default_version
    image_repository = default_repository
    image_tag = default_tag

    explicit_image = str(environment_config.get("image", "")).strip()
    if explicit_image:
        image_uri = explicit_image
    else:
        acr_name = _require_env_var("AZURE_ACR_NAME", config_path=config_path)
        image_uri = f"{acr_name}.azurecr.io/{image_repository}:{image_tag}"

    return {
        "name": environment_name,
        "version": environment_version,
        "image_repository": image_repository,
        "image_tag": image_tag,
        "image": image_uri,
    }


def load_training_runtime_defaults(
    config_path: Path | str | None = None,
) -> TrainingRuntimeDefaults:
    """Load canonical training defaults from the training config surface."""
    train_config_path = _resolve_train_config_path(config_path)
    training_config = get_config_value(
        _load_yaml_mapping(train_config_path),
        "training",
        {},
    ) or {}

    mlflow_config_path = train_config_path.parent / "mlflow.yaml"
    if not mlflow_config_path.exists():
        mlflow_config_path = DEFAULT_MLFLOW_CONFIG
    mlflow_config = get_config_value(
        _load_yaml_mapping(mlflow_config_path),
        "mlflow",
        {},
    ) or {}

    experiment_name = str(
        training_config.get("experiment_name")
        or mlflow_config.get("experiment_name")
        or "churn-prediction"
    )
    display_name_raw = training_config.get("display_name")
    display_name = str(display_name_raw) if display_name_raw is not None else None

    return TrainingRuntimeDefaults(
        experiment_name=experiment_name,
        display_name=display_name,
        class_weight=str(training_config.get("class_weight", "balanced")),
        random_state=int(training_config.get("random_state", 42)),
        use_smote=parse_bool(training_config.get("use_smote", True), default=True),
    )


def load_promotion_config(config_path: Path | str | None = None) -> PromotionConfig:
    """Load promotion thresholds from the training config surface."""
    train_config_path = _resolve_train_config_path(config_path)
    promotion_config = get_config_value(
        _load_yaml_mapping(train_config_path),
        "promotion",
        {},
    ) or {}

    return PromotionConfig(
        primary_metric=str(promotion_config.get("primary_metric", "f1")),
        minimum_improvement=float(promotion_config.get("minimum_improvement", 0.0)),
        minimum_candidate_score=float(
            promotion_config.get("minimum_candidate_score", 0.0)
        ),
    )
