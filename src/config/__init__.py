"""Runtime configuration helpers for orchestration and training flows.

@meta
name: config
type: module
domain: config
responsibility:
  - Provide config behavior for `src/config/__init__.py`.
inputs: []
outputs: []
tags:
  - config
lifecycle:
  status: active
"""

from .assets import (
    build_asset_lineage_tags,
    build_component_lineage_tags,
    component_identity,
    environment_identity,
    get_git_commit,
    load_asset_manifest,
)
from .runtime import (
    PromotionConfig,
    TrainingRuntimeDefaults,
    get_data_asset_config,
    get_environment_asset_config,
    get_pipeline_compute_name,
    get_reference_data_asset_config,
    get_release_config,
    load_azure_config,
    load_promotion_config,
    load_training_runtime_defaults,
)

__all__ = [
    "build_asset_lineage_tags",
    "build_component_lineage_tags",
    "component_identity",
    "environment_identity",
    "get_git_commit",
    "load_asset_manifest",
    "PromotionConfig",
    "TrainingRuntimeDefaults",
    "get_data_asset_config",
    "get_environment_asset_config",
    "get_pipeline_compute_name",
    "get_reference_data_asset_config",
    "get_release_config",
    "load_azure_config",
    "load_promotion_config",
    "load_training_runtime_defaults",
]
