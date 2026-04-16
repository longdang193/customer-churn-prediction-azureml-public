"""
@meta
name: run_pipeline
type: script
domain: training-orchestration
responsibility:
  - Resolve environment-backed runtime settings for the fixed training pipeline.
  - Submit the validation-aware Azure ML training pipeline.
  - Execute the promotion gate using candidate metrics and a baseline metrics artifact.
inputs:
  - config.env
  - configs/train.yaml
  - aml/components/*.yaml
outputs:
  - Azure ML pipeline job submission
  - Pipeline outputs for validation, candidate metrics, and promotion decision artifacts
tags:
  - azure-ml
  - training
  - orchestration
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from tempfile import mkdtemp
from typing import Any, Dict

from dotenv import load_dotenv

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.azureml import (
    build_asset_input,
    build_local_or_uri_folder_input,
    build_local_file_input,
    get_ml_client,
    install_azure_console_noise_filters,
    submit_job_quietly,
    write_registry_backed_baseline_file as write_registry_backed_baseline_artifact,
)
from src.config.assets import build_asset_lineage_tags, load_asset_manifest
from src.config.runtime import (
    get_data_asset_config,
    get_pipeline_compute_name,
    get_reference_data_asset_config,
    get_release_config,
)
from src.utils.config_loader import load_config


@dataclass(frozen=True)
class PipelineRuntimeSettings:
    compute_name: str
    current_data_asset_name: str
    current_data_asset_version: str
    reference_data_asset_name: str
    reference_data_asset_version: str


BASE_SMOKE_ASSET_KEY = "smoke"


def _is_smoke_data_config(data_config_path: Path | None) -> bool:
    return data_config_path is not None and "smoke" in data_config_path.stem.lower()


def _resolve_smoke_asset_key(data_config_path: Path | None) -> str:
    if data_config_path is None:
        return BASE_SMOKE_ASSET_KEY

    stem = data_config_path.stem.lower()
    if stem.startswith("data_"):
        candidate_key = stem[len("data_") :]
        asset_manifest = load_asset_manifest()
        data_assets = asset_manifest.get("data_assets", {}) or {}
        if candidate_key in data_assets:
            return candidate_key
    return BASE_SMOKE_ASSET_KEY


def _resolve_smoke_reference_asset_key(current_asset_key: str) -> str:
    """Use the base smoke fixture as reference for named smoke variants."""
    return BASE_SMOKE_ASSET_KEY


def get_pipeline_validation_gate(data_config_path: Path | None) -> bool:
    """Resolve whether validation should gate downstream preparation work."""
    if data_config_path is None or not data_config_path.exists():
        return False

    config = load_config(str(data_config_path)) or {}
    validation_cfg = config.get("validation", {}) or {}
    return bool(validation_cfg.get("gate_before_data_prep", False))


def get_pipeline_runtime_settings(
    config_path: str | None = None,
    *,
    data_config_path: Path | None = None,
) -> PipelineRuntimeSettings:
    """Resolve compute and asset settings for the fixed training pipeline."""
    current_asset = get_data_asset_config(config_path)
    reference_asset = get_reference_data_asset_config(config_path)
    if _is_smoke_data_config(data_config_path):
        asset_manifest = load_asset_manifest()
        data_assets = asset_manifest.get("data_assets", {}) or {}
        smoke_asset_key = _resolve_smoke_asset_key(data_config_path)
        smoke_asset = data_assets.get(smoke_asset_key, {}) or {}
        smoke_reference_asset = (
            data_assets.get(_resolve_smoke_reference_asset_key(smoke_asset_key), {}) or {}
        )
        smoke_asset_name = str(smoke_asset.get("name", current_asset["data_asset_name"]))
        smoke_asset_version = str(
            smoke_asset.get("version", current_asset["data_asset_version"])
        )
        smoke_reference_asset_name = str(
            smoke_reference_asset.get("name", smoke_asset_name)
        )
        smoke_reference_asset_version = str(
            smoke_reference_asset.get("version", smoke_asset_version)
        )
        current_asset = {
            "data_asset_name": smoke_asset_name,
            "data_asset_version": smoke_asset_version,
        }
        reference_asset = {
            "data_asset_name": smoke_reference_asset_name,
            "data_asset_version": smoke_reference_asset_version,
        }

    return PipelineRuntimeSettings(
        compute_name=get_pipeline_compute_name(config_path),
        current_data_asset_name=current_asset["data_asset_name"],
        current_data_asset_version=current_asset["data_asset_version"],
        reference_data_asset_name=reference_asset["data_asset_name"],
        reference_data_asset_version=reference_asset["data_asset_version"],
    )


def load_pipeline_components(components_dir: Path) -> Dict[str, Any]:
    """Load AML components from the specified directory."""
    from azure.ai.ml import load_component

    return {
        "validate_data": load_component(source=str(components_dir / "validate_data.yaml")),
        "data_prep": load_component(source=str(components_dir / "data_prep.yaml")),
        "train": load_component(source=str(components_dir / "train.yaml")),
        "promote_model": load_component(source=str(components_dir / "promote_model.yaml")),
    }


def define_pipeline(
    components: Dict[str, Any],
    compute_name: str,
    *,
    gate_validation_before_prep: bool = False,
):
    """Define the Azure ML pipeline using the loaded components."""
    from azure.ai.ml import dsl

    @dsl.pipeline(
        compute=compute_name,
        description="Validation-aware training pipeline for bank churn prediction",
    )
    def churn_prediction_pipeline(
        current_raw_data,
        reference_raw_data,
        baseline_metrics,
        data_config,
        train_config,
    ):
        validate_job = components["validate_data"](
            current_data=current_raw_data,
            reference_data=reference_raw_data,
            config=data_config,
        )
        data_prep_job = components["data_prep"](
            raw_data=current_raw_data,
            config=data_config,
            validation_summary=validate_job.outputs.validation_summary
            if gate_validation_before_prep
            else None,
        )
        train_job = components["train"](
            processed_data=data_prep_job.outputs.processed_data,
            config=train_config,
        )
        promote_job = components["promote_model"](
            candidate_metrics=train_job.outputs.candidate_metrics,
            baseline_metrics=baseline_metrics,
            config=train_config,
        )
        return {
            "validation_report": validate_job.outputs.validation_report,
            "validation_summary": validate_job.outputs.validation_summary,
            "validation_manifest": validate_job.outputs.validation_manifest,
            "data_prep_manifest": data_prep_job.outputs.data_prep_manifest,
            "model_output": train_job.outputs.model_output,
            "mlflow_model": train_job.outputs.mlflow_model,
            "train_manifest": train_job.outputs.train_manifest,
            "parent_run_id": train_job.outputs.parent_run_id,
            "candidate_metrics": train_job.outputs.candidate_metrics,
            "promotion_decision": promote_job.outputs.promotion_decision,
            "promotion_manifest": promote_job.outputs.promotion_manifest,
        }

    return churn_prediction_pipeline


def get_pipeline_metadata(config_path: Path) -> Dict[str, Any]:
    """Extract optional experiment and display names from the training config."""
    if not config_path.exists():
        return {}

    config = load_config(str(config_path)) or {}
    training_cfg = config.get("training", {}) or {}
    return {
        "experiment_name": training_cfg.get("experiment_name"),
        "display_name": training_cfg.get("display_name"),
    }


def resolve_train_config_path(config_path: str | None = None) -> Path:
    """Resolve the training config passed to AML train and promotion components."""
    return Path(config_path or "configs/train.yaml")


def resolve_data_config_path(config_path: str | None = None) -> Path:
    """Resolve the data config passed to AML validation and data prep components."""
    return Path(config_path or "configs/data.yaml")


def resolve_pipeline_data_inputs(
    *,
    current_data_override: str | None,
    reference_data_override: str | None,
    runtime: PipelineRuntimeSettings,
) -> tuple[object, object, list[Path]]:
    """Resolve pipeline data inputs from explicit overrides or runtime assets."""
    has_current_override = bool(current_data_override)
    has_reference_override = bool(reference_data_override)
    if has_current_override != has_reference_override:
        raise ValueError(
            "current_data_override and reference_data_override must be provided together"
        )

    if has_current_override and has_reference_override:
        assert current_data_override is not None
        assert reference_data_override is not None
        current_input, current_cleanup_dir = build_local_or_uri_folder_input(
            current_data_override
        )
        reference_input, reference_cleanup_dir = build_local_or_uri_folder_input(
            reference_data_override
        )
        return (
            current_input,
            reference_input,
            [path for path in (current_cleanup_dir, reference_cleanup_dir) if path is not None],
        )

    return (
        build_asset_input(
            runtime.current_data_asset_name,
            runtime.current_data_asset_version,
        ),
        build_asset_input(
            runtime.reference_data_asset_name,
            runtime.reference_data_asset_version,
        ),
        [],
    )


def build_pipeline_lineage_tags(
    runtime: PipelineRuntimeSettings,
    *,
    data_config_path: Path,
    train_config_path: Path,
    asset_manifest_path: str | None = None,
) -> dict[str, str]:
    """Build lineage tags for submitted Azure ML pipeline jobs."""
    return build_asset_lineage_tags(
        current_data_asset_name=runtime.current_data_asset_name,
        current_data_asset_version=runtime.current_data_asset_version,
        reference_data_asset_name=runtime.reference_data_asset_name,
        reference_data_asset_version=runtime.reference_data_asset_version,
        data_config_path=data_config_path,
        train_config_path=train_config_path,
        manifest=load_asset_manifest(asset_manifest_path),
    )


def write_registry_backed_baseline_file(ml_client: object, model_name: str, output_path: Path):
    """Resolve the latest approved registry model into a baseline metrics file."""
    return write_registry_backed_baseline_artifact(
        ml_client,
        model_name=model_name,
        output_path=output_path,
    )


def build_submission_messages(*, job_name: str, studio_url: str) -> list[str]:
    """Return ASCII-safe submission messages for local consoles."""
    return [
        f"OK Job submitted: {job_name}",
        f"  View in Azure ML Studio: {studio_url}",
    ]


def main() -> None:
    """Main function to define and run the Azure ML pipeline."""
    parser = argparse.ArgumentParser(
        description="Submit the validation-aware Azure ML training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-config",
        default=None,
        help="Data config file to pass to validation and data-prep components.",
    )
    parser.add_argument(
        "--train-config",
        default=None,
        help="Training config file to pass to train and promotion components.",
    )
    parser.add_argument(
        "--current-data-override",
        default=None,
        help="Explicit current dataset path or AML URI for smoke retraining handoffs.",
    )
    parser.add_argument(
        "--reference-data-override",
        default=None,
        help="Explicit reference dataset path or AML URI for smoke retraining handoffs.",
    )
    args = parser.parse_args()

    load_dotenv("config.env")
    install_azure_console_noise_filters()
    release_config = get_release_config()
    data_config_path = resolve_data_config_path(args.data_config)
    train_config_path = resolve_train_config_path(args.train_config)
    runtime = get_pipeline_runtime_settings(
        "config.env",
        data_config_path=data_config_path,
    )
    ml_client = get_ml_client()

    components = load_pipeline_components(Path("aml/components"))
    pipeline = define_pipeline(
        components,
        runtime.compute_name,
        gate_validation_before_prep=get_pipeline_validation_gate(data_config_path),
    )
    pipeline_metadata = get_pipeline_metadata(train_config_path)
    data_config_input = build_local_file_input(data_config_path)
    train_config_input = build_local_file_input(train_config_path)
    lineage_tags = build_pipeline_lineage_tags(
        runtime,
        data_config_path=data_config_path,
        train_config_path=train_config_path,
    )

    current_input, reference_input, override_cleanup_dirs = resolve_pipeline_data_inputs(
        current_data_override=args.current_data_override,
        reference_data_override=args.reference_data_override,
        runtime=runtime,
    )
    temp_dir = Path(mkdtemp(prefix="aml-promotion-baseline-"))
    try:
        baseline_metrics_path = temp_dir / "baseline_metrics.json"
        write_registry_backed_baseline_file(
            ml_client=ml_client,
            model_name=release_config["model_name"],
            output_path=baseline_metrics_path,
        )
        baseline_metrics_input = build_local_file_input(baseline_metrics_path)
        pipeline_job = pipeline(
            current_raw_data=current_input,
            reference_raw_data=reference_input,
            baseline_metrics=baseline_metrics_input,
            data_config=data_config_input,
            train_config=train_config_input,
        )
        pipeline_job.settings.force_rerun = True
        pipeline_job.tags = {
            **(getattr(pipeline_job, "tags", None) or {}),
            **lineage_tags,
        }

        experiment_name = pipeline_metadata.get("experiment_name")
        display_name = pipeline_metadata.get("display_name")
        if experiment_name:
            pipeline_job.experiment_name = experiment_name
        if display_name:
            pipeline_job.display_name = display_name

        returned_job = submit_job_quietly(ml_client.jobs, pipeline_job)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        for cleanup_dir in override_cleanup_dirs:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    for message in build_submission_messages(
        job_name=returned_job.name,
        studio_url=returned_job.studio_url,
    ):
        print(message)


if __name__ == "__main__":
    main()
