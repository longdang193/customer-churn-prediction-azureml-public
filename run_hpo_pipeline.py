"""
@meta
name: run_hpo_pipeline
type: script
domain: hpo
responsibility:
  - Submit an end-to-end Azure ML HPO pipeline from raw-data and HPO config inputs.
  - Reuse validation and data-prep stages before fan-out into per-model sweeps.
  - Emit a summary surface for the HPO run.
inputs:
  - config.env
  - configs/data.yaml
  - configs/hpo.yaml
outputs:
  - Azure ML HPO pipeline job submission
  - Pipeline outputs for validation, prep, and HPO summary artifacts
tags:
  - azure-ml
  - hpo
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
from typing import Any

from dotenv import load_dotenv

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import hpo_utils
from run_hpo import (
    SweepSpec,
    build_early_stopping_policy,
    build_model_sweep_specs,
    build_submission_messages,
    build_sweep_search_space,
    configure_sweep_limits,
    normalize_sweep_goal,
)
from run_pipeline import (
    PipelineRuntimeSettings,
    get_pipeline_runtime_settings,
    get_pipeline_validation_gate,
    resolve_data_config_path,
    resolve_train_config_path,
)
from src.azureml import (
    build_asset_input,
    build_local_or_uri_folder_input,
    build_local_file_input,
    get_ml_client,
    install_azure_console_noise_filters,
    submit_job_quietly,
)
from src.config.assets import (
    build_component_lineage_tags,
    get_git_commit,
    load_asset_manifest,
)


@dataclass(frozen=True)
class HPOPipelineMetadata:
    experiment_name: str | None
    display_name: str | None
    primary_metric: str


@dataclass(frozen=True)
class HPODataInputOverrides:
    current_data_override: str | None
    reference_data_override: str | None


def resolve_hpo_config_path(config_path: str | None = None) -> Path:
    """Resolve the HPO config passed to the HPO pipeline."""
    return Path(config_path or "configs/hpo.yaml")


def get_hpo_pipeline_metadata(config_path: Path) -> dict[str, str | None]:
    """Extract experiment and display metadata from the HPO config."""
    if not config_path.exists():
        return {
            "experiment_name": None,
            "display_name": None,
            "primary_metric": "f1",
        }

    config = hpo_utils.load_hpo_config(config_path)
    return {
        "experiment_name": (
            str(config["experiment_name"]) if config.get("experiment_name") is not None else None
        ),
        "display_name": (
            str(config["sweep_display_name"]) if config.get("sweep_display_name") is not None else None
        ),
        "primary_metric": str(config.get("metric", "f1")),
    }


def build_hpo_lineage_tags(
    runtime: PipelineRuntimeSettings,
    *,
    data_config_path: Path,
    hpo_config_path: Path,
    asset_manifest_path: str | None = None,
) -> dict[str, str]:
    """Build lineage tags for HPO pipeline submissions."""
    manifest = load_asset_manifest(asset_manifest_path)
    tags = {
        "data_asset": runtime.current_data_asset_name,
        "data_version": runtime.current_data_asset_version,
        "reference_data_asset": runtime.reference_data_asset_name,
        "reference_data_version": runtime.reference_data_asset_version,
        "data_config": data_config_path.as_posix(),
        "hpo_config": hpo_config_path.as_posix(),
        **build_component_lineage_tags(manifest),
        "git_commit": get_git_commit(),
    }
    return {key: str(value) for key, value in tags.items()}


def _resolve_hpo_data_inputs(
    runtime: PipelineRuntimeSettings,
    *,
    overrides: HPODataInputOverrides,
) -> tuple[Any, Any, list[Path]]:
    """Resolve current/reference raw-data inputs for the HPO pipeline."""
    if (overrides.current_data_override is None) != (overrides.reference_data_override is None):
        raise ValueError(
            "Both --current-data-override and --reference-data-override must be provided together."
        )
    if (
        overrides.current_data_override is not None
        and overrides.reference_data_override is not None
    ):
        current_input, current_cleanup_dir = build_local_or_uri_folder_input(
            overrides.current_data_override
        )
        reference_input, reference_cleanup_dir = build_local_or_uri_folder_input(
            overrides.reference_data_override
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


def load_hpo_pipeline_components(components_dir: Path) -> dict[str, Any]:
    """Load AML components used by the HPO pipeline."""
    from azure.ai.ml import load_component

    return {
        "validate_data": load_component(source=str(components_dir / "validate_data.yaml")),
        "data_prep": load_component(source=str(components_dir / "data_prep.yaml")),
        "hpo_trial": load_component(source=str(components_dir / "hpo_trial.yaml")),
        "collect_hpo_results": load_component(
            source=str(components_dir / "collect_hpo_results.yaml")
        ),
        "materialize_hpo_winner": load_component(
            source=str(components_dir / "materialize_hpo_winner.yaml")
        ),
    }


def _build_hpo_sweep_step(
    hpo_trial_component: Any,
    *,
    spec: SweepSpec,
    processed_data: Any,
    train_config: Any,
    hpo_config: Any,
    compute_name: str,
    early_stopping_policy: Any | None,
) -> Any:
    trial_node = hpo_trial_component(
        processed_data=processed_data,
        config=train_config,
        hpo_config=hpo_config,
        model_type=spec.model_name,
        **build_sweep_search_space(spec),
    )
    sweep_step = trial_node.sweep(
        primary_metric=spec.metric,
        goal=normalize_sweep_goal(spec.mode),
        sampling_algorithm=spec.sampling_algorithm,
        compute=compute_name,
    )
    configure_sweep_limits(sweep_step, spec)
    if early_stopping_policy is not None:
        sweep_step.early_termination = early_stopping_policy
    sweep_step.display_name = spec.display_name
    sweep_step.experiment_name = spec.experiment_name
    return sweep_step


def define_hpo_pipeline(
    components: dict[str, Any],
    *,
    specs: dict[str, SweepSpec],
    compute_name: str,
    primary_metric: str,
    current_data_asset: str,
    current_data_version: str,
    reference_data_asset: str,
    reference_data_version: str,
    early_stopping_policy: Any | None = None,
    gate_validation_before_prep: bool = False,
):
    """Define the Azure ML HPO pipeline graph."""
    from azure.ai.ml import dsl

    @dsl.pipeline(
        compute=compute_name,
        description="Validation-aware HPO pipeline for bank churn prediction",
    )
    def churn_hpo_pipeline(
        current_raw_data,
        reference_raw_data,
        data_config,
        train_config,
        hpo_config,
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

        hpo_logreg_sweep = None
        if "logreg" in specs:
            hpo_logreg_sweep = _build_hpo_sweep_step(
                components["hpo_trial"],
                spec=specs["logreg"],
                processed_data=data_prep_job.outputs.processed_data,
                train_config=train_config,
                hpo_config=hpo_config,
                compute_name=compute_name,
                early_stopping_policy=early_stopping_policy,
            )

        hpo_rf_sweep = None
        if "rf" in specs:
            hpo_rf_sweep = _build_hpo_sweep_step(
                components["hpo_trial"],
                spec=specs["rf"],
                processed_data=data_prep_job.outputs.processed_data,
                train_config=train_config,
                hpo_config=hpo_config,
                compute_name=compute_name,
                early_stopping_policy=early_stopping_policy,
            )

        hpo_xgboost_sweep = None
        if "xgboost" in specs:
            hpo_xgboost_sweep = _build_hpo_sweep_step(
                components["hpo_trial"],
                spec=specs["xgboost"],
                processed_data=data_prep_job.outputs.processed_data,
                train_config=train_config,
                hpo_config=hpo_config,
                compute_name=compute_name,
                early_stopping_policy=early_stopping_policy,
            )

        summary_inputs: dict[str, Any] = {
            "primary_metric": primary_metric,
            "data_config": data_config,
            "hpo_config": hpo_config,
            "current_data_asset": current_data_asset,
            "current_data_version": current_data_version,
            "reference_data_asset": reference_data_asset,
            "reference_data_version": reference_data_version,
        }
        if hpo_logreg_sweep is not None:
            summary_inputs["logreg_metrics"] = hpo_logreg_sweep.outputs.candidate_metrics
            summary_inputs["logreg_model_output"] = hpo_logreg_sweep.outputs.model_output
            summary_inputs["logreg_mlflow_model"] = hpo_logreg_sweep.outputs.mlflow_model
            summary_inputs["logreg_hpo_manifest"] = hpo_logreg_sweep.outputs.hpo_manifest
            summary_inputs["logreg_train_manifest"] = hpo_logreg_sweep.outputs.train_manifest
        if hpo_rf_sweep is not None:
            summary_inputs["rf_metrics"] = hpo_rf_sweep.outputs.candidate_metrics
            summary_inputs["rf_model_output"] = hpo_rf_sweep.outputs.model_output
            summary_inputs["rf_mlflow_model"] = hpo_rf_sweep.outputs.mlflow_model
            summary_inputs["rf_hpo_manifest"] = hpo_rf_sweep.outputs.hpo_manifest
            summary_inputs["rf_train_manifest"] = hpo_rf_sweep.outputs.train_manifest
        if hpo_xgboost_sweep is not None:
            summary_inputs["xgboost_metrics"] = hpo_xgboost_sweep.outputs.candidate_metrics
            summary_inputs["xgboost_model_output"] = hpo_xgboost_sweep.outputs.model_output
            summary_inputs["xgboost_mlflow_model"] = hpo_xgboost_sweep.outputs.mlflow_model
            summary_inputs["xgboost_hpo_manifest"] = hpo_xgboost_sweep.outputs.hpo_manifest
            summary_inputs["xgboost_train_manifest"] = hpo_xgboost_sweep.outputs.train_manifest

        collect_hpo_results_job = components["collect_hpo_results"](**summary_inputs)
        winner_inputs: dict[str, Any] = {
            "hpo_summary": collect_hpo_results_job.outputs.hpo_summary,
            "base_train_config": train_config,
        }
        if hpo_logreg_sweep is not None:
            winner_inputs["logreg_metrics"] = hpo_logreg_sweep.outputs.candidate_metrics
            winner_inputs["logreg_model_output"] = hpo_logreg_sweep.outputs.model_output
            winner_inputs["logreg_mlflow_model"] = hpo_logreg_sweep.outputs.mlflow_model
            winner_inputs["logreg_train_manifest"] = hpo_logreg_sweep.outputs.train_manifest
            winner_inputs["logreg_hpo_manifest"] = hpo_logreg_sweep.outputs.hpo_manifest
        if hpo_rf_sweep is not None:
            winner_inputs["rf_metrics"] = hpo_rf_sweep.outputs.candidate_metrics
            winner_inputs["rf_model_output"] = hpo_rf_sweep.outputs.model_output
            winner_inputs["rf_mlflow_model"] = hpo_rf_sweep.outputs.mlflow_model
            winner_inputs["rf_train_manifest"] = hpo_rf_sweep.outputs.train_manifest
            winner_inputs["rf_hpo_manifest"] = hpo_rf_sweep.outputs.hpo_manifest
        if hpo_xgboost_sweep is not None:
            winner_inputs["xgboost_metrics"] = hpo_xgboost_sweep.outputs.candidate_metrics
            winner_inputs["xgboost_model_output"] = hpo_xgboost_sweep.outputs.model_output
            winner_inputs["xgboost_mlflow_model"] = hpo_xgboost_sweep.outputs.mlflow_model
            winner_inputs["xgboost_train_manifest"] = hpo_xgboost_sweep.outputs.train_manifest
            winner_inputs["xgboost_hpo_manifest"] = hpo_xgboost_sweep.outputs.hpo_manifest

        materialize_hpo_winner_job = components["materialize_hpo_winner"](**winner_inputs)

        outputs: dict[str, Any] = {
            "validation_report": validate_job.outputs.validation_report,
            "validation_summary": validate_job.outputs.validation_summary,
            "validation_manifest": validate_job.outputs.validation_manifest,
            "data_prep_manifest": data_prep_job.outputs.data_prep_manifest,
            "hpo_summary": collect_hpo_results_job.outputs.hpo_summary,
            "hpo_summary_report": collect_hpo_results_job.outputs.hpo_summary_report,
            "hpo_manifest": collect_hpo_results_job.outputs.hpo_manifest,
            "winner_candidate_metrics": materialize_hpo_winner_job.outputs.winner_candidate_metrics,
            "winner_model_output": materialize_hpo_winner_job.outputs.winner_model_output,
            "winner_mlflow_model": materialize_hpo_winner_job.outputs.winner_mlflow_model,
            "winner_train_manifest": materialize_hpo_winner_job.outputs.winner_train_manifest,
            "winner_hpo_manifest": materialize_hpo_winner_job.outputs.winner_hpo_manifest,
            "winner_train_config": materialize_hpo_winner_job.outputs.winner_train_config,
            "winner_manifest": materialize_hpo_winner_job.outputs.winner_manifest,
        }
        if hpo_logreg_sweep is not None:
            outputs["logreg_model_output"] = hpo_logreg_sweep.outputs.model_output
            outputs["logreg_mlflow_model"] = hpo_logreg_sweep.outputs.mlflow_model
            outputs["logreg_candidate_metrics"] = hpo_logreg_sweep.outputs.candidate_metrics
            outputs["logreg_train_manifest"] = hpo_logreg_sweep.outputs.train_manifest
            outputs["logreg_hpo_manifest"] = hpo_logreg_sweep.outputs.hpo_manifest
        if hpo_rf_sweep is not None:
            outputs["rf_model_output"] = hpo_rf_sweep.outputs.model_output
            outputs["rf_mlflow_model"] = hpo_rf_sweep.outputs.mlflow_model
            outputs["rf_candidate_metrics"] = hpo_rf_sweep.outputs.candidate_metrics
            outputs["rf_train_manifest"] = hpo_rf_sweep.outputs.train_manifest
            outputs["rf_hpo_manifest"] = hpo_rf_sweep.outputs.hpo_manifest
        if hpo_xgboost_sweep is not None:
            outputs["xgboost_model_output"] = hpo_xgboost_sweep.outputs.model_output
            outputs["xgboost_mlflow_model"] = hpo_xgboost_sweep.outputs.mlflow_model
            outputs["xgboost_candidate_metrics"] = hpo_xgboost_sweep.outputs.candidate_metrics
            outputs["xgboost_train_manifest"] = hpo_xgboost_sweep.outputs.train_manifest
            outputs["xgboost_hpo_manifest"] = hpo_xgboost_sweep.outputs.hpo_manifest
        return outputs

    return churn_hpo_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the default end-to-end Azure ML HPO pipeline from raw-data and HPO config inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-config",
        default=None,
        help="Optional data config path. Defaults to configs/data.yaml.",
    )
    parser.add_argument(
        "--hpo-config",
        default=None,
        help="Optional HPO config path. Defaults to configs/hpo.yaml.",
    )
    parser.add_argument(
        "--current-data-override",
        default=None,
        help="Optional explicit current raw-data identifier or URI folder path.",
    )
    parser.add_argument(
        "--reference-data-override",
        default=None,
        help="Optional explicit reference raw-data identifier or URI folder path.",
    )
    args = parser.parse_args()

    install_azure_console_noise_filters()
    load_dotenv("config.env")
    data_config_path = resolve_data_config_path(args.data_config)
    hpo_config_path = resolve_hpo_config_path(args.hpo_config)
    train_config_path = resolve_train_config_path()
    runtime = get_pipeline_runtime_settings(
        "config.env",
        data_config_path=data_config_path,
    )
    overrides = HPODataInputOverrides(
        current_data_override=args.current_data_override,
        reference_data_override=args.reference_data_override,
    )
    metadata = get_hpo_pipeline_metadata(hpo_config_path)
    hpo_config = hpo_utils.load_hpo_config(hpo_config_path)
    specs = build_model_sweep_specs(hpo_config)
    components = load_hpo_pipeline_components(Path("aml/components"))
    pipeline = define_hpo_pipeline(
        components,
        specs=specs,
        compute_name=runtime.compute_name,
        primary_metric=str(metadata["primary_metric"] or "f1"),
        current_data_asset=runtime.current_data_asset_name,
        current_data_version=runtime.current_data_asset_version,
        reference_data_asset=runtime.reference_data_asset_name,
        reference_data_version=runtime.reference_data_asset_version,
        early_stopping_policy=build_early_stopping_policy(hpo_config),
        gate_validation_before_prep=get_pipeline_validation_gate(data_config_path),
    )

    current_raw_data_input, reference_raw_data_input, override_cleanup_dirs = _resolve_hpo_data_inputs(
        runtime,
        overrides=overrides,
    )
    try:
        pipeline_job = pipeline(
            current_raw_data=current_raw_data_input,
            reference_raw_data=reference_raw_data_input,
            data_config=build_local_file_input(data_config_path),
            train_config=build_local_file_input(train_config_path),
            hpo_config=build_local_file_input(hpo_config_path),
        )
        if metadata["experiment_name"]:
            pipeline_job.experiment_name = str(metadata["experiment_name"])
        if metadata["display_name"]:
            pipeline_job.display_name = str(metadata["display_name"])
        pipeline_job.tags = build_hpo_lineage_tags(
            runtime,
            data_config_path=data_config_path,
            hpo_config_path=hpo_config_path,
        )

        ml_client = get_ml_client("config.env")
        submission = submit_job_quietly(ml_client.jobs, pipeline_job)
    finally:
        for cleanup_dir in override_cleanup_dirs:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
    for message in build_submission_messages(
        model_name="hpo-pipeline",
        job_name=submission.name,
        studio_url=submission.studio_url,
    ):
        print(message)


if __name__ == "__main__":
    main()
