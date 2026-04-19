"""
@meta
name: run_hpo
type: script
domain: hpo
responsibility:
  - Build script-first Azure ML sweep specifications from configs/hpo.yaml.
  - Submit sweep jobs outside the notebook-driven workflow.
  - Provide reusable sweep builders for the HPO pipeline entrypoint.
inputs:
  - config.env
  - configs/hpo.yaml
  - Processed dataset URI
outputs:
  - Azure ML sweep job submissions
tags:
  - azure-ml
  - hpo
  - orchestration
features:
  - notebook-hpo
capabilities:
  - hpo.build-sweep-definitions-configs-hpo-yaml-another-selected
  - hpo.submit-reload-sweep-jobs-azure-ml-run-hpo
  - hpo.treat-configs-hpo-smoke-yaml-wiring-artifact-profile
  - hpo.provide-review-surface-periodic-re-optimization-becoming-sole
invariants:
  - hpo.run-hpo-py-remains-canonical-direct-rerun
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import hpo_utils
from src.azureml import (
    build_local_file_input,
    build_uri_folder_input,
    get_ml_client,
    install_azure_console_noise_filters,
    submit_job_quietly,
)
from src.config.runtime import get_pipeline_compute_name


SHARED_SWEEP_PARAM_NAMES = ("use_smote", "class_weight", "random_state")


@dataclass(frozen=True)
class SweepSpec:
    model_name: str
    command: str
    search_space: dict[str, list[Any]]
    metric: str
    mode: str
    sampling_algorithm: str
    experiment_name: str
    display_name: str
    max_total_trials: int | None
    max_concurrent_trials: int | None
    timeout_seconds: int | None
    trial_timeout_seconds: int | None


def _resolve_model_types(search_space: dict[str, Any]) -> list[str]:
    configured_types = search_space.get("model_types")
    if configured_types:
        return list(configured_types)

    inferred: list[str] = []
    for candidate in ("logreg", "rf", "xgboost"):
        if candidate in search_space:
            inferred.append(candidate)
    return inferred


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _minutes_to_seconds(value: Any) -> int | None:
    if value is None:
        return None
    return int(value) * 60


def normalize_sweep_goal(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "max":
        return "Maximize"
    if normalized == "min":
        return "Minimize"
    return value


def _build_direct_sweep_command(
    *,
    model_name: str,
    search_space_cfg: dict[str, Any],
) -> tuple[str, dict[str, list[Any]]]:
    command_segments = [
        "python run_sweep_trial.py",
        "--data ${{inputs.data}}",
        "--hpo-config ${{inputs.hpo_config}}",
        f"--model-type {model_name}",
        "--model-output ${{outputs.model_output}}",
        "--candidate-metrics-output ${{outputs.candidate_metrics}}",
        "--mlflow-model-output ${{outputs.mlflow_model}}",
        "--manifest-output ${{outputs.train_manifest}}",
        "--hpo-manifest-output ${{outputs.hpo_manifest}}",
    ]
    sweep_search_space: dict[str, list[Any]] = {}

    for param_name in SHARED_SWEEP_PARAM_NAMES:
        values = search_space_cfg.get(param_name)
        if isinstance(values, list) and values:
            command_segments.append(f"--{param_name} ${{{{search_space.{param_name}}}}}")
            sweep_search_space[param_name] = list(values)

    model_space = search_space_cfg.get(model_name) or {}
    for param_name, values in model_space.items():
        prefixed_name = f"{model_name}_{param_name}"
        command_segments.append(f"--{prefixed_name} ${{{{search_space.{prefixed_name}}}}}")
        sweep_search_space[prefixed_name] = list(values)

    return " ".join(command_segments), sweep_search_space


def build_model_sweep_specs(hpo_config: dict[str, Any]) -> dict[str, SweepSpec]:
    """Build one sweep spec per model family from configs/hpo.yaml semantics."""
    search_space_cfg = hpo_utils.build_parameter_space(hpo_config.get("search_space", {}))
    budget_cfg = hpo_config.get("budget", {}) or {}
    timeout_cfg = hpo_config.get("timeouts", {}) or {}
    specs: dict[str, SweepSpec] = {}

    for model_name in _resolve_model_types(search_space_cfg):
        model_space = search_space_cfg.get(model_name) or {}
        if not model_space:
            continue

        command, sweep_search_space = _build_direct_sweep_command(
            model_name=model_name,
            search_space_cfg=search_space_cfg,
        )
        specs[model_name] = SweepSpec(
            model_name=model_name,
            command=command,
            search_space=sweep_search_space,
            metric=str(hpo_config.get("metric", "f1")),
            mode=str(hpo_config.get("mode", "max")),
            sampling_algorithm=str(hpo_config.get("sampling_algorithm", "random")),
            experiment_name=str(hpo_config.get("experiment_name", "manual-hpo-sweep")),
            display_name=f"{hpo_config.get('sweep_display_name', 'manual-hpo-sweep')}-{model_name}",
            max_total_trials=_coerce_int(budget_cfg.get("max_trials")),
            max_concurrent_trials=_coerce_int(budget_cfg.get("max_concurrent")),
            timeout_seconds=_minutes_to_seconds(timeout_cfg.get("total_minutes")),
            trial_timeout_seconds=_minutes_to_seconds(timeout_cfg.get("trial_minutes")),
        )

    return specs


def build_sweep_search_space(spec: SweepSpec) -> dict[str, Any]:
    """Build Azure ML Choice search-space objects for a sweep spec."""
    from azure.ai.ml.sweep import Choice

    return {name: Choice(values=values) for name, values in spec.search_space.items()}


def configure_sweep_limits(sweep_step: Any, spec: SweepSpec) -> None:
    """Apply configured sweep limits to a direct or pipeline sweep step."""
    limit_kwargs: dict[str, int] = {}
    if spec.max_total_trials is not None:
        limit_kwargs["max_total_trials"] = spec.max_total_trials
    if spec.max_concurrent_trials is not None:
        limit_kwargs["max_concurrent_trials"] = spec.max_concurrent_trials
    if spec.timeout_seconds is not None:
        limit_kwargs["timeout"] = spec.timeout_seconds
    if spec.trial_timeout_seconds is not None:
        limit_kwargs["trial_timeout"] = spec.trial_timeout_seconds
    if limit_kwargs:
        sweep_step.set_limits(**limit_kwargs)


def build_early_stopping_policy(hpo_config: dict[str, Any]) -> Any | None:
    """Build the configured early-termination policy when enabled."""
    early_stopping_cfg = hpo_config.get("early_stopping", {}) or {}
    if not early_stopping_cfg.get("enabled"):
        return None
    if early_stopping_cfg.get("policy") != "bandit":
        return None

    from azure.ai.ml.sweep import BanditPolicy

    return BanditPolicy(
        evaluation_interval=int(early_stopping_cfg.get("evaluation_interval", 3)),
        slack_factor=float(early_stopping_cfg.get("slack_factor", 0.15)),
    )


def build_submission_messages(*, model_name: str, job_name: str, studio_url: str) -> list[str]:
    """Return ASCII-safe submission messages for local consoles."""
    return [
        f"OK Submitted {model_name} sweep: {job_name}",
        f"  View in Azure ML Studio: {studio_url}",
    ]


def submit_sweeps(
    processed_data_uri: str,
    specs: dict[str, SweepSpec],
    config_path: str | None = None,
) -> None:
    """Submit Azure ML sweep jobs for the provided specs."""
    from azure.ai.ml import Output, command

    install_azure_console_noise_filters()
    ml_client = get_ml_client("config.env")
    compute_name = get_pipeline_compute_name("config.env")
    hpo_config = hpo_utils.load_hpo_config(config_path)
    hpo_config_path = Path(config_path or "configs/hpo.yaml")
    early_stopping_policy = build_early_stopping_policy(hpo_config)

    for model_name, spec in specs.items():
        base_command = command(
            code="./src",
            command=spec.command,
            environment="azureml:bank-churn-env:1",
            compute=compute_name,
            inputs={
                "data": build_uri_folder_input(processed_data_uri, mode="mount"),
                "hpo_config": build_local_file_input(hpo_config_path),
            },
            outputs={
                "model_output": Output(type="uri_folder"),
                "candidate_metrics": Output(type="uri_file"),
                "mlflow_model": Output(type="uri_folder"),
                "train_manifest": Output(type="uri_folder"),
                "hpo_manifest": Output(type="uri_folder"),
            },
            display_name=f"manual-hpo-sweep-trial-{model_name}",
            experiment_name=spec.experiment_name,
        )

        sweep_job = base_command.sweep(
            search_space=build_sweep_search_space(spec),
            sampling_algorithm=spec.sampling_algorithm,
            primary_metric=spec.metric,
            goal=normalize_sweep_goal(spec.mode),
        )
        configure_sweep_limits(sweep_job, spec)
        if early_stopping_policy is not None:
            sweep_job.early_termination = early_stopping_policy
        sweep_job.display_name = spec.display_name
        sweep_job.experiment_name = spec.experiment_name
        submission = submit_job_quietly(ml_client.jobs, sweep_job)
        for message in build_submission_messages(
            model_name=model_name,
            job_name=submission.name,
            studio_url=submission.studio_url,
        ):
            print(message)


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(
        description="Submit direct Azure ML HPO sweeps from processed data (advanced rerun path).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed-data-uri",
        required=True,
        help="Processed dataset URI/folder to use for sweep trials",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional HPO config path. Defaults to configs/hpo.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sweep specs without submitting jobs",
    )
    args = parser.parse_args()

    load_dotenv("config.env")
    specs = build_model_sweep_specs(hpo_utils.load_hpo_config(args.config))
    if args.dry_run:
        for model_name, spec in specs.items():
            print(f"[{model_name}] {spec}")
        return

    submit_sweeps(args.processed_data_uri, specs, args.config)


if __name__ == "__main__":
    main()
