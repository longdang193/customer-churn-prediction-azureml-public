#!/usr/bin/env python3
"""Materialize canonical winner artifacts from per-family HPO outputs.

@meta
name: materialize_hpo_winner
type: module
domain: hpo
responsibility:
  - Provide hpo behavior for `src/materialize_hpo_winner.py`.
inputs: []
outputs: []
tags:
  - hpo
capabilities:
  - hpo.materialize-canonical-winner-outputs-winner-model-output-winner
  - hpo.materialize-winner-train-config-train-config-yaml-hpo
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Any

from hpo_winner_config import load_json, load_yaml_config, write_fixed_train_config
from utils.output_paths import resolve_named_input_file, resolve_named_output_file
from utils.step_manifest import (
    build_step_manifest,
    finalize_manifest,
    merge_section,
    resolve_manifest_output_path,
    set_failure,
)


CANDIDATE_METRICS_FILENAME = "candidate_metrics.json"
HPO_SUMMARY_FILENAME = "hpo_summary.json"
STEP_MANIFEST_FILENAME = "step_manifest.json"


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _copy_dir_contents(src_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for child in src_dir.iterdir():
        dest_child = dest_dir / child.name
        if child.is_dir():
            shutil.copytree(child, dest_child, dirs_exist_ok=True)
        else:
            shutil.copy2(child, dest_child)


def _copy_optional_file(path_str: str | None, output_path: str | None) -> str | None:
    if not path_str or not output_path:
        return None
    src = Path(path_str)
    dest = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(f"Winner file input does not exist: {src}")
    _copy_file(src, dest)
    return dest.as_posix()


def _copy_optional_named_file(
    path_str: str | None,
    output_path: str | None,
    *,
    filename: str,
) -> str | None:
    if not path_str or not output_path:
        return None
    src = resolve_named_input_file(Path(path_str), filename)
    dest = resolve_named_output_file(Path(output_path), filename)
    if not src.exists():
        raise FileNotFoundError(f"Winner file input does not exist: {src}")
    _copy_file(src, dest)
    return dest.as_posix()


def _copy_optional_dir(path_str: str | None, output_path: str | None) -> str | None:
    if not path_str or not output_path:
        return None
    src = Path(path_str)
    dest = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(f"Winner directory input does not exist: {src}")
    _copy_dir_contents(src, dest)
    return dest.as_posix()


def _winner_paths_for_family(args: argparse.Namespace, family: str) -> dict[str, str | None]:
    return {
        "candidate_metrics": getattr(args, f"{family}_metrics", None),
        "model_output": getattr(args, f"{family}_model_output", None),
        "mlflow_model": getattr(args, f"{family}_mlflow_model", None),
        "train_manifest": getattr(args, f"{family}_train_manifest", None),
        "hpo_manifest": getattr(args, f"{family}_hpo_manifest", None),
    }


def materialize_winner_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = resolve_named_input_file(Path(args.hpo_summary), HPO_SUMMARY_FILENAME)
    summary = load_json(summary_path)
    winner = summary["winner"]
    family = str(winner["model_name"])
    family_paths = _winner_paths_for_family(args, family)

    manifest = build_step_manifest(step_name="materialize_hpo_winner", stage_name="model_sweep")
    manifest_path = resolve_manifest_output_path(Path(args.winner_manifest))
    merge_section(
        manifest,
        "inputs",
        {
            "hpo_summary": args.hpo_summary,
            "winner_family": family,
            "family_paths": family_paths,
        },
    )
    merge_section(
        manifest,
        "outputs",
        {
            "winner_candidate_metrics": args.winner_candidate_metrics,
            "winner_model_output": args.winner_model_output,
            "winner_mlflow_model": args.winner_mlflow_model,
            "winner_train_manifest": args.winner_train_manifest,
            "winner_hpo_manifest": args.winner_hpo_manifest,
            "winner_train_config": getattr(args, "winner_train_config", None),
            "winner_manifest": args.winner_manifest,
        },
    )

    try:
        winner_candidate_metrics = _copy_optional_named_file(
            family_paths["candidate_metrics"],
            args.winner_candidate_metrics,
            filename=CANDIDATE_METRICS_FILENAME,
        )
        winner_model_output = _copy_optional_dir(
            family_paths["model_output"], args.winner_model_output
        )
        winner_mlflow_model = _copy_optional_dir(
            family_paths["mlflow_model"], args.winner_mlflow_model
        )
        winner_train_manifest = _copy_optional_dir(
            family_paths["train_manifest"], args.winner_train_manifest
        )
        winner_hpo_manifest = _copy_optional_dir(
            family_paths["hpo_manifest"], args.winner_hpo_manifest
        )
        winner_train_config = None
        if getattr(args, "base_train_config", None) and getattr(
            args, "winner_train_config", None
        ):
            hpo_manifest_path = resolve_named_input_file(
                Path(str(family_paths["hpo_manifest"])), STEP_MANIFEST_FILENAME
            )
            train_manifest_path = resolve_named_input_file(
                Path(str(family_paths["train_manifest"])), STEP_MANIFEST_FILENAME
            )
            winner_train_config = write_fixed_train_config(
                base_config=load_yaml_config(Path(args.base_train_config)),
                winner_family=family,
                hpo_manifest=load_json(hpo_manifest_path),
                train_manifest=load_json(train_manifest_path),
                output_dir=Path(args.winner_train_config),
            ).as_posix()

        materialized = {
            "winner_model_name": family,
            "winner_run_id": winner["run_id"],
            "winner_score": winner["score"],
            "winner_candidate_metrics": winner_candidate_metrics,
            "winner_model_output": winner_model_output,
            "winner_mlflow_model": winner_mlflow_model,
            "winner_train_manifest": winner_train_manifest,
            "winner_hpo_manifest": winner_hpo_manifest,
            "winner_train_config": winner_train_config,
        }
        merge_section(manifest, "tags", {"winner_model": family, "winner_run_id": winner["run_id"]})
        merge_section(manifest, "metrics", {"winner_score": winner["score"]})
        merge_section(
            manifest,
            "step_specific",
            {
                "materialized_outputs": materialized,
                "selection_policy": summary.get("selection_policy", {}),
                "tie_break_reason": winner.get("tie_break_reason"),
                "tie_candidates": winner.get("tie_candidates", []),
            },
        )
        finalize_manifest(manifest, output_path=manifest_path, status="success")
        return materialized
    except Exception as exc:
        set_failure(manifest, phase="materialize_hpo_winner", exc=exc)
        finalize_manifest(manifest, output_path=manifest_path, status="failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize winner outputs from HPO branch artifacts.")
    parser.add_argument("--hpo-summary", required=True)
    parser.add_argument("--logreg-metrics", default=None)
    parser.add_argument("--rf-metrics", default=None)
    parser.add_argument("--xgboost-metrics", default=None)
    parser.add_argument("--logreg-model-output", default=None)
    parser.add_argument("--rf-model-output", default=None)
    parser.add_argument("--xgboost-model-output", default=None)
    parser.add_argument("--logreg-mlflow-model", default=None)
    parser.add_argument("--rf-mlflow-model", default=None)
    parser.add_argument("--xgboost-mlflow-model", default=None)
    parser.add_argument("--logreg-train-manifest", default=None)
    parser.add_argument("--rf-train-manifest", default=None)
    parser.add_argument("--xgboost-train-manifest", default=None)
    parser.add_argument("--logreg-hpo-manifest", default=None)
    parser.add_argument("--rf-hpo-manifest", default=None)
    parser.add_argument("--xgboost-hpo-manifest", default=None)
    parser.add_argument("--winner-candidate-metrics", required=True)
    parser.add_argument("--winner-model-output", required=True)
    parser.add_argument("--winner-mlflow-model", required=True)
    parser.add_argument("--winner-train-manifest", required=True)
    parser.add_argument("--winner-hpo-manifest", required=True)
    parser.add_argument("--base-train-config", default=None)
    parser.add_argument("--winner-train-config", default=None)
    parser.add_argument("--winner-manifest", required=True)
    args = parser.parse_args()

    materialize_winner_artifacts(args)


if __name__ == "__main__":
    main()
