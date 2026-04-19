"""
@meta
name: export_hpo_winner_config
type: script
domain: hpo
responsibility:
  - Export the winning HPO family into a fixed train-config YAML.
  - Reuse downloaded or remotely fetched HPO parent outputs as the source of truth.
inputs:
  - Downloaded HPO parent run directory or Azure ML HPO parent job name
  - Base training config YAML
outputs:
  - Fixed training config YAML for standard train pipeline execution
tags:
  - hpo
  - training
  - handoff
  - config
features:
  - notebook-hpo
  - model-training-pipeline
capabilities:
  - hpo.export-selected-hpo-winner-fixed-train-config-yaml
  - hpo.materialize-winner-train-config-train-config-yaml-hpo
  - hpo.treat-inspect-hpo-run-py-export-hpo-winner
  - hpo.analyze-completed-sweep-results-update-configs-train-yaml
  - fixed-train.accept-hpo-winner-train-config
  - fixed-train.preserve-train-config-identity
  - fixed-train.carry-manifest-lineage
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from tools.hpo.inspect_hpo_run import download_hpo_run, inspect_downloaded_hpo_run
from src.hpo_winner_config import (
    build_fixed_train_config,
    load_json,
    load_yaml_config,
)
from src.utils.config_loader import load_config


def _winner_manifest_paths(report: dict[str, Any]) -> tuple[Path, Path]:
    winner_family = str(report["winner"]["model_name"])
    family_report = report["families"].get(winner_family, {})
    hpo_manifest_path = family_report.get("hpo_manifest", {}).get("path")
    train_manifest_path = family_report.get("train_manifest", {}).get("path")
    if not hpo_manifest_path or not train_manifest_path:
        raise FileNotFoundError(
            f"Could not find HPO/train manifests for winner family '{winner_family}'."
        )
    return Path(hpo_manifest_path), Path(train_manifest_path)


def export_winner_config(
    *,
    run_dir: Path,
    output_config: Path,
    base_config_path: Path,
    experiment_name: str | None = None,
    display_name: str | None = None,
) -> dict[str, Any]:
    report = inspect_downloaded_hpo_run(run_dir)
    winner_family = str(report["winner"]["model_name"])
    hpo_manifest_path, train_manifest_path = _winner_manifest_paths(report)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    exported = build_fixed_train_config(
        base_config=load_config(str(base_config_path))
        or load_yaml_config(base_config_path),
        winner_family=winner_family,
        hpo_manifest=load_json(hpo_manifest_path),
        train_manifest=load_json(train_manifest_path),
        experiment_name=experiment_name,
        display_name=display_name,
        canonical_train_config=output_config.as_posix(),
    )
    output_config.write_text(
        yaml.safe_dump(exported, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the winning HPO family into a fixed train config."
    )
    parser.add_argument("--run-dir", default=None, help="Downloaded HPO parent run directory.")
    parser.add_argument("--job-name", default=None, help="Azure ML HPO parent job name.")
    parser.add_argument("--resource-group", default=None, help="Azure resource group.")
    parser.add_argument("--workspace-name", default=None, help="Azure ML workspace name.")
    parser.add_argument(
        "--download-path",
        default=None,
        help="Optional download directory used with --job-name.",
    )
    parser.add_argument(
        "--base-config",
        default="configs/train.yaml",
        help="Base train config whose promotion/default sections should be preserved.",
    )
    parser.add_argument("--output-config", required=True, help="Path to write the fixed config.")
    parser.add_argument("--experiment-name", default=None, help="Optional experiment name override.")
    parser.add_argument("--display-name", default=None, help="Optional display name override.")
    args = parser.parse_args()

    run_dir: Path | None = Path(args.run_dir) if args.run_dir else None
    if run_dir is None:
        if not args.job_name:
            raise SystemExit("Provide either --run-dir or --job-name.")
        if not args.resource_group or not args.workspace_name:
            raise SystemExit("--resource-group and --workspace-name are required with --job-name.")
        run_dir = download_hpo_run(
            job_name=args.job_name,
            resource_group=args.resource_group,
            workspace_name=args.workspace_name,
            download_path=Path(args.download_path or f"downloaded_hpo_{args.job_name}"),
        )

    output_path = Path(args.output_config)
    exported = export_winner_config(
        run_dir=run_dir,
        output_config=output_path,
        base_config_path=Path(args.base_config),
        experiment_name=args.experiment_name,
        display_name=args.display_name,
    )
    winner_family = exported["training"]["models"][0]
    print(f"OK Exported HPO winner config: {output_path}")
    print(f"  Winner family: {winner_family}")


if __name__ == "__main__":
    main()
