"""
@meta
name: inspect_hpo_run
type: script
domain: hpo
responsibility:
  - Download or inspect a completed HPO parent run.
  - Summarize per-family manifests, metrics, and common operator warnings.
  - Reduce manual Azure ML child-job log stitching during HPO debugging.
inputs:
  - Downloaded HPO parent run directory
  - Optional Azure ML job name and workspace scope
outputs:
  - Console summary of the HPO run
  - Optional JSON report file
tags:
  - azure-ml
  - hpo
  - debugging
  - observability
features:
  - notebook-hpo
capabilities:
  - hpo.inspect-downloaded-remote-hpo-parent-run-through-inspect
  - hpo.treat-inspect-hpo-run-py-export-hpo-winner
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


FAMILY_NAMES = ("logreg", "rf", "xgboost")
STEP_MANIFEST_FILENAME = "step_manifest.json"


def _resolve_az_executable() -> str:
    candidates = (
        shutil.which("az"),
        shutil.which("az.cmd"),
        r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd",
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError(
        "Could not locate the Azure CLI executable. Install Azure CLI or add az.cmd to PATH."
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_named_output_file(run_dir: Path, output_name: str) -> Path | None:
    output_dir = run_dir / "named-outputs" / output_name
    if not output_dir.exists():
        return None
    if output_dir.is_file():
        return output_dir
    children = [child for child in output_dir.iterdir() if child.is_file()]
    if len(children) == 1:
        return children[0]
    direct_manifest = output_dir / STEP_MANIFEST_FILENAME
    if direct_manifest.exists():
        return direct_manifest
    preferred = output_dir / output_name
    if preferred.exists():
        return preferred
    return None


def _resolve_named_output_path(run_dir: Path, output_name: str) -> Path | None:
    output_dir = run_dir / "named-outputs" / output_name
    if output_dir.exists():
        return output_dir
    return None


def _resolve_manifest_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.suffix.lower() == ".json":
        return path
    return path / STEP_MANIFEST_FILENAME


def _load_optional_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return _load_json(path)


def _prefer_existing_path(primary: Path | None, fallback: Path | None) -> Path | None:
    if primary is not None and primary.exists():
        return primary
    if fallback is not None and fallback.exists():
        return fallback
    return primary or fallback


def _resolve_existing_named_output(
    run_dir: Path,
    output_name: str,
    summary_path: str | None = None,
    *,
    prefer_directory: bool = False,
) -> Path | None:
    primary = Path(summary_path) if summary_path else None
    fallback = (
        _resolve_named_output_path(run_dir, output_name)
        if prefer_directory
        else _resolve_named_output_file(run_dir, output_name)
    )
    return _prefer_existing_path(primary, fallback)


def _validation_report(validation_summary: dict[str, Any] | None) -> dict[str, Any]:
    if validation_summary is None:
        return {
            "status": "missing",
            "drifted_column_share": None,
        }
    return {
        "status": validation_summary.get("status", "unknown"),
        "drifted_column_share": validation_summary.get("drift", {}).get("drifted_column_share"),
    }


def inspect_downloaded_hpo_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary_path = _resolve_named_output_file(run_dir, "hpo_summary")
    if summary_path is None:
        raise FileNotFoundError(f"Could not find hpo_summary under {run_dir}")
    summary = _load_json(summary_path)
    validation_summary_path = _resolve_named_output_file(run_dir, "validation_summary")
    validation_summary = (
        _load_json(validation_summary_path) if validation_summary_path is not None else None
    )

    candidate_results = {
        str(candidate["model_name"]): candidate
        for candidate in summary.get("candidate_results", [])
    }
    family_artifacts = summary.get("family_artifacts", {}) or {}
    family_bundle_artifacts = summary.get("family_bundle_artifacts", {}) or {}
    families = sorted(set(candidate_results) | set(family_artifacts))

    family_reports: dict[str, Any] = {}
    for family in families:
        candidate = candidate_results.get(family, {})
        artifact_paths = family_artifacts.get(family, {}) or {}
        bundle_paths = family_bundle_artifacts.get(family, {}) or {}
        model_output_path = _resolve_existing_named_output(
            run_dir,
            f"{family}_model_output",
            bundle_paths.get("model_output"),
            prefer_directory=True,
        )
        mlflow_model_path = _resolve_existing_named_output(
            run_dir,
            f"{family}_mlflow_model",
            bundle_paths.get("mlflow_model"),
            prefer_directory=True,
        )
        fallback_hpo_manifest = _resolve_named_output_file(run_dir, f"{family}_hpo_manifest")
        fallback_train_manifest = _resolve_named_output_file(run_dir, f"{family}_train_manifest")
        hpo_manifest_path = _resolve_manifest_path(artifact_paths.get("hpo_manifest"))
        train_manifest_path = _resolve_manifest_path(artifact_paths.get("train_manifest"))
        effective_hpo_manifest_path = _prefer_existing_path(
            hpo_manifest_path, fallback_hpo_manifest
        )
        effective_train_manifest_path = _prefer_existing_path(
            train_manifest_path, fallback_train_manifest
        )
        hpo_manifest = _load_optional_manifest(effective_hpo_manifest_path)
        train_manifest = _load_optional_manifest(effective_train_manifest_path)
        warnings: list[str] = []
        for manifest in (hpo_manifest, train_manifest):
            if manifest:
                warnings.extend(str(item) for item in manifest.get("warnings", []))
        family_reports[family] = {
            "best_trial_run_id": candidate.get("run_id"),
            "metrics": candidate.get("metrics", {}),
            "model_output": str(model_output_path) if model_output_path else None,
            "mlflow_model": str(mlflow_model_path) if mlflow_model_path else None,
            "hpo_manifest": {
                "path": str(effective_hpo_manifest_path) if effective_hpo_manifest_path else None,
                "status": hpo_manifest.get("status") if hpo_manifest else "missing",
            },
            "train_manifest": {
                "path": str(effective_train_manifest_path)
                if effective_train_manifest_path
                else None,
                "status": train_manifest.get("status") if train_manifest else "missing",
            },
            "warnings": warnings,
        }

    return {
        "run_dir": str(run_dir),
        "primary_metric": summary.get("primary_metric"),
        "winner": summary.get("winner", {}),
        "validation": _validation_report(validation_summary),
        "winner_outputs": {
            "candidate_metrics": (
                str(_resolve_named_output_file(run_dir, "winner_candidate_metrics"))
                if _resolve_named_output_file(run_dir, "winner_candidate_metrics")
                else None
            ),
            "model_output": (
                str(_resolve_named_output_path(run_dir, "winner_model_output"))
                if _resolve_named_output_path(run_dir, "winner_model_output")
                else None
            ),
            "mlflow_model": (
                str(_resolve_named_output_path(run_dir, "winner_mlflow_model"))
                if _resolve_named_output_path(run_dir, "winner_mlflow_model")
                else None
            ),
            "train_manifest": (
                str(_resolve_named_output_path(run_dir, "winner_train_manifest"))
                if _resolve_named_output_path(run_dir, "winner_train_manifest")
                else None
            ),
            "hpo_manifest": (
                str(_resolve_named_output_path(run_dir, "winner_hpo_manifest"))
                if _resolve_named_output_path(run_dir, "winner_hpo_manifest")
                else None
            ),
            "train_config": (
                str(_resolve_named_output_path(run_dir, "winner_train_config"))
                if _resolve_named_output_path(run_dir, "winner_train_config")
                else None
            ),
            "winner_manifest": (
                str(_resolve_named_output_path(run_dir, "winner_manifest"))
                if _resolve_named_output_path(run_dir, "winner_manifest")
                else None
            ),
        },
        "families": family_reports,
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "HPO Run Inspection",
        f"- Run dir: {report['run_dir']}",
        f"- Primary metric: {report.get('primary_metric')}",
        f"- Winner: {report['winner'].get('model_name')} ({report['winner'].get('run_id')})",
        f"- Winner score: {report['winner'].get('score')}",
        f"- Validation status: {report['validation'].get('status')}",
        f"- Validation drifted column share: {report['validation'].get('drifted_column_share')}",
        "",
        "Winner outputs:",
        f"- candidate metrics: {report['winner_outputs'].get('candidate_metrics')}",
        f"- model output: {report['winner_outputs'].get('model_output')}",
        f"- mlflow model: {report['winner_outputs'].get('mlflow_model')}",
        f"- train manifest: {report['winner_outputs'].get('train_manifest')}",
        f"- hpo manifest: {report['winner_outputs'].get('hpo_manifest')}",
        f"- train config: {report['winner_outputs'].get('train_config')}",
        "",
        "Families:",
    ]
    for family, details in sorted(report["families"].items()):
        warnings = "; ".join(details["warnings"]) if details["warnings"] else "none"
        lines.extend(
            [
                f"- {family}:",
                f"  best trial: {details.get('best_trial_run_id')}",
                f"  model output: {details.get('model_output')}",
                f"  mlflow model: {details.get('mlflow_model')}",
                f"  hpo manifest: {details['hpo_manifest']['status']} ({details['hpo_manifest']['path']})",
                f"  train manifest: {details['train_manifest']['status']} ({details['train_manifest']['path']})",
                f"  warnings: {warnings}",
            ]
        )
    return "\n".join(lines)


def download_hpo_run(
    *,
    job_name: str,
    resource_group: str,
    workspace_name: str,
    download_path: Path,
) -> Path:
    subprocess.run(
        [
            _resolve_az_executable(),
            "ml",
            "job",
            "download",
            "--name",
            job_name,
            "--resource-group",
            resource_group,
            "--workspace-name",
            workspace_name,
            "--download-path",
            str(download_path),
            "--all",
        ],
        check=True,
    )
    return download_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a downloaded or remote Azure ML HPO parent run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default=None, help="Existing downloaded HPO parent-run directory.")
    parser.add_argument("--job-name", default=None, help="Azure ML parent HPO job name to download.")
    parser.add_argument("--resource-group", default=None, help="Azure resource group for job download.")
    parser.add_argument("--workspace-name", default=None, help="Azure ML workspace name for job download.")
    parser.add_argument(
        "--download-path",
        default=None,
        help="Directory where the job should be downloaded before inspection.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write the inspection report as JSON.",
    )
    args = parser.parse_args()

    run_dir: Path | None = Path(args.run_dir) if args.run_dir else None
    if run_dir is None:
        if not args.job_name:
            raise SystemExit("Provide either --run-dir or --job-name.")
        if not args.resource_group or not args.workspace_name:
            raise SystemExit("--resource-group and --workspace-name are required with --job-name.")
        download_path = Path(args.download_path or f"downloaded_hpo_{args.job_name}")
        run_dir = download_hpo_run(
            job_name=args.job_name,
            resource_group=args.resource_group,
            workspace_name=args.workspace_name,
            download_path=download_path,
        )

    report = inspect_downloaded_hpo_run(run_dir)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(render_report(report))


if __name__ == "__main__":
    main()
