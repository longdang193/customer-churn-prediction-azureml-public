"""
@meta
name: run_retraining_hpo_to_fixed_train
type: script
domain: retraining-continuation
responsibility:
  - Continue a completed retraining HPO smoke run into a production-facing fixed-train rerun.
  - Validate winner consistency across HPO summary, winner manifest, and winner train config before invoking fixed train.
inputs:
  - release_record.json
  - retraining_path_selection.json
  - retraining_candidate_manifest.json
  - validation_summary.json
  - retraining_hpo_smoke_summary.json
  - Downloaded HPO run directory or Azure ML HPO parent job identity
outputs:
  - retraining_exported_train_config.json
  - effective_train_config/train_config.yaml
  - retraining_hpo_to_fixed_train_summary.json
  - retraining_hpo_to_fixed_train_report.md
  - retraining_hpo_to_fixed_train_manifest/step_manifest.json
tags:
  - monitoring
  - retraining
  - hpo
  - training
  - cli
features:
  - model-training-pipeline
  - notebook-hpo
  - release-monitoring-evaluator
capabilities:
  - fixed-train.accept-hpo-to-fixed-continuation
  - fixed-train.accept-hpo-winner-train-config
  - hpo.export-selected-hpo-winner-fixed-train-config-yaml
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Optional, TypedDict, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from tools.hpo.export_hpo_winner_config import export_winner_config
from tools.hpo.inspect_hpo_run import (
    _resolve_az_executable,
    download_hpo_run,
    inspect_downloaded_hpo_run,
)
from src.hpo_winner_config import load_yaml_config
from src.utils.step_manifest import build_step_manifest, finalize_manifest, merge_config, merge_section


STATUS_BLOCKED_BY_VALIDATION = "blocked_by_validation"
STATUS_BLOCKED_BY_SELECTION = "blocked_by_selection"
STATUS_BLOCKED_BY_HPO_STATUS = "blocked_by_hpo_status"
STATUS_BLOCKED_BY_WINNER_INCONSISTENCY = "blocked_by_winner_inconsistency"
STATUS_DRY_RUN_READY = "dry_run_ready"
STATUS_SUBMITTED = "submitted"
STATUS_CONTINUATION_FAILED = "continuation_failed"

EXPECTED_HPO_STATUS = "submitted"
EXPECTED_SELECTION_PATH = "model_sweep"
TRAIN_CONFIG_FILENAME = "train_config.yaml"
TERMINAL_AZURE_JOB_STATUSES = {"completed", "failed", "canceled", "notresponding", "paused"}
HPO_WAIT_TIMEOUT_SECONDS = 5400
HPO_POLL_INTERVAL_SECONDS = 15


class SelectionPayload(TypedDict, total=False):
    selected_path: str
    reason_codes: list[str]
    downstream: dict[str, object]


class HpoSmokeSummary(TypedDict, total=False):
    status: str
    selected_path: str
    submission: dict[str, object]


@dataclass(frozen=True)
class UpstreamContext:
    release_record_path: Path
    selection_path: Path
    selection_payload: SelectionPayload
    candidate_manifest_path: Path
    candidate_manifest: dict[str, object]
    validation_summary_path: Path
    validation_summary: dict[str, object]
    hpo_smoke_summary_path: Path
    hpo_smoke_summary: HpoSmokeSummary


@dataclass(frozen=True)
class WinnerArtifacts:
    winner_family: str
    winner_run_id: str | None
    hpo_summary_path: Path
    winner_manifest_path: Path
    winner_train_config_path: Path | None
    run_dir: Path


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_context(args: argparse.Namespace) -> UpstreamContext:
    return UpstreamContext(
        release_record_path=Path(args.release_record).resolve(),
        selection_path=Path(args.selection).resolve(),
        selection_payload=cast(SelectionPayload, _load_json(Path(args.selection).resolve())),
        candidate_manifest_path=Path(args.candidate_manifest).resolve(),
        candidate_manifest=_load_json(Path(args.candidate_manifest).resolve()),
        validation_summary_path=Path(args.validation_summary).resolve(),
        validation_summary=_load_json(Path(args.validation_summary).resolve()),
        hpo_smoke_summary_path=Path(args.hpo_smoke_summary).resolve(),
        hpo_smoke_summary=cast(HpoSmokeSummary, _load_json(Path(args.hpo_smoke_summary).resolve())),
    )


def _validation_status(validation_summary: Mapping[str, object]) -> str:
    status = validation_summary.get("status")
    return status if isinstance(status, str) else "unknown"


def _selected_path(selection_payload: Mapping[str, object]) -> str:
    value = selection_payload.get("selected_path")
    return value if isinstance(value, str) else "unknown"


def _hpo_submission_job_name(hpo_smoke_summary: Mapping[str, object]) -> str | None:
    submission = hpo_smoke_summary.get("submission")
    if not isinstance(submission, Mapping):
        return None
    job_name = submission.get("job_name")
    return job_name if isinstance(job_name, str) else None


def _downstream_data_config(selection_payload: Mapping[str, object], candidate_manifest: Mapping[str, object]) -> str:
    downstream = selection_payload.get("downstream")
    if isinstance(downstream, Mapping):
        path = downstream.get("data_config_path")
        if isinstance(path, str):
            return path
    fallback = candidate_manifest.get("data_config_path")
    if isinstance(fallback, str):
        return fallback
    raise SystemExit("Could not resolve data_config_path for fixed-train continuation")


def _resolve_hpo_run_dir(
    *,
    hpo_run_dir: Path | None,
    hpo_job_name: str | None,
    resource_group: str | None,
    workspace_name: str | None,
    download_path: Path,
) -> Path:
    if hpo_run_dir is not None:
        return hpo_run_dir.resolve()
    if hpo_job_name is None or resource_group is None or workspace_name is None:
        raise SystemExit(
            "Provide either --hpo-run-dir or --hpo-job-name with --resource-group and --workspace-name"
        )
    return download_hpo_run(
        job_name=hpo_job_name,
        resource_group=resource_group,
        workspace_name=workspace_name,
        download_path=download_path,
    )


def _wait_for_hpo_job_completion(
    *,
    job_name: str,
    resource_group: str,
    workspace_name: str,
    timeout_seconds: int = HPO_WAIT_TIMEOUT_SECONDS,
    poll_interval_seconds: int = HPO_POLL_INTERVAL_SECONDS,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_status = "unknown"
    while True:
        completed = subprocess.run(
            [
                _resolve_az_executable(),
                "ml",
                "job",
                "show",
                "--name",
                job_name,
                "--resource-group",
                resource_group,
                "--workspace-name",
                workspace_name,
                "--output",
                "json",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(completed.stdout)
        status = payload.get("status")
        last_status = str(status or "unknown").strip().lower()
        if last_status in TERMINAL_AZURE_JOB_STATUSES:
            return last_status
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"Timed out waiting for HPO parent job {job_name!r}; last status={last_status!r}"
            )
        time.sleep(poll_interval_seconds)


def _resolve_optional_path(path_str: object, filename: str) -> Path | None:
    if not isinstance(path_str, str):
        return None
    path = Path(path_str)
    if path.is_file():
        return path
    candidate = path / filename
    return candidate if candidate.exists() else None


def _winner_family_from_manifest(manifest: Mapping[str, object]) -> str | None:
    step_specific = manifest.get("step_specific")
    if isinstance(step_specific, Mapping):
        materialized = step_specific.get("materialized_outputs")
        if isinstance(materialized, Mapping):
            winner = materialized.get("winner_model_name")
            if isinstance(winner, str):
                return winner
    inputs = manifest.get("inputs")
    if isinstance(inputs, Mapping):
        winner = inputs.get("winner_family")
        if isinstance(winner, str):
            return winner
    tags = manifest.get("tags")
    if isinstance(tags, Mapping):
        winner = tags.get("winner_model")
        if isinstance(winner, str):
            return winner
    return None


def _winner_family_from_train_config(path: Path) -> str | None:
    config = load_yaml_config(path)
    training = config.get("training")
    if not isinstance(training, Mapping):
        return None
    models = training.get("models")
    if not isinstance(models, list) or not models:
        return None
    first = models[0]
    return first if isinstance(first, str) else None


def _resolve_winner_artifacts(run_dir: Path) -> WinnerArtifacts:
    report = inspect_downloaded_hpo_run(run_dir)
    winner = report.get("winner", {})
    if not isinstance(winner, Mapping):
        raise SystemExit("HPO report did not contain a valid winner block")
    winner_family = winner.get("model_name")
    if not isinstance(winner_family, str):
        raise SystemExit("HPO report did not expose winner.model_name")
    winner_run_id = winner.get("run_id")
    hpo_summary_path = run_dir / "named-outputs" / "hpo_summary" / "hpo_summary.json"
    winner_outputs = report.get("winner_outputs", {})
    if not isinstance(winner_outputs, Mapping):
        raise SystemExit("HPO report did not expose winner_outputs")
    winner_manifest_path = _resolve_optional_path(winner_outputs.get("winner_manifest"), "step_manifest.json")
    if winner_manifest_path is None:
        raise SystemExit("Could not resolve winner_manifest/step_manifest.json from the HPO run")
    winner_train_config_path = _resolve_optional_path(
        winner_outputs.get("train_config"),
        TRAIN_CONFIG_FILENAME,
    )
    return WinnerArtifacts(
        winner_family=winner_family,
        winner_run_id=winner_run_id if isinstance(winner_run_id, str) else None,
        hpo_summary_path=hpo_summary_path,
        winner_manifest_path=winner_manifest_path,
        winner_train_config_path=winner_train_config_path,
        run_dir=run_dir,
    )


def _validate_winner_consistency(artifacts: WinnerArtifacts) -> tuple[str, list[str]]:
    checks: list[str] = []
    winner_manifest = _load_json(artifacts.winner_manifest_path)
    manifest_family = _winner_family_from_manifest(winner_manifest)
    if manifest_family != artifacts.winner_family:
        raise SystemExit(
            "Winner mismatch between hpo_summary and winner_manifest: "
            f"{artifacts.winner_family!r} != {manifest_family!r}"
        )
    checks.append("winner_manifest_matches_hpo_summary")
    if artifacts.winner_train_config_path is not None and artifacts.winner_train_config_path.exists():
        config_family = _winner_family_from_train_config(artifacts.winner_train_config_path)
        if config_family != artifacts.winner_family:
            raise SystemExit(
                "Winner mismatch between hpo_summary and winner_train_config: "
                f"{artifacts.winner_family!r} != {config_family!r}"
            )
        checks.append("winner_train_config_matches_hpo_summary")
    return artifacts.winner_family, checks


def _materialize_effective_train_config(
    *,
    artifacts: WinnerArtifacts,
    output_dir: Path,
    base_config_path: Path,
) -> tuple[Path, dict[str, object]]:
    effective_dir = output_dir / "effective_train_config"
    effective_dir.mkdir(parents=True, exist_ok=True)
    effective_path = effective_dir / TRAIN_CONFIG_FILENAME
    if artifacts.winner_train_config_path is not None and artifacts.winner_train_config_path.exists():
        shutil.copy2(artifacts.winner_train_config_path, effective_path)
        metadata = {
            "created_at_utc": _utc_timestamp(),
            "source_kind": "reused_winner_train_config",
            "source_path": str(artifacts.winner_train_config_path),
            "effective_train_config_path": str(effective_path),
            "winner_family": artifacts.winner_family,
            "winner_run_id": artifacts.winner_run_id,
            "consistency_checks": [
                "winner_manifest_matches_hpo_summary",
                "winner_train_config_matches_hpo_summary",
            ],
        }
        return effective_path, metadata

    export_winner_config(
        run_dir=artifacts.run_dir,
        output_config=effective_path,
        base_config_path=base_config_path,
    )
    config_family = _winner_family_from_train_config(effective_path)
    if config_family != artifacts.winner_family:
        raise SystemExit(
            "Exported train config does not match HPO winner family: "
            f"{artifacts.winner_family!r} != {config_family!r}"
        )
    metadata = {
        "created_at_utc": _utc_timestamp(),
        "source_kind": "exported_from_hpo",
        "source_path": str(artifacts.hpo_summary_path),
        "effective_train_config_path": str(effective_path),
        "winner_family": artifacts.winner_family,
        "winner_run_id": artifacts.winner_run_id,
        "consistency_checks": [
            "winner_manifest_matches_hpo_summary",
            "exported_train_config_matches_hpo_summary",
        ],
        "base_config_path": str(base_config_path),
    }
    return effective_path, metadata


def _build_fixed_train_command(
    *,
    release_record_path: Path,
    candidate_manifest_path: Path,
    validation_summary_path: Path,
    data_config_path: str,
    train_config_path: Path,
    output_dir: Path,
    submit: bool,
) -> list[str]:
    command = [
        sys.executable,
        "run_retraining_fixed_train_smoke.py",
        "--release-record",
        str(release_record_path),
        "--candidate-manifest",
        str(candidate_manifest_path),
        "--validation-summary",
        str(validation_summary_path),
        "--data-config",
        data_config_path,
        "--train-config",
        str(train_config_path),
        "--output-dir",
        str(output_dir),
    ]
    if submit:
        command.append("--submit")
    return command


def _invoke_fixed_train_bridge(
    *,
    release_record_path: Path,
    candidate_manifest_path: Path,
    validation_summary_path: Path,
    data_config_path: str,
    train_config_path: Path,
    output_dir: Path,
    submit: bool,
) -> dict[str, object]:
    completed = subprocess.run(
        _build_fixed_train_command(
            release_record_path=release_record_path,
            candidate_manifest_path=candidate_manifest_path,
            validation_summary_path=validation_summary_path,
            data_config_path=data_config_path,
            train_config_path=train_config_path,
            output_dir=output_dir,
            submit=submit,
        ),
        check=True,
        text=True,
        capture_output=True,
    )
    del completed
    summary_path = output_dir / "retraining_fixed_train_summary.json"
    summary_payload = _load_json(summary_path)
    submission = summary_payload.get("submission", {})
    job_name = submission.get("job_name") if isinstance(submission, Mapping) else None
    status = summary_payload.get("status")
    return {
        "summary_path": summary_path,
        "status": status if isinstance(status, str) else "unknown",
        "job_name": job_name if isinstance(job_name, str) else None,
    }


def _build_summary(
    *,
    status: str,
    context: UpstreamContext,
    winner_family: str | None,
    hpo_run_dir: Path | None,
    exported_train_config_path: Path | None,
    export_metadata_path: Path | None,
    downstream_summary_path: Path | None,
    submitted_job_name: str | None,
) -> dict[str, object]:
    return {
        "created_at_utc": _utc_timestamp(),
        "status": status,
        "selected_path": _selected_path(context.selection_payload),
        "validation_status": _validation_status(context.validation_summary),
        "release_record_path": str(context.release_record_path),
        "selection_path": str(context.selection_path),
        "candidate_manifest_path": str(context.candidate_manifest_path),
        "validation_summary_path": str(context.validation_summary_path),
        "hpo_smoke_summary_path": str(context.hpo_smoke_summary_path),
        "winner_family": winner_family,
        "hpo_run_dir": str(hpo_run_dir) if hpo_run_dir else None,
        "exported_train_config_path": str(exported_train_config_path) if exported_train_config_path else None,
        "export_metadata_path": str(export_metadata_path) if export_metadata_path else None,
        "downstream_summary_path": str(downstream_summary_path) if downstream_summary_path else None,
        "submitted_job_name": submitted_job_name,
    }


def _build_report(summary: Mapping[str, object], export_metadata: Mapping[str, object] | None) -> str:
    lines = [
        "# Retraining HPO To Fixed-Train Report",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Selected path: `{summary.get('selected_path')}`",
        f"- Winner family: `{summary.get('winner_family')}`",
        f"- Exported train config: `{summary.get('exported_train_config_path')}`",
        f"- Downstream summary: `{summary.get('downstream_summary_path')}`",
        f"- Submitted fixed-train job: `{summary.get('submitted_job_name')}`",
    ]
    if export_metadata:
        lines.extend(
            [
                "",
                "## Train Config",
                f"- Source kind: `{export_metadata.get('source_kind')}`",
                f"- Source path: `{export_metadata.get('source_path')}`",
                f"- Consistency checks: `{', '.join(cast(list[str], export_metadata.get('consistency_checks', [])))}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _write_manifest(
    *,
    context: UpstreamContext,
    output_dir: Path,
    status: str,
    summary_path: Path,
    report_path: Path,
    manifest_path: Path,
    export_metadata_path: Path | None,
    exported_train_config_path: Path | None,
    downstream_summary_path: Path | None,
    winner_family: str | None,
    submitted_job_name: str | None,
    hpo_run_dir: Path | None,
    base_train_config_path: Path,
) -> None:
    manifest = build_step_manifest(step_name="run_retraining_hpo_to_fixed_train", stage_name="model_sweep")
    merge_config(
        manifest,
        config_paths={
            "release_record": context.release_record_path,
            "selection": context.selection_path,
            "candidate_manifest": context.candidate_manifest_path,
            "validation_summary": context.validation_summary_path,
            "hpo_smoke_summary": context.hpo_smoke_summary_path,
            "base_train_config": base_train_config_path,
        },
        overrides={"submit": status == STATUS_SUBMITTED},
    )
    merge_section(
        manifest,
        "inputs",
        {
            "release_record": {"path": context.release_record_path},
            "selection": {"path": context.selection_path},
            "candidate_manifest": {"path": context.candidate_manifest_path},
            "validation_summary": {"path": context.validation_summary_path},
            "hpo_smoke_summary": {"path": context.hpo_smoke_summary_path},
            "hpo_run_dir": {"path": hpo_run_dir} if hpo_run_dir else None,
        },
    )
    # Remove the optional None input produced above.
    if "hpo_run_dir" in manifest["inputs"] and manifest["inputs"]["hpo_run_dir"] is None:
        del manifest["inputs"]["hpo_run_dir"]
    merge_section(
        manifest,
        "outputs",
        {
            "retraining_hpo_to_fixed_train_summary": {"path": summary_path},
            "retraining_hpo_to_fixed_train_report": {"path": report_path},
            "retraining_hpo_to_fixed_train_manifest": {"path": manifest_path},
            "retraining_exported_train_config": {"path": export_metadata_path}
            if export_metadata_path
            else None,
            "effective_train_config": {"path": exported_train_config_path}
            if exported_train_config_path
            else None,
            "downstream_fixed_train_summary": {"path": downstream_summary_path}
            if downstream_summary_path
            else None,
        },
    )
    for key in ("retraining_exported_train_config", "effective_train_config", "downstream_fixed_train_summary"):
        if key in manifest["outputs"] and manifest["outputs"][key] is None:
            del manifest["outputs"][key]
    merge_section(
        manifest,
        "step_specific",
        {
            "retraining_hpo_to_fixed_train": {
                "winner_family": winner_family,
                "submitted_job_name": submitted_job_name,
            }
        },
    )
    finalize_manifest(manifest, output_path=manifest_path, status=status)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continue a completed retraining HPO run into a fixed-train smoke rerun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--release-record", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--validation-summary", required=True)
    parser.add_argument("--hpo-smoke-summary", required=True)
    parser.add_argument("--hpo-run-dir", default=None)
    parser.add_argument("--hpo-job-name", default=None)
    parser.add_argument("--resource-group", default=None)
    parser.add_argument("--workspace-name", default=None)
    parser.add_argument("--download-path", default=None)
    parser.add_argument("--base-train-config", default="configs/train.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "retraining_hpo_to_fixed_train_summary.json"
    report_path = output_dir / "retraining_hpo_to_fixed_train_report.md"
    manifest_path = output_dir / "retraining_hpo_to_fixed_train_manifest" / "step_manifest.json"
    export_metadata_path = output_dir / "retraining_exported_train_config.json"
    downstream_output_dir = output_dir / "selected-bridge"

    context = _load_context(args)

    def write_blocked(status: str, winner_family: str | None = None, hpo_run_dir: Path | None = None) -> None:
        summary = _build_summary(
            status=status,
            context=context,
            winner_family=winner_family,
            hpo_run_dir=hpo_run_dir,
            exported_train_config_path=None,
            export_metadata_path=None,
            downstream_summary_path=None,
            submitted_job_name=None,
        )
        _write_json(summary_path, summary)
        _write_text(report_path, _build_report(summary, None))
        _write_manifest(
            context=context,
            output_dir=output_dir,
            status=status,
            summary_path=summary_path,
            report_path=report_path,
            manifest_path=manifest_path,
            export_metadata_path=None,
            exported_train_config_path=None,
            downstream_summary_path=None,
            winner_family=winner_family,
            submitted_job_name=None,
            hpo_run_dir=hpo_run_dir,
            base_train_config_path=Path(args.base_train_config),
        )

    if _validation_status(context.validation_summary) != "passed":
        write_blocked(STATUS_BLOCKED_BY_VALIDATION)
        raise SystemExit("Validation must be passed before HPO continuation")

    if _selected_path(context.selection_payload) != EXPECTED_SELECTION_PATH:
        write_blocked(STATUS_BLOCKED_BY_SELECTION)
        raise SystemExit("Selection must be model_sweep before HPO continuation")

    hpo_status = context.hpo_smoke_summary.get("status")
    hpo_job_name = _hpo_submission_job_name(context.hpo_smoke_summary)
    if hpo_status != EXPECTED_HPO_STATUS or hpo_job_name is None:
        write_blocked(STATUS_BLOCKED_BY_HPO_STATUS)
        raise SystemExit("HPO smoke summary must be submitted and expose a job name")

    if args.hpo_run_dir is None:
        assert args.hpo_job_name is not None
        assert args.resource_group is not None
        assert args.workspace_name is not None
        hpo_job_status = _wait_for_hpo_job_completion(
            job_name=args.hpo_job_name,
            resource_group=args.resource_group,
            workspace_name=args.workspace_name,
        )
        if hpo_job_status != "completed":
            write_blocked(STATUS_BLOCKED_BY_HPO_STATUS, hpo_run_dir=None)
            raise SystemExit(
                f"HPO parent job {args.hpo_job_name!r} did not complete successfully for continuation; status={hpo_job_status!r}"
            )
    hpo_run_dir = _resolve_hpo_run_dir(
        hpo_run_dir=Path(args.hpo_run_dir) if args.hpo_run_dir else None,
        hpo_job_name=args.hpo_job_name,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
        download_path=Path(args.download_path or output_dir / f"downloaded_hpo_{hpo_job_name}"),
    )
    try:
        artifacts = _resolve_winner_artifacts(hpo_run_dir)
        winner_family, consistency_checks = _validate_winner_consistency(artifacts)
        effective_train_config_path, export_metadata = _materialize_effective_train_config(
            artifacts=artifacts,
            output_dir=output_dir,
            base_config_path=Path(args.base_train_config),
        )
        if consistency_checks:
            export_metadata["consistency_checks"] = consistency_checks + [
                check
                for check in cast(list[str], export_metadata.get("consistency_checks", []))
                if check not in consistency_checks
            ]
        _write_json(export_metadata_path, export_metadata)

        downstream = _invoke_fixed_train_bridge(
            release_record_path=context.release_record_path,
            candidate_manifest_path=context.candidate_manifest_path,
            validation_summary_path=context.validation_summary_path,
            data_config_path=_downstream_data_config(context.selection_payload, context.candidate_manifest),
            train_config_path=effective_train_config_path,
            output_dir=downstream_output_dir,
            submit=bool(args.submit),
        )
    except SystemExit:
        write_blocked(STATUS_BLOCKED_BY_WINNER_INCONSISTENCY, hpo_run_dir=hpo_run_dir)
        raise
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as error:
        summary = _build_summary(
            status=STATUS_CONTINUATION_FAILED,
            context=context,
            winner_family=None,
            hpo_run_dir=hpo_run_dir,
            exported_train_config_path=None,
            export_metadata_path=None,
            downstream_summary_path=None,
            submitted_job_name=None,
        )
        _write_json(summary_path, summary)
        _write_text(report_path, _build_report(summary, None))
        _write_manifest(
            context=context,
            output_dir=output_dir,
            status=STATUS_CONTINUATION_FAILED,
            summary_path=summary_path,
            report_path=report_path,
            manifest_path=manifest_path,
            export_metadata_path=None,
            exported_train_config_path=None,
            downstream_summary_path=None,
            winner_family=None,
            submitted_job_name=None,
            hpo_run_dir=hpo_run_dir,
            base_train_config_path=Path(args.base_train_config),
        )
        raise SystemExit("HPO continuation failed; see retraining_hpo_to_fixed_train_summary.json") from error

    status = STATUS_SUBMITTED if cast(str, downstream["status"]) == STATUS_SUBMITTED else STATUS_DRY_RUN_READY
    summary = _build_summary(
        status=status,
        context=context,
        winner_family=winner_family,
        hpo_run_dir=hpo_run_dir,
        exported_train_config_path=effective_train_config_path,
        export_metadata_path=export_metadata_path,
        downstream_summary_path=cast(Path, downstream["summary_path"]),
        submitted_job_name=cast(Optional[str], downstream["job_name"]),
    )
    _write_json(summary_path, summary)
    export_metadata_payload = _load_json(export_metadata_path)
    _write_text(report_path, _build_report(summary, export_metadata_payload))
    _write_manifest(
        context=context,
        output_dir=output_dir,
        status=status,
        summary_path=summary_path,
        report_path=report_path,
        manifest_path=manifest_path,
        export_metadata_path=export_metadata_path,
        exported_train_config_path=effective_train_config_path,
        downstream_summary_path=cast(Path, downstream["summary_path"]),
        winner_family=winner_family,
        submitted_job_name=cast(Optional[str], downstream["job_name"]),
        hpo_run_dir=hpo_run_dir,
        base_train_config_path=Path(args.base_train_config),
    )
    print(f"Retraining HPO-to-fixed-train summary: {summary_path}")


if __name__ == "__main__":
    main()
