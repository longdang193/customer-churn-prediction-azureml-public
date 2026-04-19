"""
@meta
type: test
scope: unit
domain: release-failure-probe
covers:
  - Repo-owned scorer negative cloud probe orchestration
  - Root wrapper compatibility for the release failure probe command
excludes:
  - Real Azure ML endpoint invokes
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import uuid

import pytest


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"release-failure-probe-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _completed(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_main_writes_intentional_failure_summary_when_cloud_probe_hits_scorer_error(monkeypatch) -> None:
    """
    @proves online-deploy.support-one-bounded-negative-cloud-probe-intentionally-bypasses
    @proves monitor.negative-scorer-probes-may-produce-intentional-azure-scoring
    """
    import run_release_failure_probe

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "probe-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        _write_json(
            release_record_path,
            {
                "status": "succeeded",
                "registered_model": {"name": "churn-model", "version": "12"},
                "deployment": {
                    "endpoint_name": "churn-endpoint",
                    "deployment_name": "blue",
                    "repo_owned_scoring_status": "repo_owned_scoring_proven",
                },
                "monitoring_handoff": {
                    "status": "ready_for_basic_monitoring_handoff",
                    "evidence_level": "release_evidence_only",
                },
            },
        )
        bad_payload = temp_dir / "bad_payload.json"
        bad_payload.write_text('{"input_data": [[0,1,2]]}', encoding="utf-8")

        def fake_run(command: list[str], check: bool, text: bool = True, capture_output: bool = True):
            del check, text
            if "invoke" in command:
                assert "--request-file" in command
                assert command[command.index("--request-file") + 1] == str(bad_payload)
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout="",
                    stderr="Message: An unexpected error occurred in scoring script. Check the logs for more info.",
                )
            assert "get-logs" in command
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "REPO_OWNED_SCORER_RUN=proof123\n"
                    "ValueError: Endpoint payload input_data row 0 expected 10 features, got 3.\n"
                ),
                stderr="",
            )

        monkeypatch.setattr(run_release_failure_probe.subprocess, "run", fake_run)
        monkeypatch.setattr(
            run_release_failure_probe,
            "load_azure_config",
            lambda _path: {
                "resource_group": "rg-test",
                "workspace_name": "ws-test",
                "subscription_id": "sub-test",
            },
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_release_failure_probe.py",
                "--release-record",
                str(release_record_path),
                "--request-file",
                str(bad_payload),
                "--output-dir",
                str(output_dir),
            ],
        )

        run_release_failure_probe.main()

        summary = json.loads(
            (output_dir / "failure_probe_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "intentional_failure_observed"
        assert summary["probe_type"] == "repo_owned_scorer_negative_payload"
        assert summary["failure"]["source"] == "repo_owned_scorer"
        assert summary["failure"]["error_type"] == "ValueError"
        assert "expected 10 features, got 3" in summary["failure"]["error_message"]
        assert summary["failure"]["log_excerpt"] == [
            "REPO_OWNED_SCORER_RUN=proof123",
            "ValueError: Endpoint payload input_data row 0 expected 10 features, got 3.",
        ]
        assert summary["artifact_truth"]["release_status"] == "succeeded"
        assert summary["artifact_truth"]["monitor_handoff_status"] == "ready_for_basic_monitoring_handoff"
        assert summary["artifact_truth"]["repo_owned_scoring_status"] == "repo_owned_scoring_proven"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_fails_when_probe_returns_success(monkeypatch) -> None:
    import run_release_failure_probe

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "probe-output"

    try:
        release_record_path = temp_dir / "release_record.json"
        _write_json(
            release_record_path,
            {
                "status": "succeeded",
                "registered_model": {"name": "churn-model", "version": "12"},
                "deployment": {
                    "endpoint_name": "churn-endpoint",
                    "deployment_name": "blue",
                    "repo_owned_scoring_status": "repo_owned_scoring_proven",
                },
                "monitoring_handoff": {"status": "ready_for_basic_monitoring_handoff"},
            },
        )
        bad_payload = temp_dir / "bad_payload.json"
        bad_payload.write_text('{"input_data": [[0,1,2]]}', encoding="utf-8")

        monkeypatch.setattr(
            run_release_failure_probe.subprocess,
            "run",
            lambda command, check, text=True, capture_output=True: _completed(command),
        )
        monkeypatch.setattr(
            run_release_failure_probe,
            "load_azure_config",
            lambda _path: {
                "resource_group": "rg-test",
                "workspace_name": "ws-test",
                "subscription_id": "sub-test",
            },
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_release_failure_probe.py",
                "--release-record",
                str(release_record_path),
                "--request-file",
                str(bad_payload),
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(SystemExit, match="Negative probe did not fail as expected"):
            run_release_failure_probe.main()

        summary = json.loads(
            (output_dir / "failure_probe_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "unexpected_success"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
