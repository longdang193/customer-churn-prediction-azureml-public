"""
@meta
type: test
scope: unit
domain: release-monitoring-orchestration
covers:
  - Release plus monitor smoke orchestration
excludes:
  - Real Azure ML or Blob storage calls
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
    temp_dir = TEST_TEMP_ROOT / f"release-monitor-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _completed(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def _command_targets(command: list[str], script_name: str) -> bool:
    return any(script_name in part for part in command)


def test_main_runs_release_capture_download_and_monitor(monkeypatch) -> None:
    """
    @proves online-deploy.compose-release-follow-up-automation-through-thin-wrapper
    @proves online-deploy.hand-off-deployed-artifacts-release-metadata-smoke-test
    @proves monitor.support-one-thin-release-plus-monitor-automation-resolves
    @proves monitor.treat-blob-backed-caller-capture-exact-path-evidence
    """
    import run_release_monitor_smoke

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "automation"
    commands: list[list[str]] = []
    download_calls: list[tuple[str, Path]] = []

    try:
        release_record = {
            "status": "succeeded",
            "registered_model": {"name": "churn-model", "version": "7"},
            "deployment": {"endpoint_name": "churn-endpoint", "deployment_name": "blue"},
        }
        capture_paths = [
            "azureml://datastores/workspaceblobstore/paths/capture/session/request_a.jsonl",
            "azureml://datastores/workspaceblobstore/paths/capture/session/request_b.jsonl",
        ]

        probe_a = temp_dir / "probe_a.json"
        probe_b = temp_dir / "probe_b.json"
        probe_a.write_text("{}", encoding="utf-8")
        probe_b.write_text("{}", encoding="utf-8")

        def fake_run(command: list[str], check: bool, text: bool = True, capture_output: bool = False):
            assert check is True
            assert text is True
            commands.append(command)
            if _command_targets(command, "run_release.py"):
                release_path = output_dir / "release" / "job-123" / "release_record.json"
                _write_json(release_path, release_record)
                return _completed(command)
            if _command_targets(command, "run_inference_capture.py"):
                manifest_path = Path(command[command.index("--output-manifest") + 1])
                request_file = Path(command[command.index("--request-file") + 1])
                capture_path = capture_paths[0] if request_file == probe_a else capture_paths[1]
                _write_json(
                    manifest_path,
                    {
                        "capture_status": "captured",
                        "capture_path": capture_path,
                        "response_preview": "[1]",
                    },
                )
                return _completed(command)
            if _command_targets(command, "run_monitor.py"):
                summary_path = output_dir / "monitor" / "monitor_summary.json"
                _write_json(
                    summary_path,
                    {
                        "monitor_status": "capture_backed",
                        "capture_status": "healthy",
                        "capture_record_count": 2,
                        "evidence_level": "repo_owned_inference_capture",
                    },
                )
                return _completed(command)
            raise AssertionError(command)

        def fake_download_exact_capture(
            *,
            capture_uri: str,
            destination_dir: Path,
            **_kwargs: object,
        ) -> Path:
            download_calls.append((capture_uri, destination_dir))
            local_path = destination_dir / Path(capture_uri).name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text('{"prediction":"1"}\n', encoding="utf-8")
            return local_path

        monkeypatch.setattr(run_release_monitor_smoke.subprocess, "run", fake_run)
        monkeypatch.setattr(run_release_monitor_smoke, "download_exact_capture", fake_download_exact_capture)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_release_monitor_smoke.py",
                "--job-name",
                "job-123",
                "--config",
                "config.env",
                "--capture-config",
                "configs/inference_capture_blob.yaml",
                "--monitor-config",
                "configs/monitor.yaml",
                "--output-dir",
                str(output_dir),
                "--probe-request",
                str(probe_a),
                "--probe-request",
                str(probe_b),
            ],
        )

        run_release_monitor_smoke.main()

        summary_path = output_dir / "release_monitor_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["status"] == "succeeded"
        assert summary["release"]["status"] == "succeeded"
        assert summary["deployment_capture"]["status"] == "unknown"
        assert summary["caller_capture"]["status"] == "retrieved"
        assert summary["caller_capture"]["record_count"] == 2
        assert summary["monitor"]["status"] == "capture_backed"
        assert summary["monitor"]["evidence_level"] == "repo_owned_inference_capture"
        assert summary["handoff"]["status"] == "capture_backed_monitoring_ready"
        assert len(summary["capture"]["manifest_paths"]) == 2
        assert len(summary["capture"]["downloaded_files"]) == 2
        assert len(commands) == 4
        assert _command_targets(commands[0], "run_release.py")
        assert _command_targets(commands[1], "run_inference_capture.py")
        assert _command_targets(commands[2], "run_inference_capture.py")
        assert _command_targets(commands[3], "run_monitor.py")
        assert [call[0] for call in download_calls] == capture_paths
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_reuses_existing_release_record_without_running_release(monkeypatch) -> None:
    import run_release_monitor_smoke

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "automation"
    commands: list[list[str]] = []

    try:
        release_record_path = temp_dir / "existing_release_record.json"
        _write_json(
            release_record_path,
            {
                "status": "succeeded",
                "registered_model": {"name": "churn-model", "version": "9"},
                "deployment": {"endpoint_name": "churn-endpoint", "deployment_name": "blue"},
            },
        )
        probe = temp_dir / "probe.json"
        probe.write_text("{}", encoding="utf-8")

        def fake_run(command: list[str], check: bool, text: bool = True, capture_output: bool = False):
            commands.append(command)
            if _command_targets(command, "run_inference_capture.py"):
                manifest_path = Path(command[command.index("--output-manifest") + 1])
                _write_json(
                    manifest_path,
                    {
                        "capture_status": "captured",
                        "capture_path": "azureml://capture/probe.jsonl",
                        "response_preview": "[0]",
                    },
                )
                return _completed(command)
            if _command_targets(command, "run_monitor.py"):
                _write_json(
                    output_dir / "monitor" / "monitor_summary.json",
                    {
                        "monitor_status": "capture_backed",
                        "capture_status": "healthy",
                        "capture_record_count": 1,
                        "evidence_level": "repo_owned_inference_capture",
                    },
                )
                return _completed(command)
            raise AssertionError(command)

        monkeypatch.setattr(run_release_monitor_smoke.subprocess, "run", fake_run)
        monkeypatch.setattr(
            run_release_monitor_smoke,
            "download_exact_capture",
            lambda **kwargs: (Path(kwargs["destination_dir"]) / "probe.jsonl"),
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_release_monitor_smoke.py",
                "--release-record",
                str(release_record_path),
                "--capture-config",
                "configs/inference_capture_blob.yaml",
                "--monitor-config",
                "configs/monitor.yaml",
                "--output-dir",
                str(output_dir),
                "--probe-request",
                str(probe),
            ],
        )

        run_release_monitor_smoke.main()

        assert all(not _command_targets(command, "run_release.py") for command in commands)
        summary = json.loads((output_dir / "release_monitor_summary.json").read_text(encoding="utf-8"))
        assert summary["release"]["record_path"] == str(release_record_path)
        assert summary["status"] == "succeeded"
        assert summary["caller_capture"]["status"] == "retrieved"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_collect_capture_paths_uses_manifest_paths_only() -> None:
    import run_release_monitor_smoke

    temp_dir = _make_temp_dir()
    try:
        manifest_dir = temp_dir / "capture_manifests"
        stale_dir = temp_dir / "stale"
        stale_dir.mkdir()
        (stale_dir / "orphan.json").write_text("{}", encoding="utf-8")

        manifest_a = manifest_dir / "capture_a.json"
        manifest_b = manifest_dir / "capture_b.json"
        _write_json(manifest_a, {"capture_path": "azureml://capture/a.jsonl"})
        _write_json(manifest_b, {"capture_path": "azureml://capture/b.jsonl"})

        capture_paths = run_release_monitor_smoke.collect_capture_paths([manifest_a, manifest_b])

        assert capture_paths == ["azureml://capture/a.jsonl", "azureml://capture/b.jsonl"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_writes_partial_failure_summary_when_capture_step_fails(monkeypatch) -> None:
    import run_release_monitor_smoke

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "automation"

    try:
        probe = temp_dir / "probe.json"
        probe.write_text("{}", encoding="utf-8")

        def fake_run(command: list[str], check: bool, text: bool = True, capture_output: bool = False):
            if _command_targets(command, "run_release.py"):
                _write_json(
                    output_dir / "release" / "job-123" / "release_record.json",
                    {
                        "status": "succeeded",
                        "registered_model": {"name": "churn-model", "version": "7"},
                        "deployment": {"endpoint_name": "churn-endpoint", "deployment_name": "blue"},
                    },
                )
                return _completed(command)
            if _command_targets(command, "run_inference_capture.py"):
                raise subprocess.CalledProcessError(returncode=2, cmd=command, stderr="boom")
            raise AssertionError(command)

        monkeypatch.setattr(run_release_monitor_smoke.subprocess, "run", fake_run)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_release_monitor_smoke.py",
                "--job-name",
                "job-123",
                "--config",
                "config.env",
                "--capture-config",
                "configs/inference_capture_blob.yaml",
                "--monitor-config",
                "configs/monitor.yaml",
                "--output-dir",
                str(output_dir),
                "--probe-request",
                str(probe),
            ],
        )

        with pytest.raises(SystemExit, match="Release-plus-monitor automation failed"):
            run_release_monitor_smoke.main()

        summary = json.loads((output_dir / "release_monitor_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "failed"
        assert summary["failure"]["stage"] == "capture"
        assert summary["release"]["status"] == "succeeded"
        assert summary["release"]["record_path"] == str(output_dir / "release" / "job-123" / "release_record.json")
        assert summary["caller_capture"]["status"] == "failed"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
