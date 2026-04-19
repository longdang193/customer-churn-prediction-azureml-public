"""
@meta
type: test
scope: unit
domain: release-monitoring-orchestration
covers:
  - Repeatable monitoring handoff orchestration from a saved release record
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
from types import SimpleNamespace
import uuid

import pytest


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"monitor-handoff-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _completed(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def _command_targets(command: list[str], script_name: str) -> bool:
    return any(script_name in part for part in command)


def test_main_runs_capture_download_and_monitor_from_saved_release_record(monkeypatch) -> None:
    """
    @proves online-deploy.hand-off-saved-release-truth-repeatable-monitoring-first
    @proves online-deploy.support-exact-caller-side-blob-capture-retrieval-repeatable
    @proves online-deploy.provide-release-evidence-monitor-stage-retraining-policy-can
    @proves online-deploy.provide-enough-release-monitor-provenance-later-retraining-candidate
    @proves online-deploy.provide-enough-release-monitor-provenance-later-post-validation
    @proves online-deploy.accept-optional-post-release-monitoring-handoff-continuation-retraining
    @proves monitor.support-one-thin-monitoring-first-automation-consumes-saved
    @proves monitor.treat-blob-backed-caller-capture-exact-path-evidence
    """
    import run_monitor_handoff

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "handoff"
    capture_calls: list[Path] = []
    monitor_calls: list[tuple[Path, Path]] = []
    download_calls: list[tuple[str, Path]] = []

    try:
        release_record_path = temp_dir / "release_record.json"
        release_record = {
            "status": "succeeded",
            "registered_model": {"name": "churn-model", "version": "11"},
            "deployment": {
                "endpoint_name": "churn-endpoint",
                "deployment_name": "blue",
            },
            "deployment_capture": {
                "status": "disabled",
                "enabled": False,
                "evidence_plane": "deployment_owned",
            },
        }
        _write_json(release_record_path, release_record)

        probe_a = temp_dir / "probe_a.json"
        probe_b = temp_dir / "probe_b.json"
        probe_a.write_text("{}", encoding="utf-8")
        probe_b.write_text("{}", encoding="utf-8")

        capture_paths = [
            "azureml://datastores/workspaceblobstore/paths/capture/session_a.jsonl",
            "azureml://datastores/workspaceblobstore/paths/capture/session_b.jsonl",
        ]

        def fake_invoke_capture(
            *,
            endpoint_name: str,
            deployment_name: str,
            model_name: str,
            model_version: str,
            request_file: Path,
            capture_config: str,
            azure_config: str,
            manifest_path: Path,
        ) -> Path:
            assert endpoint_name == "churn-endpoint"
            assert deployment_name == "blue"
            assert model_name == "churn-model"
            assert model_version == "11"
            assert capture_config == "configs/inference_capture_blob.yaml"
            assert azure_config == "config.env"
            capture_calls.append(request_file)
            capture_path = capture_paths[0] if request_file == probe_a else capture_paths[1]
            _write_json(
                manifest_path,
                {
                    "capture_status": "captured",
                    "capture_path": capture_path,
                    "response_preview": "[0]",
                },
            )
            return manifest_path

        def fake_invoke_monitor(
            *,
            release_record_path: Path,
            monitor_config: str,
            capture_dir: Path,
            monitor_dir: Path,
        ) -> Path:
            assert monitor_config == "configs/monitor.yaml"
            monitor_calls.append((release_record_path, capture_dir))
            summary_path = monitor_dir / "monitor_summary.json"
            _write_json(
                summary_path,
                {
                    "monitor_status": "capture_backed",
                    "capture_status": "healthy",
                    "capture_record_count": 2,
                    "evidence_level": "repo_owned_inference_capture",
                    "runtime_contract": "repo_owned_scoring_proven",
                },
            )
            return summary_path

        def fake_download_exact_capture(
            *,
            capture_uri: str,
            destination_dir: Path,
            **_kwargs: object,
        ) -> Path:
            download_calls.append((capture_uri, destination_dir))
            local_path = destination_dir / Path(capture_uri).name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text('{"prediction":"0"}\n', encoding="utf-8")
            return local_path

        monkeypatch.setattr(run_monitor_handoff, "_invoke_capture", fake_invoke_capture)
        monkeypatch.setattr(run_monitor_handoff, "_invoke_monitor", fake_invoke_monitor)
        monkeypatch.setattr(run_monitor_handoff, "download_exact_capture", fake_download_exact_capture)
        monkeypatch.setattr(
            run_monitor_handoff,
            "load_caller_capture_settings",
            lambda _path: SimpleNamespace(
                storage_connection_string_env="INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING",
                storage_container_env="INFERENCE_CAPTURE_STORAGE_CONTAINER",
            ),
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_monitor_handoff.py",
                "--release-record",
                str(release_record_path),
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

        run_monitor_handoff.main()

        summary = json.loads((output_dir / "handoff_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "succeeded"
        assert summary["release"]["status"] == "succeeded"
        assert summary["release"]["runtime_contract"] == "repo_owned_scoring_proven"
        assert summary["caller_capture"]["status"] == "retrieved"
        assert summary["caller_capture"]["record_count"] == 2
        assert summary["monitor"]["status"] == "capture_backed"
        assert summary["handoff"]["status"] == "capture_backed_monitoring_ready"
        assert capture_calls == [probe_a, probe_b]
        assert monitor_calls == [(release_record_path, output_dir / "downloaded_capture")]
        assert [call[0] for call in download_calls] == capture_paths
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_writes_partial_failure_summary_when_monitor_step_fails(monkeypatch) -> None:
    import run_monitor_handoff

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "handoff"

    try:
        release_record_path = temp_dir / "release_record.json"
        _write_json(
            release_record_path,
            {
                "status": "succeeded",
                "registered_model": {"name": "churn-model", "version": "11"},
                "deployment": {
                    "endpoint_name": "churn-endpoint",
                    "deployment_name": "blue",
                },
            },
        )
        probe = temp_dir / "probe.json"
        probe.write_text("{}", encoding="utf-8")

        def fake_invoke_capture(**kwargs: object) -> Path:
            manifest_path = Path(str(kwargs["manifest_path"]))
            _write_json(
                manifest_path,
                {
                    "capture_status": "captured",
                    "capture_path": "azureml://datastores/workspaceblobstore/paths/capture/session.jsonl",
                },
            )
            return manifest_path

        def fake_invoke_monitor(**kwargs: object) -> Path:
            raise subprocess.CalledProcessError(returncode=2, cmd=["run_monitor.py"], stderr="monitor failed")

        monkeypatch.setattr(run_monitor_handoff, "_invoke_capture", fake_invoke_capture)
        monkeypatch.setattr(run_monitor_handoff, "_invoke_monitor", fake_invoke_monitor)
        monkeypatch.setattr(
            run_monitor_handoff,
            "download_exact_capture",
            lambda **kwargs: (Path(kwargs["destination_dir"]) / "session.jsonl"),
        )
        monkeypatch.setattr(
            run_monitor_handoff,
            "load_caller_capture_settings",
            lambda _path: SimpleNamespace(
                storage_connection_string_env="INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING",
                storage_container_env="INFERENCE_CAPTURE_STORAGE_CONTAINER",
            ),
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_monitor_handoff.py",
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

        with pytest.raises(SystemExit, match="Monitoring handoff automation failed"):
            run_monitor_handoff.main()

        summary = json.loads((output_dir / "handoff_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "failed"
        assert summary["failure"]["stage"] == "monitor"
        assert summary["release"]["status"] == "succeeded"
        assert summary["caller_capture"]["status"] == "retrieved"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_main_classifies_capture_download_failure_as_capture_stage(monkeypatch) -> None:
    """
    @proves monitor.treat-blob-backed-caller-capture-exact-path-evidence
    """
    import run_monitor_handoff

    temp_dir = _make_temp_dir()
    output_dir = temp_dir / "handoff"

    try:
        release_record_path = temp_dir / "release_record.json"
        _write_json(
            release_record_path,
            {
                "status": "succeeded",
                "registered_model": {"name": "churn-model", "version": "11"},
                "deployment": {
                    "endpoint_name": "churn-endpoint",
                    "deployment_name": "blue",
                },
            },
        )
        probe = temp_dir / "probe.json"
        probe.write_text("{}", encoding="utf-8")

        def fake_invoke_capture(**kwargs: object) -> Path:
            manifest_path = Path(str(kwargs["manifest_path"]))
            _write_json(
                manifest_path,
                {
                    "capture_status": "healthy",
                    "capture_path": "azureblob://storage/container/monitoring/session.jsonl",
                },
            )
            return manifest_path

        def fake_download_exact_capture(**_kwargs: object) -> Path:
            raise subprocess.CalledProcessError(returncode=1, cmd=["download_capture_blob.py"], stderr="blob missing")

        monkeypatch.setattr(run_monitor_handoff, "_invoke_capture", fake_invoke_capture)
        monkeypatch.setattr(run_monitor_handoff, "download_exact_capture", fake_download_exact_capture)
        monkeypatch.setattr(
            run_monitor_handoff,
            "load_caller_capture_settings",
            lambda _path: SimpleNamespace(
                storage_connection_string_env="INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING",
                storage_container_env="INFERENCE_CAPTURE_STORAGE_CONTAINER",
            ),
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_monitor_handoff.py",
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

        with pytest.raises(SystemExit, match="Monitoring handoff automation failed"):
            run_monitor_handoff.main()

        summary = json.loads((output_dir / "handoff_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "failed"
        assert summary["failure"]["stage"] == "capture"
        assert summary["caller_capture"]["status"] == "failed"
        assert summary["monitor"]["status"] is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_handoff_summary_prefers_monitor_capture_truth_when_available() -> None:
    import run_monitor_handoff

    release_record = {
        "status": "succeeded",
        "deployment": {"repo_owned_scoring_status": "repo_owned_scoring_proven"},
    }
    summary = run_monitor_handoff._build_handoff_summary(
        status="succeeded",
        release_record_path=Path("release_record.json"),
        release_record=release_record,
        capture_manifest_paths=[Path("capture/capture_manifest_0.json")],
        capture_downloads=[],
        monitor_summary_path=Path("monitor/monitor_summary.json"),
        monitor_summary={
            "monitor_status": "limited_but_healthy",
            "evidence_level": "release_evidence_only",
            "caller_capture": {
                "status": "not_run",
                "record_count": 0,
                "evidence_plane": "caller_side",
            },
            "deployment_capture": {
                "status": "disabled",
                "enabled": False,
                "evidence_plane": "deployment_owned",
            },
        },
    )

    assert summary["deployment_capture"]["status"] == "disabled"
    assert summary["caller_capture"]["status"] == "not_run"
    assert summary["caller_capture"]["record_count"] == 0
    assert summary["handoff"]["status"] == "release_evidence_only_ready"
