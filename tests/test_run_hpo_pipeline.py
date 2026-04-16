"""
@meta
type: test
scope: unit
domain: hpo-pipeline
covers:
  - HPO pipeline runtime tagging
  - HPO trial and summary component contracts
  - Azure ML pipeline graph construction for validation, prep, sweeps, and summary
excludes:
  - Real Azure ML job submission
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import yaml
from azure.ai.ml import Input


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_build_hpo_lineage_tags_includes_hpo_config_path() -> None:
    from run_hpo_pipeline import build_hpo_lineage_tags
    from run_pipeline import PipelineRuntimeSettings

    runtime = PipelineRuntimeSettings(
        compute_name="cpu-cluster",
        current_data_asset_name="churn-data-smoke-eval",
        current_data_asset_version="1",
        reference_data_asset_name="churn-data-smoke",
        reference_data_asset_version="2",
    )

    tags = build_hpo_lineage_tags(
        runtime,
        data_config_path=Path("configs/data_smoke_eval.yaml"),
        hpo_config_path=Path("configs/hpo_smoke.yaml"),
    )

    assert tags["data_asset"] == "churn-data-smoke-eval"
    assert tags["reference_data_asset"] == "churn-data-smoke"
    assert tags["data_config"] == "configs/data_smoke_eval.yaml"
    assert tags["hpo_config"] == "configs/hpo_smoke.yaml"
    assert tags["validate_component"] == "validate_data:1"


def test_hpo_component_contracts_support_pipeline_sweeps_and_summary() -> None:
    project_root = Path(__file__).resolve().parents[1]
    hpo_trial_component = yaml.safe_load(
        (project_root / "aml" / "components" / "hpo_trial.yaml").read_text(encoding="utf-8")
    )
    collect_component = yaml.safe_load(
        (project_root / "aml" / "components" / "collect_hpo_results.yaml").read_text(encoding="utf-8")
    )

    assert hpo_trial_component["inputs"]["processed_data"]["type"] == "uri_folder"
    assert hpo_trial_component["inputs"]["config"]["type"] == "uri_file"
    assert hpo_trial_component["inputs"]["hpo_config"]["type"] == "uri_file"
    assert hpo_trial_component["inputs"]["model_type"]["type"] == "string"
    assert hpo_trial_component["outputs"]["model_output"]["type"] == "uri_folder"
    assert hpo_trial_component["outputs"]["candidate_metrics"]["type"] == "uri_folder"
    assert hpo_trial_component["outputs"]["mlflow_model"]["type"] == "uri_folder"
    assert hpo_trial_component["outputs"]["train_manifest"]["type"] == "uri_folder"
    assert hpo_trial_component["outputs"]["hpo_manifest"]["type"] == "uri_folder"
    assert "--candidate-metrics-output ${{outputs.candidate_metrics}}" in hpo_trial_component["command"]
    assert "--model-output ${{outputs.model_output}}" in hpo_trial_component["command"]
    assert "--mlflow-model-output ${{outputs.mlflow_model}}" in hpo_trial_component["command"]
    assert "--manifest-output ${{outputs.train_manifest}}" in hpo_trial_component["command"]
    assert "--hpo-config ${{inputs.hpo_config}}" in hpo_trial_component["command"]
    assert "--hpo-manifest-output ${{outputs.hpo_manifest}}" in hpo_trial_component["command"]

    assert collect_component["inputs"]["primary_metric"]["type"] == "string"
    assert collect_component["inputs"]["logreg_metrics"]["type"] == "uri_folder"
    assert collect_component["inputs"]["rf_metrics"]["type"] == "uri_folder"
    assert collect_component["inputs"]["xgboost_metrics"]["type"] == "uri_folder"
    assert collect_component["inputs"]["logreg_metrics"]["optional"] is True
    assert collect_component["inputs"]["rf_metrics"]["optional"] is True
    assert collect_component["inputs"]["xgboost_metrics"]["optional"] is True
    assert collect_component["inputs"]["logreg_hpo_manifest"]["optional"] is True
    assert collect_component["inputs"]["rf_hpo_manifest"]["optional"] is True
    assert collect_component["inputs"]["xgboost_hpo_manifest"]["optional"] is True
    assert collect_component["inputs"]["logreg_model_output"]["optional"] is True
    assert collect_component["inputs"]["rf_model_output"]["optional"] is True
    assert collect_component["inputs"]["xgboost_model_output"]["optional"] is True
    assert collect_component["inputs"]["logreg_mlflow_model"]["optional"] is True
    assert collect_component["inputs"]["rf_mlflow_model"]["optional"] is True
    assert collect_component["inputs"]["xgboost_mlflow_model"]["optional"] is True
    assert collect_component["inputs"]["logreg_train_manifest"]["optional"] is True
    assert collect_component["inputs"]["rf_train_manifest"]["optional"] is True
    assert collect_component["inputs"]["xgboost_train_manifest"]["optional"] is True
    assert collect_component["outputs"]["hpo_summary"]["type"] == "uri_folder"
    assert collect_component["outputs"]["hpo_summary_report"]["type"] == "uri_folder"
    assert collect_component["outputs"]["hpo_manifest"]["type"] == "uri_folder"
    assert "--summary-output ${{outputs.hpo_summary}}" in collect_component["command"]
    assert "--manifest-output ${{outputs.hpo_manifest}}" in collect_component["command"]
    assert "--logreg-hpo-manifest ${{inputs.logreg_hpo_manifest}}" in collect_component["command"]
    assert "--rf-model-output ${{inputs.rf_model_output}}" in collect_component["command"]
    assert "--xgboost-mlflow-model ${{inputs.xgboost_mlflow_model}}" in collect_component["command"]
    assert "--rf-train-manifest ${{inputs.rf_train_manifest}}" in collect_component["command"]

    materialize_component = yaml.safe_load(
        (project_root / "aml" / "components" / "materialize_hpo_winner.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert materialize_component["inputs"]["hpo_summary"]["type"] == "uri_folder"
    assert materialize_component["inputs"]["logreg_metrics"]["type"] == "uri_folder"
    assert materialize_component["inputs"]["rf_metrics"]["type"] == "uri_folder"
    assert materialize_component["inputs"]["xgboost_metrics"]["type"] == "uri_folder"
    assert materialize_component["inputs"]["logreg_model_output"]["optional"] is True
    assert materialize_component["inputs"]["rf_mlflow_model"]["optional"] is True
    assert materialize_component["outputs"]["winner_candidate_metrics"]["type"] == "uri_folder"
    assert materialize_component["outputs"]["winner_model_output"]["type"] == "uri_folder"
    assert materialize_component["outputs"]["winner_mlflow_model"]["type"] == "uri_folder"
    assert materialize_component["outputs"]["winner_train_manifest"]["type"] == "uri_folder"
    assert materialize_component["outputs"]["winner_hpo_manifest"]["type"] == "uri_folder"
    assert materialize_component["outputs"]["winner_train_config"]["type"] == "uri_folder"
    assert materialize_component["outputs"]["winner_manifest"]["type"] == "uri_folder"
    assert materialize_component["inputs"]["base_train_config"]["type"] == "uri_file"
    assert "--base-train-config ${{inputs.base_train_config}}" in materialize_component["command"]
    assert "--winner-train-config ${{outputs.winner_train_config}}" in materialize_component["command"]


def test_define_hpo_pipeline_builds_expected_graph() -> None:
    import hpo_utils
    from run_hpo import build_model_sweep_specs
    from run_hpo_pipeline import define_hpo_pipeline, load_hpo_pipeline_components

    components = load_hpo_pipeline_components(Path("aml/components"))
    specs = build_model_sweep_specs(hpo_utils.load_hpo_config("configs/hpo_smoke.yaml"))
    pipeline = define_hpo_pipeline(
        components,
        specs=specs,
        compute_name="cpu-cluster",
        primary_metric="f1",
        current_data_asset="churn-data-smoke-eval",
        current_data_version="1",
        reference_data_asset="churn-data-smoke",
        reference_data_version="2",
        gate_validation_before_prep=True,
    )

    job = pipeline(
        current_raw_data=Input(type="uri_folder", path="azureml:churn-data-smoke-eval:1"),
        reference_raw_data=Input(type="uri_folder", path="azureml:churn-data-smoke:2"),
        data_config=Input(type="uri_file", path="azureml:data-smoke-eval-yaml:1"),
        train_config=Input(type="uri_file", path="azureml:train-yaml:1"),
        hpo_config=Input(type="uri_file", path="azureml:hpo-smoke-yaml:1"),
    )

    assert set(job.jobs.keys()) == {
        "validate_job",
        "data_prep_job",
        "hpo_logreg_sweep",
        "hpo_rf_sweep",
        "hpo_xgboost_sweep",
        "collect_hpo_results_job",
        "materialize_hpo_winner_job",
    }
    assert job.jobs["hpo_logreg_sweep"].type == "sweep"
    assert job.jobs["hpo_rf_sweep"].type == "sweep"
    assert job.jobs["hpo_xgboost_sweep"].type == "sweep"
    assert str(job.jobs["hpo_logreg_sweep"].objective.goal).lower() == "maximize"
    assert "current_data_asset" in job.jobs["collect_hpo_results_job"].inputs
    assert "reference_data_version" in job.jobs["collect_hpo_results_job"].inputs
    assert "logreg_model_output" in job.jobs["collect_hpo_results_job"].inputs
    assert "xgboost_mlflow_model" in job.jobs["collect_hpo_results_job"].inputs
    assert "logreg_hpo_manifest" in job.jobs["collect_hpo_results_job"].inputs
    assert "rf_train_manifest" in job.jobs["collect_hpo_results_job"].inputs
    assert "base_train_config" in job.jobs["materialize_hpo_winner_job"].inputs
    assert job.jobs["collect_hpo_results_job"].inputs["current_data_asset"].type == "string"
    assert job.jobs["collect_hpo_results_job"].inputs["reference_data_version"].type == "string"
    assert set(job.outputs.keys()) == {
        "validation_report",
        "validation_summary",
        "validation_manifest",
        "data_prep_manifest",
        "logreg_model_output",
        "logreg_mlflow_model",
        "logreg_candidate_metrics",
        "logreg_train_manifest",
        "logreg_hpo_manifest",
        "rf_model_output",
        "rf_mlflow_model",
        "rf_candidate_metrics",
        "rf_train_manifest",
        "rf_hpo_manifest",
        "xgboost_model_output",
        "xgboost_mlflow_model",
        "xgboost_candidate_metrics",
        "xgboost_train_manifest",
        "xgboost_hpo_manifest",
        "hpo_summary",
        "hpo_summary_report",
        "hpo_manifest",
        "winner_candidate_metrics",
        "winner_model_output",
        "winner_mlflow_model",
        "winner_train_manifest",
        "winner_hpo_manifest",
        "winner_train_config",
        "winner_manifest",
    }


def test_resolve_hpo_config_path_defaults_and_accepts_override() -> None:
    from run_hpo_pipeline import resolve_hpo_config_path

    assert resolve_hpo_config_path() == Path("configs/hpo.yaml")
    assert resolve_hpo_config_path("configs/hpo_smoke.yaml") == Path("configs/hpo_smoke.yaml")


def test_resolve_hpo_data_inputs_stages_local_csv_overrides() -> None:
    from run_hpo_pipeline import (
        HPODataInputOverrides,
        _resolve_hpo_data_inputs,
    )
    from run_pipeline import PipelineRuntimeSettings

    temp_dir = _make_temp_dir()
    try:
        current_csv = temp_dir / "current.csv"
        reference_csv = temp_dir / "reference.csv"
        current_csv.write_text("a,b\n1,2\n", encoding="utf-8")
        reference_csv.write_text("a,b\n3,4\n", encoding="utf-8")

        current_input, reference_input, cleanup_dirs = _resolve_hpo_data_inputs(
            PipelineRuntimeSettings(
                compute_name="cpu-cluster",
                current_data_asset_name="prod-current",
                current_data_asset_version="7",
                reference_data_asset_name="prod-reference",
                reference_data_asset_version="9",
            ),
            overrides=HPODataInputOverrides(
                current_data_override=str(current_csv),
                reference_data_override=str(reference_csv),
            ),
        )

        assert current_input.type == "uri_folder"
        assert reference_input.type == "uri_folder"
        assert len(cleanup_dirs) == 2
        assert (Path(current_input.path) / current_csv.name).exists()
        assert (Path(reference_input.path) / reference_csv.name).exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        for cleanup_dir in locals().get("cleanup_dirs", []):
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def test_main_accepts_explicit_current_and_reference_data_overrides(
    monkeypatch,
) -> None:
    import run_hpo_pipeline

    observed: dict[str, object] = {}

    def fake_build_asset_input(name: str, version: str):
        observed.setdefault("asset_inputs", []).append((name, version))
        return {"type": "asset", "name": name, "version": version}

    def fake_build_local_or_uri_folder_input(path: str):
        observed.setdefault("uri_inputs", []).append(path)
        return {"type": "uri_folder", "path": path}, None

    def fake_build_local_file_input(path: Path):
        return {"type": "uri_file", "path": str(path)}

    class FakePipelineJob:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.experiment_name = None
            self.display_name = None
            self.tags = None

    def fake_pipeline(**kwargs):
        observed["pipeline_kwargs"] = kwargs
        return FakePipelineJob(kwargs)

    monkeypatch.setattr(run_hpo_pipeline, "install_azure_console_noise_filters", lambda: None)
    monkeypatch.setattr(run_hpo_pipeline, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run_hpo_pipeline,
        "get_pipeline_runtime_settings",
        lambda *_args, **_kwargs: run_hpo_pipeline.PipelineRuntimeSettings(
            compute_name="cpu-cluster",
            current_data_asset_name="prod-current",
            current_data_asset_version="7",
            reference_data_asset_name="prod-reference",
            reference_data_asset_version="9",
        ),
    )
    monkeypatch.setattr(
        run_hpo_pipeline,
        "get_hpo_pipeline_metadata",
        lambda _path: {
            "experiment_name": "hpo-smoke",
            "display_name": "hpo-smoke-display",
            "primary_metric": "f1",
        },
    )
    monkeypatch.setattr(run_hpo_pipeline.hpo_utils, "load_hpo_config", lambda _path: {"metric": "f1"})
    monkeypatch.setattr(run_hpo_pipeline, "build_model_sweep_specs", lambda _config: {})
    monkeypatch.setattr(run_hpo_pipeline, "load_hpo_pipeline_components", lambda _path: {})
    monkeypatch.setattr(run_hpo_pipeline, "define_hpo_pipeline", lambda *args, **kwargs: fake_pipeline)
    monkeypatch.setattr(run_hpo_pipeline, "build_early_stopping_policy", lambda _config: None)
    monkeypatch.setattr(run_hpo_pipeline, "get_pipeline_validation_gate", lambda _path: True)
    monkeypatch.setattr(run_hpo_pipeline, "build_asset_input", fake_build_asset_input)
    monkeypatch.setattr(
        run_hpo_pipeline,
        "build_local_or_uri_folder_input",
        fake_build_local_or_uri_folder_input,
    )
    monkeypatch.setattr(run_hpo_pipeline, "build_local_file_input", fake_build_local_file_input)
    monkeypatch.setattr(
        run_hpo_pipeline,
        "build_hpo_lineage_tags",
        lambda *_args, **_kwargs: {"git_commit": "abc123"},
    )
    monkeypatch.setattr(
        run_hpo_pipeline,
        "get_ml_client",
        lambda _config: type("FakeClient", (), {"jobs": object()})(),
    )
    monkeypatch.setattr(
        run_hpo_pipeline,
        "submit_job_quietly",
        lambda _jobs, pipeline_job: type(
            "FakeSubmission",
            (),
            {"name": "hpo-job", "studio_url": "https://studio/hpo/job"},
        )(),
    )
    monkeypatch.setattr(
        run_hpo_pipeline,
        "build_submission_messages",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_hpo_pipeline.py",
            "--data-config",
            "configs/data_smoke_eval.yaml",
            "--hpo-config",
            "configs/hpo_smoke.yaml",
            "--current-data-override",
            "azureml://datastores/workspaceblobstore/paths/retraining/current/",
            "--reference-data-override",
            "azureml:approved-reference:9",
        ],
    )

    run_hpo_pipeline.main()

    pipeline_kwargs = observed["pipeline_kwargs"]
    assert observed.get("asset_inputs") in (None, [])
    assert observed["uri_inputs"] == [
        "azureml://datastores/workspaceblobstore/paths/retraining/current/",
        "azureml:approved-reference:9",
    ]
    assert pipeline_kwargs["current_raw_data"]["path"].startswith("azureml://")
    assert pipeline_kwargs["reference_raw_data"]["path"] == "azureml:approved-reference:9"


def test_get_hpo_pipeline_metadata_reads_hpo_config() -> None:
    from run_hpo_pipeline import get_hpo_pipeline_metadata

    temp_dir = _make_temp_dir()
    try:
        hpo_config = temp_dir / "hpo.yaml"
        hpo_config.write_text(
            "\n".join(
                [
                    'experiment_name: "hpo-smoke"',
                    'sweep_display_name: "hpo-graph"',
                    'metric: "f1"',
                ]
            ),
            encoding="utf-8",
        )

        metadata = get_hpo_pipeline_metadata(hpo_config)

        assert metadata == {
            "experiment_name": "hpo-smoke",
            "display_name": "hpo-graph",
            "primary_metric": "f1",
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
