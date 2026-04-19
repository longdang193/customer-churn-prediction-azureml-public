"""
@meta
type: test
scope: unit
domain: hpo
covers:
  - Script-first sweep spec construction from configs/hpo.yaml semantics
  - Azure ML SDK compatibility for direct sweep submission
  - Shared search-space builders used by direct and pipeline HPO surfaces
excludes:
  - Real Azure ML workspace submission
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

from pathlib import Path
import sys
import types


def test_build_model_sweep_specs_creates_per_model_search_spaces() -> None:
    """
    @proves hpo.build-sweep-definitions-configs-hpo-yaml-another-selected
    """
    from run_hpo import build_model_sweep_specs

    hpo_config = {
        "experiment_name": "hpo-prod",
        "sweep_display_name": "hpo-prod",
        "metric": "f1",
        "mode": "max",
        "sampling_algorithm": "random",
        "budget": {"max_trials": 4, "max_concurrent": 2},
        "timeouts": {"total_minutes": 60, "trial_minutes": 20},
        "search_space": {
            "model_types": ["rf", "xgboost"],
            "use_smote": ["true"],
            "rf": {"n_estimators": [100, 200], "max_depth": [4, 6]},
            "xgboost": {"n_estimators": [100], "learning_rate": [0.05, 0.1]},
        },
    }

    specs = build_model_sweep_specs(hpo_config)

    assert set(specs) == {"rf", "xgboost"}
    rf_spec = specs["rf"]
    assert rf_spec.display_name == "hpo-prod-rf"
    assert "--model-type rf" in rf_spec.command
    assert "--rf_n_estimators ${{search_space.rf_n_estimators}}" in rf_spec.command
    assert "--model-output ${{outputs.model_output}}" in rf_spec.command
    assert "--candidate-metrics-output ${{outputs.candidate_metrics}}" in rf_spec.command
    assert "--mlflow-model-output ${{outputs.mlflow_model}}" in rf_spec.command
    assert "--manifest-output ${{outputs.train_manifest}}" in rf_spec.command
    assert "--hpo-config ${{inputs.hpo_config}}" in rf_spec.command
    assert "--hpo-manifest-output ${{outputs.hpo_manifest}}" in rf_spec.command
    assert rf_spec.search_space["use_smote"] == ["true"]
    assert rf_spec.search_space["rf_n_estimators"] == [100, 200]
    assert rf_spec.max_total_trials == 4
    assert rf_spec.max_concurrent_trials == 2
    assert rf_spec.trial_timeout_seconds == 1200


def test_build_submission_messages_are_ascii_safe() -> None:
    from run_hpo import build_submission_messages

    messages = build_submission_messages(
        model_name="rf",
        job_name="job-123",
        studio_url="https://example.invalid",
    )

    assert messages == [
        "OK Submitted rf sweep: job-123",
        "  View in Azure ML Studio: https://example.invalid",
    ]
    assert all(message.isascii() for message in messages)


def test_build_sweep_search_space_returns_choice_objects() -> None:
    import run_hpo

    class FakeChoice:
        def __init__(self, values: list[object]) -> None:
            self.values = values

    monkey_module = types.SimpleNamespace(Choice=FakeChoice)
    sys.modules["azure.ai.ml.sweep"] = monkey_module
    spec = run_hpo.SweepSpec(
        model_name="rf",
        command="python run_sweep_trial.py",
        search_space={"rf_n_estimators": [100, 200]},
        metric="f1",
        mode="max",
        sampling_algorithm="random",
        experiment_name="hpo-smoke",
        display_name="hpo-smoke-rf",
        max_total_trials=1,
        max_concurrent_trials=1,
        timeout_seconds=60,
        trial_timeout_seconds=60,
    )

    search_space = run_hpo.build_sweep_search_space(spec)

    assert search_space["rf_n_estimators"].values == [100, 200]


def test_submit_sweeps_normalizes_goal_for_sdk(monkeypatch) -> None:
    import run_hpo

    observed: dict[str, object] = {}

    class FakeChoice:
        def __init__(self, values: list[object]) -> None:
            self.values = values

    class FakeSweepJob:
        def __init__(self) -> None:
            self.display_name = None
            self.experiment_name = None
            self.early_termination = None

        def set_limits(self, **kwargs) -> None:
            observed["limits"] = kwargs

    class FakeBaseCommand:
        def sweep(self, **kwargs):
            observed["goal"] = kwargs["goal"]
            observed["search_space"] = kwargs["search_space"]
            return FakeSweepJob()

    def fake_command(**kwargs):
        observed["command_kwargs"] = kwargs
        return FakeBaseCommand()

    class FakeJobs:
        def create_or_update(self, sweep_job):
            observed["submitted_job"] = sweep_job
            return types.SimpleNamespace(name="fake-sweep", studio_url="https://example.invalid")

    monkeypatch.setitem(
        sys.modules,
        "azure.ai.ml",
        types.SimpleNamespace(Output=lambda type: {"type": type}, command=fake_command),
    )
    monkeypatch.setitem(
        sys.modules,
        "azure.ai.ml.sweep",
        types.SimpleNamespace(Choice=FakeChoice, BanditPolicy=lambda **_kwargs: None),
    )
    monkeypatch.setattr(run_hpo, "get_ml_client", lambda _config_path: types.SimpleNamespace(jobs=FakeJobs()))
    monkeypatch.setattr(run_hpo, "get_pipeline_compute_name", lambda _config_path: "cpu-cluster")
    monkeypatch.setattr(run_hpo.hpo_utils, "load_hpo_config", lambda _config_path: {"early_stopping": {}})

    specs = {
        "logreg": run_hpo.SweepSpec(
            model_name="logreg",
            command="python run_sweep_trial.py",
            search_space={"logreg_C": [1.0]},
            metric="f1",
            mode="max",
            sampling_algorithm="random",
            experiment_name="hpo-smoke",
            display_name="hpo-smoke-logreg",
            max_total_trials=1,
            max_concurrent_trials=1,
            timeout_seconds=60,
            trial_timeout_seconds=60,
        )
    }

    run_hpo.submit_sweeps(
        "azureml://datastores/workspaceblobstore/paths/processed_data/",
        specs,
        "configs/hpo_smoke.yaml",
    )

    assert observed["goal"] == "Maximize"
    assert observed["command_kwargs"]["outputs"]["model_output"] == {"type": "uri_folder"}
    assert observed["command_kwargs"]["outputs"]["candidate_metrics"] == {"type": "uri_file"}
    assert observed["command_kwargs"]["outputs"]["mlflow_model"] == {"type": "uri_folder"}
    assert observed["command_kwargs"]["outputs"]["train_manifest"] == {"type": "uri_folder"}
    assert observed["command_kwargs"]["outputs"]["hpo_manifest"] == {"type": "uri_folder"}
    assert "hpo_config" in observed["command_kwargs"]["inputs"]
    assert observed["search_space"]["logreg_C"].values == [1.0]


def test_run_hpo_main_accepts_explicit_config_path(monkeypatch) -> None:
    """
    @proves hpo.submit-reload-sweep-jobs-azure-ml-run-hpo
    @proves hpo.treat-configs-hpo-smoke-yaml-wiring-artifact-profile
    @proves hpo.provide-review-surface-periodic-re-optimization-becoming-sole
    """
    import run_hpo

    observed: dict[str, object] = {}

    def fake_load_hpo_config(config_path: str | Path | None = None) -> dict[str, object]:
        observed["config_path"] = str(config_path)
        return {
            "experiment_name": "hpo-smoke",
            "sweep_display_name": "hpo-smoke",
            "metric": "f1",
            "mode": "max",
            "sampling_algorithm": "random",
            "budget": {"max_trials": 2, "max_concurrent": 2},
            "timeouts": {"total_minutes": 60, "trial_minutes": 20},
            "search_space": {
                "model_types": ["rf"],
                "rf": {"n_estimators": [100], "max_depth": [4]},
            },
        }

    def fake_submit_sweeps(processed_data_uri: str, specs: dict[str, object], config_path: str | None) -> None:
        observed["processed_data_uri"] = processed_data_uri
        observed["specs"] = specs
        observed["submit_config_path"] = config_path

    monkeypatch.setattr(run_hpo.hpo_utils, "load_hpo_config", fake_load_hpo_config)
    monkeypatch.setattr(run_hpo, "submit_sweeps", fake_submit_sweeps)
    monkeypatch.setattr(run_hpo, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_hpo.py",
            "--config",
            "configs/hpo_smoke.yaml",
            "--processed-data-uri",
            "azureml://jobs/fake/outputs/processed_data",
        ],
    )

    run_hpo.main()

    assert observed["config_path"] == "configs/hpo_smoke.yaml"
    assert observed["submit_config_path"] == "configs/hpo_smoke.yaml"
    assert observed["processed_data_uri"] == "azureml://jobs/fake/outputs/processed_data"
    assert set(observed["specs"]) == {"rf"}


def test_submit_sweeps_imports_bandit_policy_from_sweep_module(monkeypatch) -> None:
    import run_hpo

    observed: dict[str, object] = {}

    class FakeBanditPolicy:
        def __init__(self, evaluation_interval: int, slack_factor: float) -> None:
            observed["policy"] = {
                "evaluation_interval": evaluation_interval,
                "slack_factor": slack_factor,
            }

    class FakeChoice:
        def __init__(self, values: list[object]) -> None:
            self.values = values

    class FakeSweepJob:
        def __init__(self) -> None:
            self.display_name = None
            self.experiment_name = None
            self.early_termination = None

        def set_limits(self, **kwargs) -> None:
            observed["limits"] = kwargs

    class FakeBaseCommand:
        def sweep(self, **kwargs):
            observed["sweep_kwargs"] = kwargs
            return FakeSweepJob()

    def fake_command(**kwargs):
        observed["command_kwargs"] = kwargs
        return FakeBaseCommand()

    class FakeJobs:
        def create_or_update(self, sweep_job):
            observed["submitted_job"] = sweep_job
            return types.SimpleNamespace(name="fake-sweep", studio_url="https://example.invalid")

    fake_ml_module = types.SimpleNamespace(
        Output=lambda type: {"type": type},
        command=fake_command,
    )
    fake_sweep_module = types.SimpleNamespace(
        BanditPolicy=FakeBanditPolicy,
        Choice=FakeChoice,
    )

    monkeypatch.setitem(sys.modules, "azure.ai.ml", fake_ml_module)
    monkeypatch.setitem(sys.modules, "azure.ai.ml.sweep", fake_sweep_module)
    monkeypatch.setattr(
        run_hpo,
        "get_ml_client",
        lambda _config_path: types.SimpleNamespace(jobs=FakeJobs()),
    )
    monkeypatch.setattr(run_hpo, "get_pipeline_compute_name", lambda _config_path: "cpu-cluster")
    monkeypatch.setattr(
        run_hpo.hpo_utils,
        "load_hpo_config",
        lambda _config_path: {
            "early_stopping": {
                "enabled": True,
                "policy": "bandit",
                "evaluation_interval": 2,
                "slack_factor": 0.2,
            }
        },
    )

    specs = {
        "logreg": run_hpo.SweepSpec(
            model_name="logreg",
            command="python run_sweep_trial.py",
            search_space={"logreg_C": [1.0]},
            metric="f1",
            mode="max",
            sampling_algorithm="random",
            experiment_name="hpo-smoke",
            display_name="hpo-smoke-logreg",
            max_total_trials=1,
            max_concurrent_trials=1,
            timeout_seconds=60,
            trial_timeout_seconds=60,
        )
    }

    run_hpo.submit_sweeps(
        "azureml://datastores/workspaceblobstore/paths/processed_data/",
        specs,
        "configs/hpo_smoke.yaml",
    )

    assert observed["policy"] == {"evaluation_interval": 2, "slack_factor": 0.2}
    assert observed["submitted_job"].early_termination.__class__ is FakeBanditPolicy
    assert observed["sweep_kwargs"]["search_space"]["logreg_C"].values == [1.0]


def test_hpo_utils_loads_config_and_filters_null_values() -> None:
    """
    @proves hpo.keep-hpo-utils-py-supporting-shared-utility-code
    """
    import hpo_utils
    import shutil
    import uuid

    temp_dir = Path(__file__).resolve().parents[1] / ".tmp-tests" / f"hpo-utils-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        config_path = temp_dir / "hpo.yaml"
        config_path.write_text(
            "search_space:\n  model_types:\n    - rf\n  rf:\n    max_depth:\n      - 4\n      - null\n",
            encoding="utf-8",
        )

        loaded = hpo_utils.load_hpo_config(config_path)
        filtered = hpo_utils.build_parameter_space(loaded["search_space"])

        assert loaded["search_space"]["model_types"] == ["rf"]
        assert filtered["rf"]["max_depth"] == [4]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
