"""
@meta
type: test
scope: unit
domain: inference-serving
covers:
  - Repo-owned online scoring request preparation and response shaping
  - Collector initialization and bounded collector warning behavior
excludes:
  - Real Azure ML endpoint containers or collector backends
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any, cast
import uuid

import pandas as pd
import pytest


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"online-scoring-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_score_module_supports_top_level_azure_import(monkeypatch) -> None:
    score_path = Path(__file__).resolve().parents[1] / "src" / "inference" / "score.py"
    repo_root = str(score_path.parents[2])
    src_root = str(score_path.parents[1])
    monkeypatch.setattr(
        sys,
        "path",
        [path for path in sys.path if path not in {repo_root, src_root}],
    )
    spec = importlib.util.spec_from_file_location("azure_score_entrypoint", score_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("azure_score_entrypoint", None)

    spec.loader.exec_module(module)

    assert callable(module.init)
    assert callable(module.run)


def test_online_scoring_service_validates_payload_scores_and_collects() -> None:
    from src.inference.online_scoring import (
        InferenceCollectorSettings,
        OnlineScoringService,
        create_collector_bundle,
    )

    collected_inputs: list[pd.DataFrame] = []
    collected_outputs: list[pd.DataFrame] = []

    class FakeCollector:
        def __init__(self, *, name: str, on_error: object | None = None) -> None:
            self.name = name
            self.on_error = on_error

        def collect(
            self,
            dataframe: pd.DataFrame,
            context: object | None = None,
        ) -> object:
            if self.name == "model_inputs":
                collected_inputs.append(dataframe.copy())
                return {"collector": self.name}
            collected_outputs.append(dataframe.copy())
            return context

    class FakeModel:
        def predict(self, dataframe: pd.DataFrame) -> list[int]:
            assert list(dataframe.columns) == [f"feature_{index}" for index in range(3)]
            return [0]

    collector_bundle = create_collector_bundle(
        InferenceCollectorSettings(
            enabled=True,
            mode="azureml_data_collector",
            sample_rate=1.0,
            rolling_rate="Day",
            collect_inputs=True,
            collect_outputs=True,
            inputs_name="model_inputs",
            outputs_name="model_outputs",
        ),
        collector_factory=lambda **kwargs: FakeCollector(**kwargs),
    )
    service = OnlineScoringService(model=FakeModel(), collector_bundle=collector_bundle)

    response = service.run(
        json.dumps({"input_data": [[1.0, 2.0, 3.0]]}),
        expected_feature_count=3,
    )

    assert response == [0]
    assert len(collected_inputs) == 1
    assert len(collected_outputs) == 1
    assert collected_inputs[0].to_dict(orient="records") == [
        {"feature_0": 1.0, "feature_1": 2.0, "feature_2": 3.0}
    ]
    assert collected_outputs[0].to_dict(orient="records") == [{"prediction": 0}]


def test_online_scoring_service_uses_model_feature_names_when_available() -> None:
    from src.inference.online_scoring import CollectorBundle, InferenceCollectorSettings, OnlineScoringService

    class FakeModel:
        _model_impl = type(
            "Impl",
            (),
            {
                "sklearn_model": type(
                    "SkModel",
                    (),
                    {"feature_names_in_": ["Age", "Balance", "CreditScore"]},
                )()
            },
        )()

        def predict(self, dataframe: pd.DataFrame) -> list[int]:
            assert list(dataframe.columns) == ["Age", "Balance", "CreditScore"]
            return [1]

    service = OnlineScoringService(
        model=FakeModel(),
        collector_bundle=CollectorBundle(
            settings=InferenceCollectorSettings(
                enabled=False,
                mode="release_evidence_only",
                sample_rate=1.0,
                rolling_rate="Day",
                collect_inputs=False,
                collect_outputs=False,
                inputs_name="model_inputs",
                outputs_name="model_outputs",
            ),
            status="disabled",
        ),
        feature_columns=["Age", "Balance", "CreditScore"],
    )

    response = service.run(
        json.dumps({"input_data": [[1.0, 2.0, 3.0]]}),
        expected_feature_count=3,
    )

    assert response == [1]


def test_online_scoring_service_rejects_invalid_payload_without_collector_work() -> None:
    from src.inference.online_scoring import (
        InferenceCollectorSettings,
        OnlineScoringService,
        create_collector_bundle,
    )

    collect_called = False

    class FakeCollector:
        def __init__(self, *, name: str, on_error: object | None = None) -> None:
            self.name = name
            self.on_error = on_error

        def collect(self, dataframe: pd.DataFrame, context: object | None = None) -> object:
            nonlocal collect_called
            collect_called = True
            return context

    class FakeModel:
        def predict(self, dataframe: pd.DataFrame) -> list[int]:
            raise AssertionError("model.predict should not run for invalid payloads")

    collector_bundle = create_collector_bundle(
        InferenceCollectorSettings(
            enabled=True,
            mode="azureml_data_collector",
            sample_rate=1.0,
            rolling_rate="Day",
            collect_inputs=True,
            collect_outputs=True,
            inputs_name="model_inputs",
            outputs_name="model_outputs",
        ),
        collector_factory=lambda **kwargs: FakeCollector(**kwargs),
    )
    service = OnlineScoringService(model=FakeModel(), collector_bundle=collector_bundle)

    with pytest.raises(ValueError, match="expected 3 features"):
        service.run(json.dumps({"input_data": [[1.0, 2.0]]}), expected_feature_count=3)

    assert collect_called is False


def test_create_collector_bundle_records_degraded_state_when_sdk_is_missing() -> None:
    from src.inference.online_scoring import (
        InferenceCollectorSettings,
        create_collector_bundle,
    )

    collector_bundle = create_collector_bundle(
        InferenceCollectorSettings(
            enabled=True,
            mode="azureml_data_collector",
            sample_rate=1.0,
            rolling_rate="Day",
            collect_inputs=True,
            collect_outputs=True,
            inputs_name="model_inputs",
            outputs_name="model_outputs",
        ),
        collector_factory=None,
        collector_import_error=ModuleNotFoundError("azureml.ai.monitoring"),
    )

    assert collector_bundle.enabled is True
    assert collector_bundle.status == "degraded"
    assert collector_bundle.warnings == [
        "azureml.ai.monitoring Collector is unavailable in the serving environment."
    ]


def test_score_init_exposes_module_level_collectors(monkeypatch) -> None:
    from src.inference.online_scoring import (
        CollectorBundle,
        InferenceCollectorSettings,
        OnlineScoringService,
    )
    import src.inference.score as score_module

    class FakeModel:
        def predict(self, dataframe: pd.DataFrame) -> list[int]:
            return [0] * len(dataframe)

    class FakeCollector:
        def collect(
            self,
            dataframe: pd.DataFrame,
            context: object | None = None,
        ) -> object:
            del dataframe, context
            return None

    inputs = FakeCollector()
    outputs = FakeCollector()
    service = OnlineScoringService(
        model=FakeModel(),
            collector_bundle=CollectorBundle(
                settings=InferenceCollectorSettings(
                    enabled=True,
                    mode="azureml_data_collector",
                    sample_rate=1.0,
                    rolling_rate="Day",
                    collect_inputs=True,
                    collect_outputs=True,
                    inputs_name="model_inputs",
                outputs_name="model_outputs",
            ),
            status="healthy",
            inputs_collector=inputs,
            outputs_collector=outputs,
        ),
    )

    monkeypatch.setattr(score_module, "build_online_scoring_service", lambda: service)
    score_module._SERVICE = None
    score_module.inputs_collector = None
    score_module.outputs_collector = None

    score_module.init()

    assert score_module._SERVICE is service
    assert score_module.inputs_collector is inputs
    assert score_module.outputs_collector is outputs


def test_score_init_emits_repo_owned_proof_marker(monkeypatch, capsys) -> None:
    from src.inference.online_scoring import (
        CollectorBundle,
        InferenceCollectorSettings,
        OnlineScoringService,
    )
    import src.inference.score as score_module

    class FakeModel:
        def predict(self, dataframe: pd.DataFrame) -> list[int]:
            return [0] * len(dataframe)

    service = OnlineScoringService(
        model=FakeModel(),
        collector_bundle=CollectorBundle(
            settings=InferenceCollectorSettings(
                enabled=False,
                mode="release_evidence_only",
                sample_rate=1.0,
                rolling_rate="Day",
                collect_inputs=False,
                collect_outputs=False,
                inputs_name="model_inputs",
                outputs_name="model_outputs",
            ),
            status="disabled",
        ),
    )

    monkeypatch.setattr(score_module, "build_online_scoring_service", lambda: service)
    score_module._SERVICE = None
    score_module.inputs_collector = None
    score_module.outputs_collector = None

    score_module.init()

    captured = capsys.readouterr()
    assert (
        f"{score_module.REPO_OWNED_SCORER_INIT_PREFIX}"
        f"{score_module.REPO_OWNED_SCORER_MODE}"
    ) in captured.out


def test_score_run_emits_repo_owned_request_proof_marker_without_changing_response(
    monkeypatch,
    capsys,
) -> None:
    import src.inference.score as score_module

    class FakeService:
        collector_bundle = type(
            "Bundle",
            (),
            {"inputs_collector": None, "outputs_collector": None},
        )()

        def run(self, raw_data: object) -> list[int]:
            assert raw_data == {"input_data": [[1.0, 2.0, 3.0]]}
            return [1]

    class FakeUuid:
        hex = "proof123"

    score_module._SERVICE = cast(Any, FakeService())
    monkeypatch.setattr(score_module.uuid, "uuid4", lambda: FakeUuid())

    response = score_module.run({"input_data": [[1.0, 2.0, 3.0]]})

    captured = capsys.readouterr()
    assert response == [1]
    assert f"{score_module.REPO_OWNED_SCORER_RUN_PREFIX}proof123" in captured.out


def test_online_scoring_service_writes_repo_owned_capture_record() -> None:
    from src.inference.capture import (
        FileInferenceCaptureSink,
        RepoOwnedInferenceCaptureRuntime,
        RepoOwnedInferenceCaptureSettings,
    )
    from src.inference.online_scoring import CollectorBundle, InferenceCollectorSettings, OnlineScoringService

    temp_dir = _make_temp_dir()
    try:
        class FakeModel:
            def predict(self, dataframe: pd.DataFrame) -> list[int]:
                return [1] * len(dataframe)

        capture_runtime = RepoOwnedInferenceCaptureRuntime(
            settings=RepoOwnedInferenceCaptureSettings(
                enabled=True,
                mode="jsonl_file",
                sample_rate=1.0,
                max_rows_per_request=1,
                capture_inputs=True,
                capture_outputs=True,
                redact_inputs=False,
                output_path=str(temp_dir),
            ),
            sink=FileInferenceCaptureSink(temp_dir),
            endpoint_name="churn-endpoint",
            deployment_name="blue",
            model_name="churn-model",
            model_version="12",
            random_value_factory=lambda: 0.0,
        )
        service = OnlineScoringService(
            model=FakeModel(),
            collector_bundle=CollectorBundle(
                settings=InferenceCollectorSettings(
                    enabled=False,
                    mode="release_evidence_only",
                    sample_rate=1.0,
                    rolling_rate="Day",
                    collect_inputs=False,
                    collect_outputs=False,
                    inputs_name="model_inputs",
                    outputs_name="model_outputs",
                ),
                status="disabled",
            ),
            capture_runtime=capture_runtime,
        )

        response = service.run(
            json.dumps({"input_data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}),
            expected_feature_count=3,
        )

        assert response == [1, 1]
        records = list(temp_dir.rglob("*.jsonl"))
        assert len(records) == 1
        payload = json.loads(records[0].read_text(encoding="utf-8").splitlines()[0])
        assert payload["endpoint_name"] == "churn-endpoint"
        assert payload["deployment_name"] == "blue"
        assert payload["row_count"] == 2
        assert payload["captured_row_count"] == 1
        assert payload["outputs"] == [1]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_online_scoring_service_uses_workspace_blob_capture_runtime(monkeypatch) -> None:
    from src.inference.online_scoring import build_online_scoring_service

    captured: dict[str, object] = {}

    class FakeModel:
        def predict(self, dataframe: pd.DataFrame) -> list[int]:
            return [0] * len(dataframe)

    monkeypatch.setenv("INFERENCE_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("INFERENCE_CAPTURE_MODE", "workspaceblobstore_jsonl")
    monkeypatch.setenv("INFERENCE_CAPTURE_OUTPUT_PATH", "monitoring/inference_capture")
    monkeypatch.setenv("INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true")
    monkeypatch.setenv("INFERENCE_CAPTURE_STORAGE_CONTAINER", "monitoring")
    monkeypatch.setenv("INFERENCE_CAPTURE_SESSION_ID", "session-123")
    monkeypatch.setenv("ONLINE_ENDPOINT_NAME", "churn-endpoint")
    monkeypatch.setenv("ONLINE_DEPLOYMENT_NAME", "blue")
    monkeypatch.setenv("ONLINE_MODEL_NAME", "churn-model")
    monkeypatch.setenv("ONLINE_MODEL_VERSION", "12")

    def fake_load_model(model_root=None, *, model_loader=None):
        del model_root, model_loader
        return FakeModel()

    def fake_create_capture_runtime(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("src.inference.online_scoring.load_model", fake_load_model)
    monkeypatch.setattr(
        "src.inference.online_scoring.create_capture_runtime",
        fake_create_capture_runtime,
    )

    service = build_online_scoring_service()

    assert service.capture_runtime is None
    settings = cast(Any, captured["settings"])
    assert settings.mode == "workspaceblobstore_jsonl"
    assert settings.storage_connection_string == "UseDevelopmentStorage=true"
    assert settings.storage_container == "monitoring"
    assert settings.session_id == "session-123"
