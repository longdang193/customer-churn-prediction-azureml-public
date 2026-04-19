"""
@meta
name: online_scoring
type: utility
domain: inference-serving
responsibility:
  - Provide a repo-owned Azure ML online scoring surface.
  - Reuse request payload validation and optionally log bounded collector evidence.
inputs:
  - Raw endpoint request payloads
  - MLflow model bundle path or AZUREML_MODEL_DIR
  - Inference collector settings
outputs:
  - Endpoint scoring responses
  - Collector status and warning markers in stdout
tags:
  - inference
  - deployment
  - monitoring
features:
  - online-endpoint-deployment
capabilities:
  - online-deploy.configure-approved-model-repo-owned-src-inference-score
lifecycle:
  status: active
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import pandas as pd

from .capture import (
    DEFAULT_CAPTURE_MODE,
    RepoOwnedInferenceCaptureRuntime,
    RepoOwnedInferenceCaptureSettings,
    create_capture_runtime,
)
from .payloads import DEFAULT_ENDPOINT_FEATURE_COUNT, validate_endpoint_payload


COLLECTOR_WARNING_PREFIX = "COLLECTOR_WARNING="
DEFAULT_COLLECTOR_MODE = "azureml_data_collector"
REPO_OWNED_SCORER_MODE = "repo_owned_score_py"
REPO_OWNED_SCORER_INIT_PREFIX = "REPO_OWNED_SCORER_INIT="
REPO_OWNED_SCORER_RUN_PREFIX = "REPO_OWNED_SCORER_RUN="


class SupportsPredict(Protocol):
    def predict(self, dataframe: pd.DataFrame) -> Any:
        ...


class SupportsCollect(Protocol):
    def collect(self, dataframe: pd.DataFrame, context: object | None = None) -> object:
        ...


CollectorFactory = Callable[..., SupportsCollect]
ModelLoader = Callable[[str], SupportsPredict]


@dataclass(frozen=True)
class InferenceCollectorSettings:
    enabled: bool
    mode: str
    sample_rate: float
    rolling_rate: str
    collect_inputs: bool
    collect_outputs: bool
    inputs_name: str
    outputs_name: str

    @classmethod
    def from_environment(cls) -> "InferenceCollectorSettings":
        return cls(
            enabled=os.getenv("INFERENCE_CAPTURE_ENABLED", "false").strip().lower() == "true",
            mode=os.getenv("INFERENCE_CAPTURE_MODE", DEFAULT_CAPTURE_MODE),
            sample_rate=1.0,
            rolling_rate="Day",
            collect_inputs=True,
            collect_outputs=True,
            inputs_name=os.getenv("INFERENCE_INPUTS_NAME", "model_inputs"),
            outputs_name=os.getenv("INFERENCE_OUTPUTS_NAME", "model_outputs"),
        )


@dataclass
class CollectorBundle:
    settings: InferenceCollectorSettings
    status: str
    warnings: list[str] = field(default_factory=list)
    inputs_collector: SupportsCollect | None = None
    outputs_collector: SupportsCollect | None = None

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def collect_inputs(self, input_df: pd.DataFrame) -> object | None:
        if self.inputs_collector is None:
            return None
        try:
            return self.inputs_collector.collect(input_df)
        except Exception as error:  # pragma: no cover - defensive for runtime
            self._record_warning(f"inputs collector collect() failed: {error}")
            return None

    def collect_outputs(self, output_df: pd.DataFrame, context: object | None) -> None:
        if self.outputs_collector is None:
            return
        try:
            self.outputs_collector.collect(output_df, context)
        except Exception as error:  # pragma: no cover - defensive for runtime
            self._record_warning(f"outputs collector collect() failed: {error}")

    def _record_warning(self, warning: str) -> None:
        if warning not in self.warnings:
            self.warnings.append(warning)
            if self.enabled:
                self.status = "degraded"
            logging.warning("%s%s", COLLECTOR_WARNING_PREFIX, warning)


def _default_collector_factory() -> CollectorFactory:
    from azureml.ai.monitoring import Collector  # type: ignore[import-not-found]

    return Collector


def create_collector_bundle(
    settings: InferenceCollectorSettings,
    *,
    collector_factory: CollectorFactory | None = None,
    collector_import_error: Exception | None = None,
) -> CollectorBundle:
    """Build collector instances for repo-owned online scoring."""
    if not settings.enabled:
        return CollectorBundle(settings=settings, status="disabled")

    if settings.mode != DEFAULT_COLLECTOR_MODE:
        return CollectorBundle(settings=settings, status="disabled")

    if collector_import_error is not None:
        return CollectorBundle(
            settings=settings,
            status="degraded",
            warnings=["azureml.ai.monitoring Collector is unavailable in the serving environment."],
        )

    try:
        factory = collector_factory or _default_collector_factory()
    except ModuleNotFoundError:
        return CollectorBundle(
            settings=settings,
            status="degraded",
            warnings=["azureml.ai.monitoring Collector is unavailable in the serving environment."],
        )

    inputs_collector = None
    outputs_collector = None
    warnings: list[str] = []
    status = "healthy"
    try:
        if settings.collect_inputs:
            inputs_collector = factory(name=settings.inputs_name)
        if settings.collect_outputs:
            outputs_collector = factory(name=settings.outputs_name)
    except Exception as error:
        status = "degraded"
        warnings.append(f"collector initialization failed: {error}")

    bundle = CollectorBundle(
        settings=settings,
        status=status,
        warnings=warnings,
        inputs_collector=inputs_collector,
        outputs_collector=outputs_collector,
    )
    for warning in warnings:
        logging.warning("%s%s", COLLECTOR_WARNING_PREFIX, warning)
    return bundle


def _load_model_loader() -> ModelLoader:
    import mlflow.pyfunc

    return mlflow.pyfunc.load_model


def resolve_model_dir(model_root: str | Path | None = None) -> Path:
    """Resolve the MLflow model directory from AML or an explicit path."""
    resolved_root = model_root if model_root is not None else os.getenv("AZUREML_MODEL_DIR", "")
    if not resolved_root:
        raise FileNotFoundError("AZUREML_MODEL_DIR is not set and no model path was provided.")
    root = Path(resolved_root).resolve()
    if (root / "MLmodel").exists():
        return root
    child_candidates = sorted(
        child for child in root.iterdir() if child.is_dir() and (child / "MLmodel").exists()
    )
    if child_candidates:
        return child_candidates[0]
    raise FileNotFoundError(f"Could not locate MLflow model bundle under {root}")


def load_model(model_root: str | Path | None = None, *, model_loader: ModelLoader | None = None) -> SupportsPredict:
    """Load the deployed MLflow model bundle."""
    resolved_dir = resolve_model_dir(model_root)
    loader = model_loader or _load_model_loader()
    return loader(str(resolved_dir))


def infer_model_feature_columns(model: object) -> list[str] | None:
    """Infer trained feature names from an MLflow-wrapped sklearn model when available."""
    direct_feature_names = getattr(model, "feature_names_in_", None)
    if direct_feature_names is not None:
        return [str(column) for column in direct_feature_names]

    model_impl = getattr(model, "_model_impl", None)
    wrapped_model = getattr(model_impl, "sklearn_model", None)
    wrapped_feature_names = getattr(wrapped_model, "feature_names_in_", None)
    if wrapped_feature_names is None:
        return None
    return [str(column) for column in wrapped_feature_names]


def prepare_input_dataframe(
    payload: Mapping[str, object],
    *,
    expected_feature_count: int = DEFAULT_ENDPOINT_FEATURE_COUNT,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Validate and convert the request payload into a model-ready DataFrame."""
    resolved_feature_count = (
        len(feature_columns)
        if feature_columns is not None
        else expected_feature_count
    )
    validated_rows = validate_endpoint_payload(
        payload,
        expected_feature_count=resolved_feature_count,
    )
    columns = (
        list(feature_columns)
        if feature_columns is not None
        else [f"feature_{index}" for index in range(resolved_feature_count)]
    )
    return pd.DataFrame(validated_rows, columns=columns)


def _load_request_payload(raw_request: str | bytes | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(raw_request, Mapping):
        return raw_request
    if isinstance(raw_request, bytes):
        raw_request = raw_request.decode("utf-8")
    parsed = json.loads(raw_request)
    if not isinstance(parsed, Mapping):
        raise ValueError("Endpoint request must be a JSON object.")
    return parsed


def _normalize_predictions(predictions: Any) -> list[Any]:
    if isinstance(predictions, pd.DataFrame):
        if "prediction" in predictions.columns:
            return predictions["prediction"].tolist()
        if len(predictions.columns) == 1:
            return predictions.iloc[:, 0].tolist()
        return predictions.to_dict(orient="records")
    if isinstance(predictions, pd.Series):
        return predictions.tolist()
    if hasattr(predictions, "tolist"):
        result = predictions.tolist()
        return result if isinstance(result, list) else [result]
    if isinstance(predictions, list):
        return predictions
    return [predictions]


def run_with_model_and_collectors(
    *,
    model: SupportsPredict,
    collector_bundle: CollectorBundle,
    raw_request: str | bytes | Mapping[str, object],
    expected_feature_count: int = DEFAULT_ENDPOINT_FEATURE_COUNT,
    feature_columns: Sequence[str] | None = None,
) -> list[Any]:
    """Run scoring directly from a model and collector bundle."""
    payload = _load_request_payload(raw_request)
    input_df = prepare_input_dataframe(
        payload,
        expected_feature_count=expected_feature_count,
        feature_columns=feature_columns,
    )
    context = collector_bundle.collect_inputs(input_df)
    predictions = model.predict(input_df)
    normalized_predictions = _normalize_predictions(predictions)
    output_df = pd.DataFrame({"prediction": normalized_predictions})
    collector_bundle.collect_outputs(output_df, context)
    return normalized_predictions


@dataclass
class OnlineScoringService:
    model: SupportsPredict
    collector_bundle: CollectorBundle
    capture_runtime: RepoOwnedInferenceCaptureRuntime | None = None
    feature_columns: list[str] | None = None

    def run(
        self,
        raw_request: str | bytes | Mapping[str, object],
        *,
        expected_feature_count: int = DEFAULT_ENDPOINT_FEATURE_COUNT,
    ) -> list[Any]:
        predictions = run_with_model_and_collectors(
            model=self.model,
            collector_bundle=self.collector_bundle,
            raw_request=raw_request,
            expected_feature_count=expected_feature_count,
            feature_columns=self.feature_columns,
        )
        if self.capture_runtime is not None:
            payload = _load_request_payload(raw_request)
            input_df = prepare_input_dataframe(
                payload,
                expected_feature_count=expected_feature_count,
                feature_columns=self.feature_columns,
            )
            self.capture_runtime.maybe_capture(
                input_df=input_df,
                predictions=predictions,
            )
        return predictions


def build_online_scoring_service(
    *,
    model_root: str | Path | None = None,
    settings: InferenceCollectorSettings | None = None,
    collector_factory: CollectorFactory | None = None,
    model_loader: ModelLoader | None = None,
) -> OnlineScoringService:
    """Build the repo-owned scoring service for Azure ML managed online deployment."""
    collector_settings = settings or InferenceCollectorSettings.from_environment()
    collector_bundle = create_collector_bundle(
        collector_settings,
        collector_factory=collector_factory,
    )
    model = load_model(model_root, model_loader=model_loader)
    feature_columns = infer_model_feature_columns(model)
    capture_settings = RepoOwnedInferenceCaptureSettings.from_environment()
    capture_runtime = create_capture_runtime(
        settings=capture_settings,
        endpoint_name=os.getenv("ONLINE_ENDPOINT_NAME", "unknown-endpoint"),
        deployment_name=os.getenv("ONLINE_DEPLOYMENT_NAME", "unknown-deployment"),
        model_name=os.getenv("ONLINE_MODEL_NAME", "unknown-model"),
        model_version=os.getenv("ONLINE_MODEL_VERSION", "unknown-version"),
    )
    return OnlineScoringService(
        model=model,
        collector_bundle=collector_bundle,
        capture_runtime=capture_runtime,
        feature_columns=feature_columns,
    )
