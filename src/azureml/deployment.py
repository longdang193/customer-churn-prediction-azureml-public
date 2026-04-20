"""
@meta
name: azureml_deployment
type: utility
domain: azure-ml
responsibility:
  - Bridge Azure ML managed online endpoint deployment operations.
  - Keep deployment and smoke-test orchestration out of release entrypoints.
inputs:
  - MLClient endpoint and deployment operations
  - Registered model reference
  - Deployment config and sample request payload
outputs:
  - Deployment result payload with smoke-test response
tags:
  - azure-ml
  - deployment
  - release
capabilities:
  - online-deploy.configure-approved-model-repo-owned-src-inference-score
  - online-deploy.recover-transient-sdk-side-deployment-operation-failures-when
  - online-deploy.distinguish-delayed-azure-success-terminal-deployment-failure-deployment
  - online-deploy.preserve-truthful-deployment-state-metadata-release-record-when
  - online-deploy.externalize-repo-owned-inference-capture-azure-accessible-jsonl
  - online-deploy.keep-release-smoke-validation-broad-enough-catch-registration
  - online-deploy.reuse-shared-src-azureml-registry-deployment-adapters-release
lifecycle:
  status: active
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import time
from typing import Protocol, TypedDict
import uuid

from typing_extensions import NotRequired

from azure.ai.ml.entities import (  # type: ignore[import-not-found]
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)
from azure.core.exceptions import ResourceNotFoundError  # type: ignore[import-not-found]

from src.config.assets import (
    RELEASE_EVIDENCE_ONLY_MODE,
    deployment_online_base_image,
    load_asset_manifest,
    repo_owned_online_inference_capture_settings,
)
from src.inference import validate_endpoint_payload_file
from src.inference.capture import WORKSPACE_BLOBSTORE_JSONL_CAPTURE_MODE, create_blob_service_client
from src.inference.online_scoring import (
    REPO_OWNED_SCORER_INIT_PREFIX,
    REPO_OWNED_SCORER_MODE,
    REPO_OWNED_SCORER_RUN_PREFIX,
)
from src.utils.mlflow_conda import normalize_mlflow_conda_for_azure_serving


TERMINAL_DEPLOYMENT_STATES = {"Succeeded", "Failed", "Canceled"}
DEFAULT_DEPLOYMENT_POLL_INTERVAL_SECONDS = 30
DEFAULT_DEPLOYMENT_POLL_TIMEOUT_SECONDS = 1800
REPO_OWNED_COLLECTOR_DOWNGRADE_WARNING = (
    "Azure ML data_collector is disabled for repo-owned online scoring; using "
    "release-evidence-only monitoring handoff."
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_CODE_ROOT = PROJECT_ROOT / "src"
DEFAULT_CAPTURE_PROOF_TIMEOUT_SECONDS = 60
DEFAULT_CAPTURE_PROOF_POLL_INTERVAL_SECONDS = 5


class DeploymentFailureRecord(TypedDict):
    failure_stage: str
    error_type: str
    error_message: str


class InferenceCaptureMetadata(TypedDict):
    inference_capture_enabled: bool
    inference_capture_mode: str
    inference_capture_status: str
    inference_capture_warnings: list[str]
    inference_capture_output_path: str | None


class RepoOwnedScoringProofMetadata(TypedDict):
    repo_owned_scoring_expected: bool
    repo_owned_scoring_observed: bool
    repo_owned_scoring_status: str
    repo_owned_scoring_log_markers: list[str]
    repo_owned_scoring_warnings: list[str]


class DeploymentResult(TypedDict):
    endpoint_name: str
    deployment_name: str
    deployment_state: str
    recovery_used: bool
    finalization_timed_out: bool
    traffic_updated: bool
    smoke_invoked: bool
    smoke_test_response: object | None
    inference_capture_enabled: bool
    inference_capture_mode: str
    inference_capture_status: str
    inference_capture_warnings: list[str]
    inference_capture_output_path: str | None
    repo_owned_scoring_expected: bool
    repo_owned_scoring_observed: bool
    repo_owned_scoring_status: str
    repo_owned_scoring_log_markers: list[str]
    repo_owned_scoring_warnings: list[str]
    payload_summary: NotRequired[dict[str, object]]
    failure: NotRequired[DeploymentFailureRecord]


class CaptureProofResult(TypedDict):
    verified: bool
    warnings: list[str]


class OnlineDeploymentOperations(Protocol):
    def begin_create_or_update(self, deployment: object) -> "AzureOperationPoller":
        ...

    def get(self, *, name: str, endpoint_name: str) -> object:
        ...

    def get_logs(
        self,
        name: str,
        endpoint_name: str,
        lines: int,
        *,
        container_type: str | None = None,
        local: bool = False,
    ) -> str:
        ...


class AzureOperationPoller(Protocol):
    def result(self) -> object:
        ...


def _deployment_state(deployment: object) -> str:
    return str(getattr(deployment, "provisioning_state", ""))


def _wait_for_deployment_terminal_state(
    online_deployments: OnlineDeploymentOperations,
    *,
    endpoint_name: str,
    deployment_name: str,
    poll_interval_seconds: int = DEFAULT_DEPLOYMENT_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_DEPLOYMENT_POLL_TIMEOUT_SECONDS,
) -> str:
    """Poll deployment state when Azure keeps provisioning after an SDK error."""
    deadline = time.monotonic() + timeout_seconds
    state = ""
    while time.monotonic() < deadline:
        deployment = online_deployments.get(
            name=deployment_name,
            endpoint_name=endpoint_name,
        )
        state = _deployment_state(deployment)
        if state in TERMINAL_DEPLOYMENT_STATES:
            return state
        time.sleep(poll_interval_seconds)
    return state


def _build_failure(stage: str, error: Exception) -> DeploymentFailureRecord:
    return {
        "failure_stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _resolve_model_bundle_dir(model_bundle_path: Path) -> Path:
    if (model_bundle_path / "conda.yaml").exists():
        return model_bundle_path
    nested_candidates = sorted(
        candidate.parent
        for candidate in model_bundle_path.rglob("conda.yaml")
    )
    if nested_candidates:
        return nested_candidates[0]
    raise FileNotFoundError(
        f"Could not locate conda.yaml under downloaded MLflow model bundle: {model_bundle_path}"
    )


def _inference_capture_metadata(
    *,
    settings_enabled: bool,
    settings_mode: str,
    output_path: str | None,
    extra_warnings: list[str] | None = None,
) -> InferenceCaptureMetadata:
    warnings = list(extra_warnings or [])
    if not settings_enabled:
        status = "disabled"
    elif settings_mode == RELEASE_EVIDENCE_ONLY_MODE:
        status = "disabled"
    elif warnings:
        status = "degraded"
    else:
        status = "configured"
    return {
        "inference_capture_enabled": settings_enabled,
        "inference_capture_mode": settings_mode,
        "inference_capture_status": status,
        "inference_capture_warnings": warnings,
        "inference_capture_output_path": output_path,
    }


def _repo_owned_scoring_proof_metadata(*, expected: bool) -> RepoOwnedScoringProofMetadata:
    if not expected:
        return {
            "repo_owned_scoring_expected": False,
            "repo_owned_scoring_observed": False,
            "repo_owned_scoring_status": "not_applicable",
            "repo_owned_scoring_log_markers": [],
            "repo_owned_scoring_warnings": [],
        }
    return {
        "repo_owned_scoring_expected": True,
        "repo_owned_scoring_observed": False,
        "repo_owned_scoring_status": "inconclusive",
        "repo_owned_scoring_log_markers": [],
        "repo_owned_scoring_warnings": [
            "Repo-owned scorer proof has not been observed in Azure deployment logs yet."
        ],
    }


def _fetch_deployment_logs(
    online_deployments: object,
    *,
    endpoint_name: str,
    deployment_name: str,
    lines: int = 200,
) -> str | None:
    get_logs = getattr(online_deployments, "get_logs", None)
    if not callable(get_logs):
        return None
    try:
        return str(
            get_logs(
                deployment_name,
                endpoint_name,
                lines,
            )
        )
    except TypeError:
        return str(
            get_logs(
                name=deployment_name,
                endpoint_name=endpoint_name,
                lines=lines,
            )
        )
    except Exception:
        return None


def _classify_repo_owned_scoring_contract(
    log_text: str | None,
) -> RepoOwnedScoringProofMetadata:
    expected_metadata = _repo_owned_scoring_proof_metadata(expected=True)
    if not log_text:
        expected_metadata["repo_owned_scoring_warnings"] = [
            "Azure deployment logs were unavailable for repo-owned scorer proof inspection."
        ]
        return expected_metadata

    observed_markers: list[str] = []
    for marker in (REPO_OWNED_SCORER_INIT_PREFIX, REPO_OWNED_SCORER_RUN_PREFIX):
        if marker in log_text:
            observed_markers.append(marker)
    if observed_markers:
        return {
            "repo_owned_scoring_expected": True,
            "repo_owned_scoring_observed": True,
            "repo_owned_scoring_status": "repo_owned_scoring_proven",
            "repo_owned_scoring_log_markers": observed_markers,
            "repo_owned_scoring_warnings": [],
        }

    generated_runtime_clues = (
        "ERROR:entry_module:",
        "inputs_collector is not defined",
        "outputs_collector is not defined",
    )
    if any(clue in log_text for clue in generated_runtime_clues):
        return {
            "repo_owned_scoring_expected": True,
            "repo_owned_scoring_observed": False,
            "repo_owned_scoring_status": "generated_runtime_still_in_control",
            "repo_owned_scoring_log_markers": [],
            "repo_owned_scoring_warnings": [
                "Azure deployment logs did not show repo-owned scorer proof markers after canary invoke."
            ],
        }

    expected_metadata["repo_owned_scoring_warnings"] = [
        "Azure deployment logs remained inconclusive for repo-owned scorer proof."
    ]
    return expected_metadata


def _resolve_capture_runtime_credentials(
    capture_settings: object,
) -> tuple[str | None, str | None, list[str]]:
    warnings: list[str] = []
    connection_string = None
    container_name = None
    connection_env_name = getattr(capture_settings, "storage_connection_string_env", None)
    container_env_name = getattr(capture_settings, "storage_container_env", None)
    if getattr(capture_settings, "mode", "") != WORKSPACE_BLOBSTORE_JSONL_CAPTURE_MODE:
        return None, None, warnings
    if isinstance(connection_env_name, str) and connection_env_name:
        connection_string = os.getenv(connection_env_name)
        if not connection_string:
            warnings.append(
                f"inference capture connection string env var '{connection_env_name}' is not set"
            )
    if isinstance(container_env_name, str) and container_env_name:
        container_name = os.getenv(container_env_name)
        if not container_name:
            warnings.append(
                f"inference capture container env var '{container_env_name}' is not set"
            )
    return connection_string, container_name, warnings


def _capture_session_prefix(
    *,
    output_path: str,
    endpoint_name: str,
    deployment_name: str,
    session_id: str,
    date_utc: str,
) -> str:
    return "/".join(
        [
            output_path.strip("/"),
            date_utc.replace("-", "/"),
            endpoint_name,
            deployment_name,
            session_id,
        ]
    ).strip("/")


def _verify_external_capture_sink(
    *,
    connection_string: str,
    container_name: str,
    output_path: str,
    endpoint_name: str,
    deployment_name: str,
    session_id: str,
    timeout_seconds: int = DEFAULT_CAPTURE_PROOF_TIMEOUT_SECONDS,
    poll_interval_seconds: int = DEFAULT_CAPTURE_PROOF_POLL_INTERVAL_SECONDS,
) -> tuple[bool, list[str]]:
    service_client = create_blob_service_client(connection_string)
    container_client = service_client.get_container_client(container_name)
    try:
        if not container_client.exists():
            return False, [
                (
                    "external inference capture container was not found: "
                    f"{container_name}"
                )
            ]
    except Exception as error:
        return False, [
            (
                "external inference capture container check failed for "
                f"{container_name}: {type(error).__name__}: {error}"
            )
        ]
    date_utc = time.strftime("%Y-%m-%d", time.gmtime())
    prefix = _capture_session_prefix(
        output_path=output_path,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        session_id=session_id,
        date_utc=date_utc,
    )
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            blobs = list(container_client.list_blobs(name_starts_with=prefix))
        except ResourceNotFoundError as error:
            return False, [
                (
                    "external inference capture proof could not inspect "
                    f"{container_name}/{prefix}: {type(error).__name__}: {error}"
                )
            ]
        except Exception as error:
            return False, [
                (
                    "external inference capture proof failed while inspecting "
                    f"{container_name}/{prefix}: {type(error).__name__}: {error}"
                )
            ]
        if blobs:
            return True, []
        time.sleep(poll_interval_seconds)
    return False, [
        (
            "external inference capture proof was not found under "
            f"{container_name}/{prefix} within {timeout_seconds} seconds"
        )
    ]


def _create_or_update_deployment_with_recovery(
    online_deployments: OnlineDeploymentOperations,
    deployment: object,
    *,
    endpoint_name: str,
    deployment_name: str,
) -> tuple[str, bool, DeploymentFailureRecord | None, bool]:
    try:
        online_deployments.begin_create_or_update(deployment).result()
    except Exception as error:
        state = _wait_for_deployment_terminal_state(
            online_deployments,
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
        )
        if state == "Succeeded":
            return state, True, None, False
        if state in TERMINAL_DEPLOYMENT_STATES:
            return state, True, _build_failure("deployment", error), False
        return (
            state,
            True,
            _build_failure("deployment_finalization_timeout", error),
            True,
        )
    return "Succeeded", False, None, False


def deploy_registered_model(
    ml_client: object,
    *,
    registered_model: object,
    endpoint_name: str,
    deployment_name: str,
    instance_type: str,
    instance_count: int,
    sample_data_path: Path,
    model_bundle_path: Path | None = None,
    asset_manifest: Mapping[str, object] | None = None,
) -> DeploymentResult:
    """Create or update the managed online deployment and invoke a smoke test."""
    online_endpoints = getattr(ml_client, "online_endpoints")
    online_deployments = getattr(ml_client, "online_deployments")
    manifest = dict(asset_manifest or load_asset_manifest())
    requested_capture_settings = repo_owned_online_inference_capture_settings(manifest)
    capture_warnings = (
        [REPO_OWNED_COLLECTOR_DOWNGRADE_WARNING]
        if requested_capture_settings.mode == "release_evidence_only"
        else []
    )
    capture_connection_string, capture_container_name, credential_warnings = (
        _resolve_capture_runtime_credentials(requested_capture_settings)
    )
    capture_warnings.extend(credential_warnings)
    capture_session_id = uuid.uuid4().hex if requested_capture_settings.enabled else None
    capture_metadata = _inference_capture_metadata(
        settings_enabled=requested_capture_settings.enabled,
        settings_mode=requested_capture_settings.mode,
        output_path=requested_capture_settings.output_path,
        extra_warnings=capture_warnings,
    )
    repo_owned_scoring_metadata = _repo_owned_scoring_proof_metadata(
        expected=model_bundle_path is not None
    )

    try:
        online_endpoints.get(endpoint_name)
    except ResourceNotFoundError:
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            auth_mode="key",
            description="Online endpoint serving the churn model",
        )
        online_endpoints.begin_create_or_update(endpoint).result()

    if model_bundle_path is not None:
        resolved_bundle_path = _resolve_model_bundle_dir(model_bundle_path)
        normalize_mlflow_conda_for_azure_serving(resolved_bundle_path / "conda.yaml")
        environment = Environment(
            image=deployment_online_base_image(manifest),
            conda_file=resolved_bundle_path / "conda.yaml",
        )
        code_configuration = CodeConfiguration(
            code=str(DEPLOYMENT_CODE_ROOT),
            scoring_script="inference/score.py",
        )
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=registered_model,
            instance_type=instance_type,
            instance_count=instance_count,
            environment=environment,
            code_configuration=code_configuration,
            environment_variables={
                **requested_capture_settings.as_environment_variables(
                    storage_connection_string=capture_connection_string,
                    storage_container=capture_container_name,
                    session_id=capture_session_id,
                ),
                "ONLINE_ENDPOINT_NAME": endpoint_name,
                "ONLINE_DEPLOYMENT_NAME": deployment_name,
                "ONLINE_MODEL_NAME": str(getattr(registered_model, "name", "unknown-model")),
                "ONLINE_MODEL_VERSION": str(getattr(registered_model, "version", "unknown-version")),
                "REPO_OWNED_SCORER_MODE": REPO_OWNED_SCORER_MODE,
            },
        )
    else:
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=registered_model,
            instance_type=instance_type,
            instance_count=instance_count,
        )
    deployment_state, recovery_used, failure, finalization_timed_out = (
        _create_or_update_deployment_with_recovery(
        online_deployments,
        deployment,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
    )
    )

    result: DeploymentResult = {
        "endpoint_name": endpoint_name,
        "deployment_name": deployment_name,
        "deployment_state": deployment_state,
        "recovery_used": recovery_used,
        "finalization_timed_out": finalization_timed_out,
        "traffic_updated": False,
        "smoke_invoked": False,
        "smoke_test_response": None,
        "inference_capture_enabled": capture_metadata["inference_capture_enabled"],
        "inference_capture_mode": capture_metadata["inference_capture_mode"],
        "inference_capture_status": capture_metadata["inference_capture_status"],
        "inference_capture_warnings": capture_metadata["inference_capture_warnings"],
        "inference_capture_output_path": capture_metadata["inference_capture_output_path"],
        "repo_owned_scoring_expected": repo_owned_scoring_metadata["repo_owned_scoring_expected"],
        "repo_owned_scoring_observed": repo_owned_scoring_metadata["repo_owned_scoring_observed"],
        "repo_owned_scoring_status": repo_owned_scoring_metadata["repo_owned_scoring_status"],
        "repo_owned_scoring_log_markers": repo_owned_scoring_metadata["repo_owned_scoring_log_markers"],
        "repo_owned_scoring_warnings": repo_owned_scoring_metadata["repo_owned_scoring_warnings"],
    }
    if failure is not None:
        result["failure"] = failure
        return result

    endpoint = online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    online_endpoints.begin_create_or_update(endpoint).result()
    result["traffic_updated"] = True

    try:
        payload_summary = validate_endpoint_payload_file(sample_data_path)
    except Exception as error:
        result["failure"] = _build_failure("deployment", error)
        return result

    response = online_endpoints.invoke(
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        request_file=str(sample_data_path),
    )
    result["smoke_invoked"] = True
    result["smoke_test_response"] = response
    result["payload_summary"] = payload_summary.to_record()
    if model_bundle_path is not None:
        scoring_contract_metadata = _classify_repo_owned_scoring_contract(
            _fetch_deployment_logs(
                online_deployments,
                endpoint_name=endpoint_name,
                deployment_name=deployment_name,
            )
        )
        result["repo_owned_scoring_observed"] = scoring_contract_metadata["repo_owned_scoring_observed"]
        result["repo_owned_scoring_status"] = scoring_contract_metadata["repo_owned_scoring_status"]
        result["repo_owned_scoring_log_markers"] = scoring_contract_metadata["repo_owned_scoring_log_markers"]
        result["repo_owned_scoring_warnings"] = scoring_contract_metadata["repo_owned_scoring_warnings"]
    if (
        requested_capture_settings.enabled
        and requested_capture_settings.mode == WORKSPACE_BLOBSTORE_JSONL_CAPTURE_MODE
        and capture_connection_string
        and capture_container_name
        and capture_session_id
    ):
        capture_verified, proof_warnings = _verify_external_capture_sink(
            connection_string=capture_connection_string,
            container_name=capture_container_name,
            output_path=requested_capture_settings.output_path,
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            session_id=capture_session_id,
        )
        if capture_verified:
            result["inference_capture_status"] = "healthy"
        else:
            result["inference_capture_status"] = "degraded"
            if proof_warnings:
                result["inference_capture_warnings"] = proof_warnings
    return result
