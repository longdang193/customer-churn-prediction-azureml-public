"""
@meta
name: online_score_entrypoint
type: script
domain: inference-serving
responsibility:
  - Expose Azure ML managed online endpoint `init()` and `run()` entrypoints.
inputs:
  - AZUREML_MODEL_DIR
  - Endpoint request payloads
outputs:
  - Scoring responses
tags:
  - inference
  - deployment
  - monitoring
lifecycle:
  status: active
"""

from __future__ import annotations

import logging
from pathlib import Path
import importlib
import sys
from typing import Any, TYPE_CHECKING, Protocol, cast
import uuid

class SupportsOnlineScoringModule(Protocol):
    REPO_OWNED_SCORER_INIT_PREFIX: str
    REPO_OWNED_SCORER_MODE: str
    REPO_OWNED_SCORER_RUN_PREFIX: str

    def build_online_scoring_service(self) -> "OnlineScoringService":
        ...


def _load_online_scoring_module() -> object:
    try:
        return importlib.import_module("src.inference.online_scoring")
    except ImportError:
        score_dir = Path(__file__).resolve().parent
        package_root = score_dir.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        return importlib.import_module("inference.online_scoring")


online_scoring_module = cast(SupportsOnlineScoringModule, _load_online_scoring_module())

if TYPE_CHECKING:
    from src.inference.online_scoring import OnlineScoringService

REPO_OWNED_SCORER_INIT_PREFIX = online_scoring_module.REPO_OWNED_SCORER_INIT_PREFIX
REPO_OWNED_SCORER_MODE = online_scoring_module.REPO_OWNED_SCORER_MODE
REPO_OWNED_SCORER_RUN_PREFIX = online_scoring_module.REPO_OWNED_SCORER_RUN_PREFIX
build_online_scoring_service = online_scoring_module.build_online_scoring_service


_SERVICE: OnlineScoringService | None = None
inputs_collector: object | None = None
outputs_collector: object | None = None


def _emit_repo_owned_scorer_marker(prefix: str, marker: str) -> None:
    message = f"{prefix}{marker}"
    print(message, flush=True)
    logging.warning(message)


def init() -> None:
    """Initialize the shared scoring service once per container."""
    global _SERVICE
    _SERVICE = build_online_scoring_service()
    global inputs_collector, outputs_collector
    inputs_collector = _SERVICE.collector_bundle.inputs_collector
    outputs_collector = _SERVICE.collector_bundle.outputs_collector
    _emit_repo_owned_scorer_marker(
        REPO_OWNED_SCORER_INIT_PREFIX,
        REPO_OWNED_SCORER_MODE,
    )


def run(raw_data: Any) -> list[Any]:
    """Score a request with the repo-owned online scoring service."""
    global _SERVICE
    if _SERVICE is None:
        init()
    assert _SERVICE is not None
    _emit_repo_owned_scorer_marker(REPO_OWNED_SCORER_RUN_PREFIX, uuid.uuid4().hex)
    return _SERVICE.run(raw_data)
