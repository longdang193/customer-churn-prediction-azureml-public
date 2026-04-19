"""
@meta
name: azureml_inputs
type: utility
domain: azure-ml
responsibility:
  - Provide shared AML Input builders for asset and local-file based workflows.
  - Keep common input-construction details out of orchestration entrypoints.
inputs:
  - Data asset name/version
  - Local file paths
outputs:
  - Azure ML Input objects
tags:
  - azure-ml
  - inputs
  - orchestration
features:
  - model-training-pipeline
  - notebook-hpo
capabilities:
  - fixed-train.use-azureml-adapters
  - hpo.reuse-shared-src-azureml-client-input-adapters-instead
lifecycle:
  status: active
"""

from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from azure.ai.ml import Input  # type: ignore[import-not-found]


def build_uri_folder_input(path: str, *, mode: str = "mount") -> Input:
    """Build a uri_folder input from an arbitrary path or AML URI."""
    return Input(
        type="uri_folder",
        path=path,
        mode=mode,
    )


def build_asset_input(asset_name: str, asset_version: str) -> Input:
    """Build a mounted uri_folder input from a versioned AML data asset."""
    return build_uri_folder_input(
        f"azureml:{asset_name}:{asset_version}",
        mode="mount",
    )


def build_local_file_input(path: Path) -> Input:
    """Build a downloaded uri_file input from a local path."""
    return Input(
        type="uri_file",
        path=str(path),
        mode="download",
    )


def build_local_or_uri_folder_input(path: str, *, mode: str = "mount") -> tuple[Input, Path | None]:
    """Build a uri_folder input, staging a local file into a temp folder when needed."""
    candidate = Path(path)
    if candidate.exists():
        if candidate.is_dir():
            return build_uri_folder_input(str(candidate), mode=mode), None
        stage_root = Path.cwd() / ".tmp-tests"
        stage_root.mkdir(parents=True, exist_ok=True)
        staged_dir = stage_root / f"aml-tabular-input-{uuid.uuid4().hex}"
        staged_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(candidate, staged_dir / candidate.name)
        return build_uri_folder_input(str(staged_dir), mode=mode), staged_dir

    return build_uri_folder_input(path, mode=mode), None
