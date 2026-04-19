"""Helpers for Azure ML folder outputs that should contain named files.

@meta
name: output_paths
type: module
domain: utils
responsibility:
  - Provide utils behavior for `src/utils/output_paths.py`.
inputs: []
outputs: []
tags:
  - utils
lifecycle:
  status: active
"""

from __future__ import annotations

from pathlib import Path


def resolve_named_output_file(path: Path, filename: str) -> Path:
    """Resolve an AML output path to a concrete, extension-bearing file path."""
    if path.suffix:
        return path
    return path / filename


def resolve_named_input_file(path: Path, filename: str) -> Path:
    """Resolve a file or folder input while staying compatible with legacy paths."""
    if path.is_dir():
        return path / filename
    if path.suffix:
        return path
    named_path = path / filename
    return named_path if named_path.exists() else path
