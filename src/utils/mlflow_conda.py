"""Helpers for normalizing MLflow conda environments for Azure serving.

@meta
name: mlflow_conda
type: module
domain: utils
responsibility:
  - Provide utils behavior for `src/utils/mlflow_conda.py`.
inputs: []
outputs: []
tags:
  - utils
lifecycle:
  status: active
"""

from __future__ import annotations

from pathlib import Path


MLFLOW_DEPLOYABLE_PYTHON_SPEC = "python=3.9"
AZUREML_MONITORING_REQUIREMENT = "azureml-ai-monitoring==1.0.0"
AZUREML_INFERENCE_SERVER_REQUIREMENT = "azureml-inference-server-http"
AZURE_STORAGE_BLOB_REQUIREMENT = "azure-storage-blob==12.19.0"
AZURE_SERVING_PIP_REQUIREMENTS = (
    AZUREML_MONITORING_REQUIREMENT,
    AZUREML_INFERENCE_SERVER_REQUIREMENT,
    AZURE_STORAGE_BLOB_REQUIREMENT,
)


def normalize_mlflow_conda_for_azure_serving(conda_path: Path) -> None:
    """Patch an MLflow conda spec in place so Azure ML can build and serve it."""
    if not conda_path.exists():
        return

    normalized_lines = [
        f"- {MLFLOW_DEPLOYABLE_PYTHON_SPEC}"
        if line.strip().startswith("- python=")
        else line
        for line in conda_path.read_text(encoding="utf-8").splitlines()
    ]

    pip_section_index = next(
        (
            index
            for index, line in enumerate(normalized_lines)
            if line.strip() == "- pip:"
        ),
        None,
    )
    if pip_section_index is None:
        normalized_lines.append("- pip:")
        pip_section_index = len(normalized_lines) - 1

    pip_entries = {
        normalized_lines[index].strip()[2:].strip()
        for index in range(pip_section_index + 1, len(normalized_lines))
        if normalized_lines[index].startswith("  - ")
    }
    insert_index = pip_section_index + 1
    while (
        insert_index < len(normalized_lines)
        and normalized_lines[insert_index].startswith("  - ")
    ):
        insert_index += 1

    required_lines = [
        f"  - {requirement}"
        for requirement in AZURE_SERVING_PIP_REQUIREMENTS
        if requirement not in pip_entries
    ]
    if required_lines:
        normalized_lines[insert_index:insert_index] = required_lines

    conda_path.write_text("\n".join(normalized_lines) + "\n", encoding="utf-8")
