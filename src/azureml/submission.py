"""
@meta
name: azureml_submission
type: utility
domain: azure-ml
responsibility:
  - Submit Azure ML jobs while filtering known benign SDK console noise.
  - Preserve real submission output and exceptions for operator-facing scripts.
inputs:
  - Azure ML jobs client
  - Azure ML job entity
outputs:
  - Job submission result
tags:
  - azure-ml
  - submission
  - ergonomics
lifecycle:
  status: active
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import logging
import sys
from typing import Any, TextIO
import warnings


_KNOWN_AZURE_NOISE_SUBSTRINGS = (
    "This is an experimental class, and may change at any time.",
    "pathOnCompute is not a known attribute of class",
)
_AZURE_NOISE_LOGGER_NAMES = ("", "azure", "azure.ai.ml")
_FILTERS_INSTALLED = False


def _is_known_azure_noise(line: str) -> bool:
    return any(fragment in line for fragment in _KNOWN_AZURE_NOISE_SUBSTRINGS)


class _KnownAzureNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _is_known_azure_noise(record.getMessage())


def install_azure_console_noise_filters() -> None:
    """Install process-level filters for known benign Azure SDK submission noise."""
    global _FILTERS_INSTALLED
    if _FILTERS_INSTALLED:
        return

    warnings.filterwarnings(
        "ignore",
        message=r".*This is an experimental class, and may change at any time\..*",
    )

    noise_filter = _KnownAzureNoiseFilter()
    for logger_name in _AZURE_NOISE_LOGGER_NAMES:
        logger = logging.getLogger(logger_name)
        logger.addFilter(noise_filter)
        for handler in logger.handlers:
            handler.addFilter(noise_filter)

    _FILTERS_INSTALLED = True


def _replay_filtered_output(buffer: str, *, stream: TextIO) -> None:
    for line in buffer.splitlines():
        if not line or _is_known_azure_noise(line):
            continue
        print(line, file=stream)


def submit_job_quietly(jobs_client: Any, job: Any) -> Any:
    """Submit an Azure ML job while suppressing known benign SDK console noise."""
    install_azure_console_noise_filters()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        submission = jobs_client.create_or_update(job)
    _replay_filtered_output(stdout_buffer.getvalue(), stream=sys.stdout)
    _replay_filtered_output(stderr_buffer.getvalue(), stream=sys.stderr)
    return submission
