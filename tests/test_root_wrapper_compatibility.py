"""
@meta
type: test
scope: unit
domain: repo-layout
covers:
  - Root helper wrappers remaining aligned with their tools-based implementations
  - The compatibility-only nature of representative root helper command surfaces
excludes:
  - End-to-end command execution
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations


def test_monitor_handoff_root_wrapper_aliases_tools_module() -> None:
    import run_monitor_handoff
    from tools.monitoring import run_monitor_handoff as impl

    assert run_monitor_handoff is impl


def test_inspect_hpo_run_root_wrapper_aliases_tools_module() -> None:
    import inspect_hpo_run
    from tools.hpo import inspect_hpo_run as impl

    assert inspect_hpo_run is impl


def test_export_hpo_winner_config_root_wrapper_aliases_tools_module() -> None:
    import export_hpo_winner_config
    from tools.hpo import export_hpo_winner_config as impl

    assert export_hpo_winner_config is impl


def test_retraining_candidate_root_wrapper_aliases_tools_module() -> None:
    import run_retraining_candidate
    from tools.retraining import run_retraining_candidate as impl

    assert run_retraining_candidate is impl


def test_release_failure_probe_root_wrapper_aliases_tools_module() -> None:
    import run_release_failure_probe
    from tools.release import run_release_failure_probe as impl

    assert run_release_failure_probe is impl
