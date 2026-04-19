"""Monitoring evaluation helpers.

@meta
name: monitoring
type: module
domain: monitoring
responsibility:
  - Provide monitoring behavior for `src/monitoring/__init__.py`.
inputs: []
outputs: []
tags:
  - monitoring
lifecycle:
  status: active
"""

from .evaluate_release import evaluate_release_monitoring

__all__ = ["evaluate_release_monitoring"]
