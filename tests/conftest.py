"""
@meta
name: tests_conftest
type: utility
domain: testing
responsibility:
  - Ensure repo-root modules and src-root modules are importable during unit tests.
inputs:
  - Repository root path
outputs:
  - Stable Python import path for test execution
tags:
  - test-infra
  - ci-safe
lifecycle:
  status: active
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
