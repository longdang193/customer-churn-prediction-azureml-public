"""Compatibility wrapper for the moved HPO smoke retraining helper.

@meta
name: run_retraining_hpo_smoke
type: script
domain: hpo
responsibility:
  - Provide hpo behavior for `run_retraining_hpo_smoke.py`.
inputs: []
outputs: []
tags:
  - hpo
lifecycle:
  status: active
"""

from tools.retraining import run_retraining_hpo_smoke as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
