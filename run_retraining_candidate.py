"""Compatibility wrapper for the moved retraining candidate helper.

@meta
name: run_retraining_candidate
type: script
domain: retraining
responsibility:
  - Provide retraining behavior for `run_retraining_candidate.py`.
inputs: []
outputs: []
tags:
  - retraining
lifecycle:
  status: active
"""

from tools.retraining import run_retraining_candidate as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
