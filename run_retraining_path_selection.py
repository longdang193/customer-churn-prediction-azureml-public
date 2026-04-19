"""Compatibility wrapper for the moved retraining path selector helper.

@meta
name: run_retraining_path_selection
type: script
domain: retraining
responsibility:
  - Provide retraining behavior for `run_retraining_path_selection.py`.
inputs: []
outputs: []
tags:
  - retraining
lifecycle:
  status: active
"""

from tools.retraining import run_retraining_path_selection as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
