"""Compatibility wrapper for the moved HPO continuation helper.

@meta
name: run_retraining_hpo_to_fixed_train
type: script
domain: hpo
responsibility:
  - Provide hpo behavior for `run_retraining_hpo_to_fixed_train.py`.
inputs: []
outputs: []
tags:
  - hpo
lifecycle:
  status: active
"""

from tools.retraining import run_retraining_hpo_to_fixed_train as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
