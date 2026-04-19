"""Compatibility wrapper for the moved HPO inspection helper.

@meta
name: inspect_hpo_run
type: script
domain: hpo
responsibility:
  - Provide hpo behavior for `inspect_hpo_run.py`.
inputs: []
outputs: []
tags:
  - hpo
lifecycle:
  status: active
"""

from tools.hpo import inspect_hpo_run as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
