"""Compatibility wrapper for the moved release failure probe helper.

@meta
name: run_release_failure_probe
type: script
domain: release
responsibility:
  - Provide release behavior for `run_release_failure_probe.py`.
inputs: []
outputs: []
tags:
  - release
lifecycle:
  status: active
"""

from tools.release import run_release_failure_probe as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
