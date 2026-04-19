"""Compatibility wrapper for the moved release-monitor smoke helper.

@meta
name: run_release_monitor_smoke
type: script
domain: monitoring
responsibility:
  - Provide monitoring behavior for `run_release_monitor_smoke.py`.
inputs: []
outputs: []
tags:
  - monitoring
lifecycle:
  status: active
"""

from tools.monitoring import run_release_monitor_smoke as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
