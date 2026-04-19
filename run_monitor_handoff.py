"""Compatibility wrapper for the moved monitor handoff helper.

@meta
name: run_monitor_handoff
type: script
domain: monitoring
responsibility:
  - Provide monitoring behavior for `run_monitor_handoff.py`.
inputs: []
outputs: []
tags:
  - monitoring
lifecycle:
  status: active
"""

from tools.monitoring import run_monitor_handoff as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
