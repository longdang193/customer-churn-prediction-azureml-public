"""Compatibility wrapper for the moved inference capture helper.

@meta
name: run_inference_capture
type: script
domain: project
responsibility:
  - Provide project behavior for `run_inference_capture.py`.
inputs: []
outputs: []
tags:
  - project
lifecycle:
  status: active
"""

from tools.monitoring import run_inference_capture as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
