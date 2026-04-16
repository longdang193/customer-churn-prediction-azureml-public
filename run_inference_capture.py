"""Compatibility wrapper for the moved inference capture helper."""

from tools.monitoring import run_inference_capture as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
