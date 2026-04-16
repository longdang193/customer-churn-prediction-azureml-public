"""Compatibility wrapper for the moved retraining candidate helper."""

from tools.retraining import run_retraining_candidate as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
