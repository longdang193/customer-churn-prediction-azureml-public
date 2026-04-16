"""Compatibility wrapper for the moved release failure probe helper."""

from tools.release import run_release_failure_probe as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
