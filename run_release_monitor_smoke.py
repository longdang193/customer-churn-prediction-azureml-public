"""Compatibility wrapper for the moved release-monitor smoke helper."""

from tools.monitoring import run_release_monitor_smoke as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
