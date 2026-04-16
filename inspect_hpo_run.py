"""Compatibility wrapper for the moved HPO inspection helper."""

from tools.hpo import inspect_hpo_run as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
