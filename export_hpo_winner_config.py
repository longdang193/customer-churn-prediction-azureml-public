"""Compatibility wrapper for the moved HPO winner export helper."""

from tools.hpo import export_hpo_winner_config as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
