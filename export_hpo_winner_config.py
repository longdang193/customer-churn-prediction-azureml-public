"""Compatibility wrapper for the moved HPO winner export helper.

@meta
name: export_hpo_winner_config
type: script
domain: hpo
responsibility:
  - Provide hpo behavior for `export_hpo_winner_config.py`.
inputs: []
outputs: []
tags:
  - hpo
lifecycle:
  status: active
"""

from tools.hpo import export_hpo_winner_config as _impl

if __name__ == "__main__":
    _impl.main()
else:
    import sys as _sys

    _sys.modules[__name__] = _impl
