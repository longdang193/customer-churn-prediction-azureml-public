"""Data processing modules for loading, preprocessing, and I/O.

@meta
name: data
type: module
domain: data-prep
responsibility:
  - Provide data-prep behavior for `src/data/__init__.py`.
inputs: []
outputs: []
tags:
  - data-prep
lifecycle:
  status: active
"""

from .data_io import load_data, save_artifacts, save_preprocessed_data
from .data_utils import apply_smote, load_prepared_data
from .preprocessing import encode_categoricals, remove_columns, scale_features

__all__ = [
    # Data I/O
    "load_data",
    "save_artifacts",
    "save_preprocessed_data",
    # Data utils
    "apply_smote",
    "load_prepared_data",
    # Preprocessing
    "encode_categoricals",
    "remove_columns",
    "scale_features",
]



