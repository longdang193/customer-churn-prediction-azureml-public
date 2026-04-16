"""
@meta
type: test
scope: unit
domain: training-package
covers:
  - Public training package exports stay aligned with real modules
excludes:
  - Real model training
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

import sys
import types


def test_training_package_import_exports_train_pipeline_stage(monkeypatch) -> None:
    fake_azureml_mlflow = types.ModuleType("azureml.mlflow")
    fake_azureml = types.ModuleType("azureml")
    fake_azureml.mlflow = fake_azureml_mlflow
    fake_joblib = types.ModuleType("joblib")
    fake_joblib.dump = lambda *_args, **_kwargs: None
    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow_sklearn = types.ModuleType("mlflow.sklearn")
    fake_mlflow.sklearn = fake_mlflow_sklearn
    fake_data = types.ModuleType("data")
    fake_data.apply_smote = lambda *args, **kwargs: args[:2]
    fake_data.load_prepared_data = lambda *_args, **_kwargs: (None, None, None, None)
    fake_models_factory = types.ModuleType("models.factory")
    fake_models_factory.apply_class_weight_adjustments = lambda *_args, **_kwargs: {}
    fake_models_factory.apply_hyperparameters = lambda model, *_args, **_kwargs: (model, {})
    fake_models_factory.get_model = lambda *_args, **_kwargs: object()
    fake_utils_metrics = types.ModuleType("utils.metrics")
    fake_utils_metrics.calculate_metrics = (
        lambda *_args, **_kwargs: {"f1": 0.0, "roc_auc": 0.0}
    )
    fake_utils_mlflow_utils = types.ModuleType("utils.mlflow_utils")
    fake_utils_mlflow_utils.get_active_run = lambda: None
    fake_utils_mlflow_utils.get_run_id = lambda _run: "run-id"
    fake_utils_mlflow_utils.is_azure_ml = lambda: True
    fake_utils_mlflow_utils.start_nested_run = lambda _name: (None, "nested-run-id")
    fake_utils_mlflow_utils.start_parent_run = lambda _name: None

    monkeypatch.setitem(sys.modules, "azureml", fake_azureml)
    monkeypatch.setitem(sys.modules, "azureml.mlflow", fake_azureml_mlflow)
    monkeypatch.setitem(sys.modules, "joblib", fake_joblib)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.sklearn", fake_mlflow_sklearn)
    monkeypatch.setitem(sys.modules, "data", fake_data)
    monkeypatch.setitem(sys.modules, "models.factory", fake_models_factory)
    monkeypatch.setitem(sys.modules, "utils.metrics", fake_utils_metrics)
    monkeypatch.setitem(sys.modules, "utils.mlflow_utils", fake_utils_mlflow_utils)

    import training

    assert hasattr(training, "train_pipeline_stage")
