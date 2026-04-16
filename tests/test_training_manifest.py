from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import pandas as pd


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_save_mlflow_model_bundle_normalizes_conda_python_spec(monkeypatch) -> None:
    import src.training.training as training_module

    temp_dir = _make_temp_dir()
    try:
        model_dir = temp_dir / "mlflow_model"

        def fake_save_model(*, sk_model: object, path: str) -> None:
            del sk_model
            output_dir = Path(path)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "conda.yaml").write_text(
                "\n".join(
                    [
                        "name: mlflow-env",
                        "dependencies:",
                        "- python=3.9.25",
                        "- pip<=23.0.1",
                        "- pip:",
                        "  - mlflow==3.1.4",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

        monkeypatch.setattr(training_module.mlflow.sklearn, "save_model", fake_save_model)

        training_module.save_mlflow_model_bundle(object(), model_dir)

        conda_yaml = (model_dir / "conda.yaml").read_text(encoding="utf-8")
        assert "- python=3.9\n" in conda_yaml
        assert "python=3.9.25" not in conda_yaml
        assert "  - azureml-ai-monitoring==1.0.0\n" in conda_yaml
        assert "  - azureml-inference-server-http\n" in conda_yaml
        assert "  - azure-storage-blob==12.19.0\n" in conda_yaml
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_train_pipeline_stage_writes_step_manifest(monkeypatch) -> None:
    import src.training.training as training_module

    temp_dir = _make_temp_dir()
    try:
        data_dir = temp_dir / "processed"
        model_output_dir = temp_dir / "model_output"
        manifest_output_path = temp_dir / "train_manifest.json"
        data_dir.mkdir()
        model_output_dir.mkdir()

        pd.DataFrame({"feature": [1.0, 2.0]}).to_csv(data_dir / "X_train.csv", index=False)
        pd.DataFrame({"feature": [3.0]}).to_csv(data_dir / "X_test.csv", index=False)
        pd.DataFrame({"Exited": [0, 1]}).to_csv(data_dir / "y_train.csv", index=False)
        pd.DataFrame({"Exited": [1]}).to_csv(data_dir / "y_test.csv", index=False)

        monkeypatch.setattr(training_module, "is_azure_ml", lambda: False)

        class FakeRunInfo:
            run_id = "parent-run-id"

        class FakeRun:
            info = FakeRunInfo()

        monkeypatch.setattr(training_module, "start_parent_run", lambda _name: FakeRun())
        monkeypatch.setattr(training_module, "get_active_run", lambda: FakeRun())
        monkeypatch.setattr(training_module.mlflow, "log_params", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(training_module.mlflow, "log_metric", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(training_module.mlflow, "set_tag", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(training_module.mlflow, "end_run", lambda: None)
        monkeypatch.setattr(
            training_module,
            "train_model",
            lambda *_args, **_kwargs: {
                "test_metrics": {"f1": 0.81, "roc_auc": 0.79},
                "run_id": "child-run-id",
                "artifact_path": "model_logreg",
                "model": object(),
            },
        )
        monkeypatch.setattr(
            training_module,
            "save_mlflow_model_bundle",
            lambda _model, output_dir: (
                Path(output_dir).mkdir(parents=True, exist_ok=True),
                (Path(output_dir) / "MLmodel").write_text("ok", encoding="utf-8"),
            ),
        )
        monkeypatch.setattr(
            training_module.joblib,
            "dump",
            lambda _model, path: Path(path).write_bytes(b"model"),
        )

        results = training_module.train_pipeline_stage(
            data_dir=str(data_dir),
            models=["logreg"],
            class_weight="balanced",
            random_state=42,
            experiment_name="train-smoke",
            use_smote=False,
            hyperparams_by_model={"logreg": {}},
            model_artifact_dir=str(model_output_dir),
            parent_run_id_output=str(temp_dir / "parent_run_id"),
            candidate_metrics_output=str(temp_dir / "candidate_metrics"),
            mlflow_model_output=str(temp_dir / "mlflow_model"),
            manifest_output_path=str(manifest_output_path),
            config_path="configs/train_smoke.yaml",
            execution_mode="smoke_preflight",
        )

        assert "logreg" in results
        manifest = json.loads((model_output_dir / "step_manifest.json").read_text(encoding="utf-8"))
        mirrored_manifest = json.loads(manifest_output_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "success"
        assert manifest["run_context"]["execution_mode"] == "smoke_preflight"
        assert manifest["tags"]["best_model"] == "logreg"
        assert manifest["metrics"]["best_model_f1"] == 0.81
        assert any("Very small test split" in warning for warning in manifest["warnings"])
        assert manifest == mirrored_manifest
        assert (temp_dir / "parent_run_id" / "parent_run_id.txt").exists()
        assert (temp_dir / "candidate_metrics" / "candidate_metrics.json").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
