"""
@meta
type: test
scope: unit
domain: data-validation
covers:
  - Evidently-backed validation artifact generation
  - Validation summary gating for batch drift checks
excludes:
  - Real Azure ML job submission
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import pandas as pd


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


class FakeReport:
    def __init__(self) -> None:
        self.ran = False

    def run(self, *, current_data: object, reference_data: object) -> "FakeReport":
        self.ran = True
        return self

    def save_html(self, path: str) -> None:
        Path(path).write_text("<html>fake-evidently</html>", encoding="utf-8")

    def save_json(self, path: str) -> None:
        Path(path).write_text('{"kind":"fake-evidently"}', encoding="utf-8")


class FakeDriftReport(FakeReport):
    def save_json(self, path: str) -> None:
        payload = {
            "metrics": [
                {
                    "metric_name": "DriftedColumnsCount(drift_share=0.3)",
                    "config": {"type": "evidently:metric_v2:DriftedColumnsCount"},
                    "value": {"count": 2.0, "share": 0.5},
                }
            ]
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")


def _write_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"test-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_run_validation_creates_summary_and_evidently_artifacts() -> None:
    """
    @proves data-prep.support-pre-training-validation-drift-analysis-against-baseline
    @proves data-prep.emit-structured-step-manifest-json-artifact-validation-data
    """
    from src.validate_data import ValidationConfig, run_validation

    temp_dir = _make_temp_dir()
    try:
        reference_path = temp_dir / "reference.csv"
        current_path = temp_dir / "current.csv"
        output_dir = temp_dir / "validation_artifacts"
        summary_path = temp_dir / "validation_summary.json"
        manifest_output_path = temp_dir / "validation_manifest.json"

        _write_dataset(
            reference_path,
            [
                {"CreditScore": 600, "Balance": 0.0, "Exited": 0},
                {"CreditScore": 720, "Balance": 1200.0, "Exited": 1},
            ],
        )
        _write_dataset(
            current_path,
            [
                {"CreditScore": 610, "Balance": 20.0, "Exited": 0},
                {"CreditScore": 700, "Balance": 1100.0, "Exited": 1},
            ],
        )

        summary = run_validation(
            reference_path=reference_path,
            current_path=current_path,
            output_dir=output_dir,
            summary_path=summary_path,
            config=ValidationConfig(
                target_column="Exited",
                max_missing_fraction=0.2,
                max_row_count_delta_fraction=0.5,
            ),
            manifest_output_path=manifest_output_path,
            report_factory=FakeReport,
        )

        assert summary["status"] == "passed"
        assert summary["checks"]["schema_match"] is True
        assert summary["checks"]["target_column_present"] is True
        assert summary["checks"]["missing_fraction_within_threshold"] is True
        assert (output_dir / "evidently_report.html").exists()
        assert (output_dir / "evidently_report.json").exists()
        assert (output_dir / "step_manifest.json").exists()
        assert manifest_output_path.exists()
        assert summary_path.exists()

        saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert saved_summary["row_counts"] == {"reference": 2, "current": 2}
        saved_manifest = json.loads((output_dir / "step_manifest.json").read_text(encoding="utf-8"))
        assert saved_manifest["status"] == "passed"
        assert saved_manifest["step_name"] == "validate_data"
        assert saved_manifest["metrics"]["reference_row_count"] == 2
        assert saved_manifest["metrics"]["current_row_count"] == 2
        assert saved_manifest == json.loads(manifest_output_path.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_run_validation_fails_when_missing_fraction_crosses_threshold() -> None:
    from src.validate_data import ValidationConfig, run_validation

    temp_dir = _make_temp_dir()
    try:
        reference_path = temp_dir / "reference.csv"
        current_path = temp_dir / "current.csv"

        _write_dataset(
            reference_path,
            [
                {"CreditScore": 600, "Balance": 0.0, "Exited": 0},
                {"CreditScore": 720, "Balance": 1200.0, "Exited": 1},
            ],
        )
        _write_dataset(
            current_path,
            [
                {"CreditScore": None, "Balance": 20.0, "Exited": 0},
                {"CreditScore": 700, "Balance": 1100.0, "Exited": 1},
            ],
        )

        summary = run_validation(
            reference_path=reference_path,
            current_path=current_path,
            output_dir=temp_dir / "validation_artifacts",
            summary_path=temp_dir / "validation_summary.json",
            config=ValidationConfig(
                target_column="Exited",
                max_missing_fraction=0.1,
                fail_on_missing_fraction=True,
            ),
            report_factory=FakeReport,
        )

        assert summary["status"] == "failed"
        assert summary["checks"]["missing_fraction_within_threshold"] is False
        assert "max_missing_fraction" in summary["failed_checks"]
        saved_manifest = json.loads(
            (temp_dir / "validation_artifacts" / "step_manifest.json").read_text(encoding="utf-8")
        )
        assert saved_manifest["status"] == "failed"
        assert saved_manifest["step_specific"]["summary_status"] == "failed"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_run_validation_short_circuits_before_evidently_when_guardrails_fail() -> None:
    from src.validate_data import ValidationConfig, run_validation

    temp_dir = _make_temp_dir()
    try:
        reference_path = temp_dir / "reference.csv"
        current_path = temp_dir / "current.csv"
        output_dir = temp_dir / "validation_artifacts"
        summary_path = temp_dir / "validation_summary.json"

        _write_dataset(
            reference_path,
            [
                {"CreditScore": 600, "Balance": 0.0, "Exited": 0},
                {"CreditScore": 720, "Balance": 1200.0, "Exited": 1},
            ],
        )
        _write_dataset(
            current_path,
            [
                {"CreditScore": None, "Balance": 20.0, "Exited": 0},
                {"CreditScore": None, "Balance": 1100.0, "Exited": 1},
            ],
        )

        class ExplodingReport(FakeReport):
            def run(self, *, current_data: object, reference_data: object) -> "FakeReport":
                raise AssertionError("Evidently should not run after pre-report guardrail failure")

        summary = run_validation(
            reference_path=reference_path,
            current_path=current_path,
            output_dir=output_dir,
            summary_path=summary_path,
            config=ValidationConfig(
                target_column="Exited",
                max_missing_fraction=0.2,
                fail_on_missing_fraction=True,
            ),
            report_factory=ExplodingReport,
        )

        assert summary["status"] == "failed"
        assert summary["drift"]["drifted_column_share"] is None
        assert summary["checks"]["drift_within_threshold"] is None
        assert summary["checks"]["missing_fraction_within_threshold"] is False
        assert summary["failed_checks"] == ["max_missing_fraction"]
        assert summary_path.exists()
        assert (output_dir / "evidently_report.html").exists()
        skipped_report = json.loads((output_dir / "evidently_report.json").read_text(encoding="utf-8"))
        assert skipped_report == {
            "status": "skipped",
            "reason": "pre_report_guardrail_failed",
            "failed_checks": ["max_missing_fraction"],
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_run_validation_fails_when_drift_crosses_threshold() -> None:
    """
    @proves data-prep.support-pre-training-validation-drift-analysis-against-baseline
    """
    from src.validate_data import ValidationConfig, run_validation

    temp_dir = _make_temp_dir()
    try:
        reference_path = temp_dir / "reference.csv"
        current_path = temp_dir / "current.csv"

        _write_dataset(
            reference_path,
            [
                {"CreditScore": 600, "Balance": 0.0, "Exited": 0},
                {"CreditScore": 720, "Balance": 1200.0, "Exited": 1},
            ],
        )
        _write_dataset(
            current_path,
            [
                {"CreditScore": 610, "Balance": 20.0, "Exited": 0},
                {"CreditScore": 700, "Balance": 1100.0, "Exited": 1},
            ],
        )

        summary = run_validation(
            reference_path=reference_path,
            current_path=current_path,
            output_dir=temp_dir / "validation_artifacts",
            summary_path=temp_dir / "validation_summary.json",
            config=ValidationConfig(
                target_column="Exited",
                drift_share_threshold=0.3,
                fail_on_drift=True,
            ),
            report_factory=FakeDriftReport,
        )

        assert summary["status"] == "failed"
        assert summary["checks"]["drift_within_threshold"] is False
        assert summary["drift"]["drifted_column_count"] == 2.0
        assert summary["drift"]["drifted_column_share"] == 0.5
        assert "data_drift_detected" in summary["failed_checks"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_validation_config_parses_quoted_boolean_values() -> None:
    """
    @proves data-prep.accept-selected-data-config-files-through-aml-validation
    """
    from src.validate_data import load_validation_config

    temp_dir = _make_temp_dir()
    try:
        config_path = temp_dir / "data.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "data:",
                    '  target_column: "Exited"',
                    "validation:",
                    '  fail_on_drift: "true"',
                    '  fail_on_schema_mismatch: "false"',
                    '  fail_on_missing_fraction: "true"',
                    '  fail_on_row_count_delta: "false"',
                ]
            ),
            encoding="utf-8",
        )

        config = load_validation_config(str(config_path))

        assert config.fail_on_schema_mismatch is False
        assert config.fail_on_missing_fraction is True
        assert config.fail_on_row_count_delta is False
        assert config.fail_on_drift is True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_create_report_uses_plain_data_definition_for_raw_validation(monkeypatch) -> None:
    import src.validate_data as validate_data

    captured: dict[str, object] = {}

    class FakeDataDefinition:
        def __init__(self, **kwargs: object) -> None:
            captured["definition_kwargs"] = kwargs

    class FakeDataset:
        @staticmethod
        def from_pandas(dataframe: object, *, data_definition: object) -> object:
            captured.setdefault("datasets", []).append(
                {
                    "dataframe": dataframe,
                    "data_definition": data_definition,
                }
            )
            return dataframe

    class FakePreset:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    class FakeReportWithRun(FakeReport):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            captured["report_args"] = args
            captured["report_kwargs"] = kwargs

    monkeypatch.setattr(
        validate_data,
        "_build_default_report",
        lambda: (
            FakeReportWithRun,
            FakeDataset,
            FakeDataDefinition,
            FakePreset,
            FakePreset,
        ),
    )

    reference = pd.DataFrame([{"feature": 1.0, "Exited": 0}])
    current = pd.DataFrame([{"feature": 2.0, "Exited": 1}])

    report = validate_data._create_report(
        reference,
        current,
        validate_data.ValidationConfig(target_column="Exited"),
        report_factory=None,
    )

    assert isinstance(report, FakeReportWithRun)
    assert captured["definition_kwargs"] == {}
    assert len(captured["datasets"]) == 2


def test_run_validation_persists_snapshot_exports_when_report_run_returns_snapshot() -> None:
    from src.validate_data import ValidationConfig, run_validation

    temp_dir = _make_temp_dir()
    try:
        reference_path = temp_dir / "reference.csv"
        current_path = temp_dir / "current.csv"
        output_dir = temp_dir / "validation_artifacts"
        summary_path = temp_dir / "validation_summary.json"

        _write_dataset(
            reference_path,
            [{"CreditScore": 600, "Balance": 0.0, "Exited": 0}],
        )
        _write_dataset(
            current_path,
            [{"CreditScore": 610, "Balance": 20.0, "Exited": 0}],
        )

        saved_paths: dict[str, str] = {}

        class FakeSnapshot:
            def save_html(self, path: str) -> None:
                saved_paths["html"] = path
                Path(path).write_text("<html>snapshot</html>", encoding="utf-8")

            def save_json(self, path: str) -> None:
                saved_paths["json"] = path
                Path(path).write_text('{"snapshot": true}', encoding="utf-8")

        class FakeReportReturningSnapshot:
            def run(self, *, current_data: object, reference_data: object) -> FakeSnapshot:
                return FakeSnapshot()

        summary = run_validation(
            reference_path=reference_path,
            current_path=current_path,
            output_dir=output_dir,
            summary_path=summary_path,
            config=ValidationConfig(target_column="Exited"),
            report_factory=FakeReportReturningSnapshot,
        )

        assert summary["status"] == "passed"
        assert saved_paths["html"].endswith("evidently_report.html")
        assert saved_paths["json"].endswith("evidently_report.json")
        assert (output_dir / "evidently_report.html").exists()
        assert (output_dir / "evidently_report.json").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
