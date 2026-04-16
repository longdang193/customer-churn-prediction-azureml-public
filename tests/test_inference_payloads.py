"""
@meta
type: test
scope: unit
domain: inference
covers:
  - Endpoint smoke payload validation for Azure ML online endpoint invokes
excludes:
  - Real Azure ML endpoint calls
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import pytest


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


def _make_temp_dir() -> Path:
    temp_dir = TEST_TEMP_ROOT / f"inference-payloads-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def _write_payload(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_endpoint_payload_file_accepts_input_data_2d_array() -> None:
    from src.inference.payloads import validate_endpoint_payload_file

    temp_dir = _make_temp_dir()
    try:
        payload_path = temp_dir / "sample-data.json"
        _write_payload(payload_path, {"input_data": [[0.0, 1, 2.5]]})

        summary = validate_endpoint_payload_file(
            payload_path,
            expected_feature_count=3,
        )

        assert summary.payload_path == payload_path
        assert summary.row_count == 1
        assert summary.feature_count == 3
        assert summary.payload_format == "input_data_2d_array"
        assert summary.to_record() == {
            "path": str(payload_path),
            "format": "input_data_2d_array",
            "row_count": 1,
            "feature_count": 3,
            "validation_status": "passed",
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_endpoint_payload_file_rejects_missing_file() -> None:
    from src.inference.payloads import validate_endpoint_payload_file

    with pytest.raises(FileNotFoundError, match="Endpoint payload file not found"):
        validate_endpoint_payload_file(Path("missing-payload.json"))


def test_validate_endpoint_payload_file_rejects_invalid_json() -> None:
    from src.inference.payloads import validate_endpoint_payload_file

    temp_dir = _make_temp_dir()
    try:
        payload_path = temp_dir / "sample-data.json"
        payload_path.write_text("{not-json", encoding="utf-8")

        with pytest.raises(ValueError, match="valid JSON"):
            validate_endpoint_payload_file(payload_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_endpoint_payload_file_rejects_missing_input_data() -> None:
    from src.inference.payloads import validate_endpoint_payload_file

    temp_dir = _make_temp_dir()
    try:
        payload_path = temp_dir / "sample-data.json"
        _write_payload(payload_path, {"data": []})

        with pytest.raises(ValueError, match="input_data"):
            validate_endpoint_payload_file(payload_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_endpoint_payload_file_rejects_non_list_input_data() -> None:
    from src.inference.payloads import validate_endpoint_payload_file

    temp_dir = _make_temp_dir()
    try:
        payload_path = temp_dir / "sample-data.json"
        _write_payload(payload_path, {"input_data": {"row": []}})

        with pytest.raises(ValueError, match="input_data"):
            validate_endpoint_payload_file(payload_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_endpoint_payload_file_rejects_non_list_row() -> None:
    from src.inference.payloads import validate_endpoint_payload_file

    temp_dir = _make_temp_dir()
    try:
        payload_path = temp_dir / "sample-data.json"
        _write_payload(payload_path, {"input_data": [{"not": "a row"}]})

        with pytest.raises(ValueError, match="row 0"):
            validate_endpoint_payload_file(payload_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_endpoint_payload_file_rejects_wrong_feature_count() -> None:
    from src.inference.payloads import validate_endpoint_payload_file

    temp_dir = _make_temp_dir()
    try:
        payload_path = temp_dir / "sample-data.json"
        _write_payload(payload_path, {"input_data": [[1.0, 2.0]]})

        with pytest.raises(ValueError, match="expected 3 features"):
            validate_endpoint_payload_file(payload_path, expected_feature_count=3)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_endpoint_payload_file_rejects_non_numeric_value() -> None:
    from src.inference.payloads import validate_endpoint_payload_file

    temp_dir = _make_temp_dir()
    try:
        payload_path = temp_dir / "sample-data.json"
        _write_payload(payload_path, {"input_data": [[1.0, "bad", 3.0]]})

        with pytest.raises(ValueError, match="row 0 column 1"):
            validate_endpoint_payload_file(payload_path, expected_feature_count=3)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_endpoint_payload_file_rejects_boolean_value() -> None:
    from src.inference.payloads import validate_endpoint_payload_file

    temp_dir = _make_temp_dir()
    try:
        payload_path = temp_dir / "sample-data.json"
        _write_payload(payload_path, {"input_data": [[1.0, True, 3.0]]})

        with pytest.raises(ValueError, match="row 0 column 1"):
            validate_endpoint_payload_file(payload_path, expected_feature_count=3)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

