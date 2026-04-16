"""
@meta
type: test
scope: unit
domain: release
covers:
  - Approved-model baseline selection from registry metadata
  - Promotion-gated release handoff behavior
excludes:
  - Real Azure ML registry or endpoint calls
tags:
  - fast
  - ci-safe
"""

from __future__ import annotations

import pytest


def test_select_latest_approved_model_baseline_prefers_highest_approved_version() -> None:
    from src.release.workflow import select_latest_approved_model_baseline

    models = [
        {
            "name": "churn-model",
            "version": "3",
            "tags": {"approval_status": "candidate", "f1": "0.77", "roc_auc": "0.82"},
        },
        {
            "name": "churn-model",
            "version": "4",
            "tags": {
                "approval_status": "approved",
                "primary_metric": "f1",
                "f1": "0.79",
                "roc_auc": "0.84",
            },
        },
        {
            "name": "churn-model",
            "version": "2",
            "tags": {
                "approval_status": "approved",
                "primary_metric": "f1",
                "f1": "0.75",
                "roc_auc": "0.81",
            },
        },
    ]

    baseline = select_latest_approved_model_baseline(models)

    assert baseline is not None
    assert baseline.version == "4"
    assert baseline.primary_metric == "f1"
    assert baseline.metrics["f1"] == 0.79
    assert baseline.metrics["roc_auc"] == 0.84


def test_ensure_promotable_decision_rejects_non_promoted_candidates() -> None:
    from src.release.workflow import ensure_promotable_decision

    with pytest.raises(ValueError, match="Promotion decision is reject"):
        ensure_promotable_decision({"status": "reject", "reasons": ["improvement_below_threshold"]})


def test_build_job_output_uri_targets_azureml_job_outputs() -> None:
    from src.release.workflow import build_job_output_uri

    assert (
        build_job_output_uri("job-123", "mlflow_model")
        == "azureml://jobs/job-123/outputs/mlflow_model/paths/"
    )


def test_build_registered_model_tags_carries_approval_and_metric_metadata() -> None:
    from src.release.workflow import build_registered_model_tags

    tags = build_registered_model_tags(
        candidate_metrics={"model_name": "xgboost", "f1": 0.81, "roc_auc": 0.88},
        promotion_decision={"status": "promote", "primary_metric": "f1"},
        job_name="job-123",
    )

    assert tags["approval_status"] == "approved"
    assert tags["source_job_name"] == "job-123"
    assert tags["primary_metric"] == "f1"
    assert tags["f1"] == "0.81"
    assert tags["roc_auc"] == "0.88"


def test_build_registered_model_tags_merges_lineage_without_overriding_core_tags() -> None:
    from src.release.workflow import build_registered_model_tags

    tags = build_registered_model_tags(
        candidate_metrics={"model_name": "xgboost", "f1": 0.81, "roc_auc": 0.88},
        promotion_decision={"status": "promote", "primary_metric": "f1"},
        job_name="job-123",
        lineage_tags={
            "data_asset": "churn-data",
            "data_config": "configs/data.yaml",
            "train_config": "configs/train.yaml",
            "source_job_name": "should-not-win",
        },
    )

    assert tags["data_asset"] == "churn-data"
    assert tags["data_config"] == "configs/data.yaml"
    assert tags["train_config"] == "configs/train.yaml"
    assert tags["source_job_name"] == "job-123"


def test_build_release_lineage_rejects_train_config_mismatch() -> None:
    from src.release.workflow import build_release_lineage

    lineage = build_release_lineage(
        declared_data_config="configs/data_smoke_eval.yaml",
        declared_train_config="configs/train_smoke_hpo_winner_rf.yaml",
        train_manifest={
            "config": {
                "config_paths": {
                    "train_config": "/mnt/azureml/wd/INPUT_config/train_smoke.yaml",
                },
            },
            "tags": {"best_model": "logreg"},
        },
    )

    assert lineage["validation"]["status"] == "failed"
    assert "train_config mismatch" in lineage["validation"]["errors"][0]
    assert lineage["effective_lineage"]["train_config"] == "configs/train_smoke_hpo_winner_rf.yaml"
    assert lineage["source_job_lineage"]["train_config"] == "train_smoke.yaml"


def test_build_release_lineage_prefers_canonical_train_config_from_manifest() -> None:
    from src.release.workflow import build_release_lineage

    lineage = build_release_lineage(
        declared_data_config="configs/data_smoke_eval.yaml",
        declared_train_config="configs/train_smoke_hpo_winner_rf.yaml",
        train_manifest={
            "config": {
                "config_paths": {
                    "train_config": "/mnt/azureml/wd/INPUT_config/train_config.yaml",
                    "canonical_train_config": "configs/train_smoke_hpo_winner_rf.yaml",
                },
            },
            "tags": {"best_model": "rf"},
        },
    )

    assert lineage["validation"]["status"] == "validated"
    assert lineage["validation"]["errors"] == []
    assert lineage["source_job_lineage"]["train_config"] == "configs/train_smoke_hpo_winner_rf.yaml"


def test_build_release_lineage_override_allows_mismatch_with_marker() -> None:
    from src.release.workflow import build_release_lineage

    lineage = build_release_lineage(
        declared_data_config="configs/data_smoke_eval.yaml",
        declared_train_config="configs/train_smoke_hpo_winner_rf.yaml",
        train_manifest={
            "config": {
                "config_paths": {
                    "train_config": "/mnt/azureml/wd/INPUT_config/train_smoke.yaml",
                },
            },
        },
        allow_mismatch=True,
    )

    assert lineage["validation"]["status"] == "override"
    assert lineage["effective_lineage"]["lineage_status"] == "mismatch_allowed"
    assert lineage["validation"]["errors"] == []
    assert lineage["validation"]["warnings"]


def test_build_registered_model_tags_uses_effective_lineage() -> None:
    from src.release.workflow import build_registered_model_tags

    tags = build_registered_model_tags(
        candidate_metrics={"model_name": "logreg", "f1": 0.75, "roc_auc": 0.8125},
        promotion_decision={"status": "promote", "primary_metric": "f1"},
        job_name="job-123",
        lineage_tags={
            "data_config": "data_smoke_eval.yaml",
            "train_config": "train_smoke.yaml",
            "lineage_status": "validated",
            "best_model": "logreg",
        },
    )

    assert tags["train_config"] == "train_smoke.yaml"
    assert tags["lineage_status"] == "validated"
    assert tags["best_model"] == "logreg"


def test_select_reusable_registered_model_prefers_highest_matching_version() -> None:
    from src.release.workflow import select_reusable_registered_model

    models = [
        {
            "name": "churn-model",
            "version": "9",
            "tags": {
                "approval_status": "approved",
                "source_job_name": "job-123",
                "lineage_status": "validated",
                "data_config": "configs/data_smoke_eval.yaml",
                "train_config": "configs/train_smoke.yaml",
                "candidate_model_name": "rf",
            },
        },
        {
            "name": "churn-model",
            "version": "10",
            "tags": {
                "approval_status": "approved",
                "source_job_name": "job-123",
                "lineage_status": "validated",
                "data_config": "configs/data_smoke_eval.yaml",
                "train_config": "configs/train_smoke.yaml",
                "candidate_model_name": "rf",
            },
        },
        {
            "name": "churn-model",
            "version": "11",
            "tags": {
                "approval_status": "approved",
                "source_job_name": "job-999",
                "lineage_status": "validated",
                "data_config": "configs/data_smoke_eval.yaml",
                "train_config": "configs/train_smoke.yaml",
                "candidate_model_name": "rf",
            },
        },
    ]

    selected = select_reusable_registered_model(
        models,
        source_job_name="job-123",
        effective_lineage={
            "lineage_status": "validated",
            "data_config": "configs/data_smoke_eval.yaml",
            "train_config": "configs/train_smoke.yaml",
        },
        candidate_metrics={"model_name": "rf"},
    )

    assert selected is not None
    assert selected["version"] == "10"


def test_select_reusable_registered_model_ignores_lineage_drift() -> None:
    from src.release.workflow import select_reusable_registered_model

    selected = select_reusable_registered_model(
        [
            {
                "name": "churn-model",
                "version": "10",
                "tags": {
                    "approval_status": "approved",
                    "source_job_name": "job-123",
                    "lineage_status": "validated",
                    "data_config": "configs/data.yaml",
                    "train_config": "configs/train_smoke.yaml",
                    "candidate_model_name": "rf",
                },
            },
        ],
        source_job_name="job-123",
        effective_lineage={
            "lineage_status": "validated",
            "data_config": "configs/data_smoke_eval.yaml",
            "train_config": "configs/train_smoke.yaml",
        },
        candidate_metrics={"model_name": "rf"},
    )

    assert selected is None
