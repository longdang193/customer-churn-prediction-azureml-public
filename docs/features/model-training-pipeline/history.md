# History

## 2026-04-10

- Seeded the initial feature contract from `run_pipeline.py`, `src/train.py`, and the AML train component.
- Recorded the target architecture requirement that fixed training should feed a promotion gate rather than deploy directly.
- Recorded the implementation shift to environment-backed compute and asset selection plus a concrete promotion utility in `src/promotion/promote_model.py`.
- Recorded the config ownership patch that makes `configs/train.yaml` the canonical owner of training experiment naming while keeping `configs/mlflow.yaml` only as a fallback.
- Recorded the first real release handoff pass: the AML train component now emits both candidate metrics and a deployable MLflow bundle, the fixed pipeline materializes baseline metrics from the latest approved Azure ML model metadata, and `run_release.py` gates registration/deployment on `promotion_decision`.
- Consolidated runtime config ownership into `src/config/runtime.py`, moved model-factory behavior under `src/models/`, and removed the old `utils` barrel so the fixed-training path is easier to trace.
- Moved the Azure ML SDK boundary behind `src/azureml/` adapters so `run_pipeline.py`, `run_hpo.py`, and `run_release.py` share client/input/registry wiring instead of duplicating AML SDK setup.
- Fixed `src/train.py` so training-level `--set` overrides such as `use_smote=false`, `class_weight=...`, and `random_state=...` now reach the actual runtime call instead of being shadowed by truthiness-based fallback logic.

## 2026-04-11

- Added `configs/train_smoke.yaml` and `run_pipeline.py --train-config` support so fixed-training smoke runs can use smoke-scoped training and promotion policy without weakening production defaults.
- Added `run_pipeline.py --data-config` and lightweight lineage tagging so submitted fixed-training jobs record data assets, selected configs, component identities, environment, and git commit.
- Added structured `step_manifest.json` output for the training step and surfaced the validation, prep, and training manifests together in local smoke preflight for easier debugging.
- Extended the structured manifest surface to `promote_model` and added warning coverage for tiny smoke test splits and suspiciously perfect training metrics.
- Promoted validation, data-prep, training, and promotion manifests to declared AML manifest output folders containing `step_manifest.json` while preserving nested download fallbacks.
- Fixed the eval smoke asset resolver so `data_smoke_eval.yaml` uses `churn-data-smoke-eval` as current data and `churn-data-smoke` as the validation reference.

## 2026-04-12

- Recorded that the fixed training pipeline now shares the same validation and data-prep component contracts with the new HPO pipeline entrypoint so optimization and retraining stay aligned on gate semantics and preprocessing behavior.
- Recorded that declared AML manifest folders are now the canonical operator surface, with nested manifest files preserved only as compatibility/download fallbacks.
- Added release-lineage validation framing: fixed-train and validation manifests are now treated as evidence that `run_release.py` checks before registry/deployment mutation.
- Recorded the new monitor-triggered retraining policy contract: monitor may now open a retraining candidate or investigation state, but fixed training and HPO still require a concrete dataset freeze plus `validate_data` before execution.
- Recorded the next bridge after monitor policy: `run_retraining_candidate.py` now freezes explicit dataset identities and prepares a `validate_data` handoff, but still stops short of auto-submitting fixed training or HPO.
- Added the first validated-candidate reconnection into fixed training: `run_retraining_fixed_train_smoke.py` now consumes a passed candidate validation, writes an exact smoke fixed-train invocation contract, and can submit `run_pipeline.py` with explicit current/reference dataset overrides without mutating production asset defaults.
- Added `run_retraining_path_selection.py` as the explicit post-validation selector: it now preserves release/monitor/candidate/validation lineage, invokes the already-proven fixed-train smoke bridge when `fixed_train` is selected, and keeps `model_sweep` truthful as a prepared HPO handoff artifact rather than overclaiming a fully bridged HPO retraining path.
- Added `run_retraining_hpo_smoke.py` plus explicit current/reference override support in `run_hpo_pipeline.py`, so the selected `model_sweep` branch can now reconnect a passed retraining candidate into the smoke HPO pipeline with the same lineage discipline already proven on the fixed-train side.
- Replaced the implicit HPO family tie-break `(primary_metric, roc_auc, model_name)` with an explicit selection policy driven by `selection.secondary_metric`, `selection.family_priority`, and a stable final fallback, and surfaced the winning reason in HPO summary and winner-manifest artifacts.
- Added `run_retraining_loop.py` as the thin phase-1 orchestrator that composes candidate freeze, validation, path selection, and at most one selected downstream smoke bridge while keeping fixed-train and HPO submission semantics owned by their existing scripts.
- Added `run_retraining_hpo_to_fixed_train.py` as the next bounded continuation after `model_sweep`: it validates the HPO winner across `hpo_summary`, `winner_manifest`, and `winner_train_config`, reuses or exports one effective train config, and invokes the fixed-train smoke bridge so the HPO path reconnects to promotion-facing evidence without releasing directly.
- Completed the first bounded end-to-end loop proof through the fixed-train branch: `run_retraining_loop.py` selected `fixed_train`, submitted AML job `modest_sponge_hxk5b23vty`, and the resulting train outputs still matched the smoke expectations (`validation_summary = passed`, `best_model = logreg`, `candidate f1 = 0.75`, `promotion_decision = promote`) before release continuation started.
- Added negative-proof coverage for the HPO continuation seam and confirmed the existing winner-consistency gate already blocks truthfully when `hpo_summary` disagrees with `winner_manifest` or required winner artifacts are missing; blocked continuation still records no downstream fixed-train submission.
- Hardened `run_retraining_hpo_to_fixed_train.py` so remote HPO continuation now waits for the parent HPO AML job to complete before downloading winner artifacts, avoiding the earlier race where the loop tried to continue while the HPO parent was still running.
- Reached the full `model_sweep` child-surface lifecycle depth with promotable continuation child `heroic_boniato_jk85w0dvkg`: HPO smoke completed, HPO continuation submitted the fixed-train rerun, downstream promotion was `promote`, release deployed to `churn-endpoint/churn-deployment`, and post-release monitor handoff finished `capture_backed`.
- Closed the remaining top-level `model_sweep` completeness gap with a bounded resume seam in `run_retraining_loop.py`: the loop can now resume from already-promotable continuation evidence, record resumed provenance explicitly, and deterministically reach `final_stage = monitor_handoff` without rerunning the stochastic HPO branch.
