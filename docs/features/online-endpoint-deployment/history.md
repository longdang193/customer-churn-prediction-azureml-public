# History

## 2026-04-10

- Seeded the initial feature contract from the deployment notebook and the tracked sample model bundle references.
- Recorded the target lifecycle change that deployment should follow promotion and feed monitoring.
- Tightened the release framing so deployment is documented as promoted-model release plus smoke-test and monitoring handoff, not just a notebook push of the latest artifact.
- Removed the legacy tracked `outputs/...` MLflow bundle reference from the feature contract so deployment docs no longer depend on runtime-artifact snapshots.
- Added the script-first release handoff in `run_release.py`, which now consumes `promotion_decision`, registers approved MLflow job outputs, and can deploy them to the managed online endpoint.
- Folded release-record construction into `src/release/workflow.py` so the deployment handoff no longer depends on a separate tiny helper module.
- Moved Azure ML registry and managed-endpoint SDK wiring behind `src/azureml/` adapters so deployment docs and contracts no longer imply that `src/release/` owns the raw Azure SDK boundary.

## 2026-04-11

- Restored ownership of `sample-data.json` as the endpoint smoke-test request payload and documented that it targets the processed-feature MLflow model scoring shape.
- Enriched registered model metadata with lightweight data/config/component lineage tags while preserving promotion and metric tags.

## 2026-04-12

- Ran the full fixed-train-to-release deploy smoke against Azure ML and confirmed the release path now registers, deploys, and invokes a managed online endpoint.
- Fixed release registration to use the Azure ML job-output folder-root URI form, `azureml://jobs/<job>/outputs/<output>/paths/`, so model registration does not look for a literal `paths` blob.
- Hardened exported MLflow model bundles for Azure ML generated deployment images by normalizing `conda.yaml` to a conda-resolvable Python minor version and including `azureml-ai-monitoring==1.0.0`, which Azure's generated MLflow scoring script imports.
- Updated `sample-data.json` to the `input_data` 2D-array shape verified by `az ml online-endpoint invoke` for the processed-feature MLflow model.
- Recorded the successful smoke evidence from `silver_neck_77x7rny6s4`, which registered `churn-prediction-model:6` and deployed to `churn-endpoint/churn-smoke-silver`.
- Added release-time lineage validation and enriched release records so registry tags are derived from source job manifests when available, and smoke payload/response evidence is preserved in `release_record.json`.

## 2026-04-13

- Made release registration idempotent by default: `run_release.py` now reuses an existing validated approved model version when source job, effective lineage, and candidate model match.
- Added `run_release.py --force-reregister` for intentional duplicate-version experiments.
- Hardened deployment failure handling so errors after model registration or reuse still write a `release_record.json` with `status=failed`, model resolution, deployment target, and exception details.
- Verified the no-deploy idempotency path against `silver_neck_77x7rny6s4`; release reused `churn-prediction-model:10` instead of creating another approved model version.
- Confirmed the default new-deployment path surfaces Azure quota failure as a diagnostic release record with `status=failed`, `model_resolution=reused`, and deployment-stage `HttpResponseError` details.
- Verified the quota-safe full deploy-reuse path by targeting `churn-endpoint/churn-smoke-silver`; the release reused `churn-prediction-model:10`, smoke-tested with response `"[0]"`, and left endpoint traffic at `churn-smoke-silver: 100`.
- Confirmed deploy smoke remains healthy while Azure's generated MLflow scoring path logs non-blocking undefined `inputs_collector` / `outputs_collector` collection warnings; full model-data collection remains out of scope until the planned monitor stage owns a custom collector implementation.
- Added local endpoint smoke payload validation and release canary inference metadata so deployment smoke fails before Azure invoke for malformed `input_data` payloads and successful releases hand off payload shape plus response preview to monitoring.
- Hardened release deployment finalization so Azure SDK create/update errors are no longer treated as automatically terminal: the deployment adapter now classifies delayed `Succeeded`, terminal `Failed`/`Canceled`, and explicit finalization-timeout outcomes separately.
- Updated `run_release.py` so post-model-resolution deployment timeout or terminal-failure outcomes still write a local `release_record.json` before the command exits non-zero.
- Recorded the motivating cloud example: the `model version 12` rollout succeeded in Azure after the original synchronous release path timed out, and the repo now treats that as a recoverable delayed-success case rather than a silent local-audit gap.
- Patched deploy-path payload-validation failure handling so a malformed smoke payload after deployment setup no longer erases truthful deployment metadata from `release_record.json`; failed records now preserve the deployment state already reached before Azure invoke was skipped.
- Added a derived `monitoring_handoff` block to `release_record.json` so successful and failed deploy paths expose a bounded, operator-readable monitoring handoff summary without pretending full production monitoring already exists.
- Replaced the generated MLflow online scoring path with a repo-owned `src/inference/score.py` entrypoint when inference capture is enabled, so endpoint payload validation and bounded canary-time collection are under repo control.
- Added deployment-level collector metadata to `release_record.json`, including collector mode, status, collection names, and any `COLLECTOR_WARNING=` degradation markers found in Azure deployment logs after canary invoke.
- Extended `monitoring_handoff` to distinguish healthy collector-backed handoff (`ready_for_collector_backed_monitoring_handoff`) from degraded collector-backed handoff (`collector_degraded_after_deploy`).
- Ran a bounded managed-online collector-binding proof spike with three repo-owned variants and confirmed all of them still reproduced Azure `inputs_collector` / `outputs_collector` runtime failures during successful canary scoring.
- Closed that spike as unsupported for the current managed-online repo-owned path, removed probe-only runtime/config scaffolding, and kept the shipped monitoring contract at `release_evidence_only`.

## 2026-04-14

- Replaced the temporary container-local repo-owned inference-capture default with an externalized `workspaceblobstore_jsonl` configuration surface in `configs/assets.yaml`.
- Added a sink-capable capture abstraction so repo-owned online scoring can write bounded JSONL records either to local debug files or to Azure Blob-backed storage.
- Patched the fixed-train MLflow bundle normalization so the repo-owned online serving environment now includes `azure-storage-blob==12.19.0` in addition to `azureml-ai-monitoring==1.0.0`.
- Tightened deployment-time monitoring truth: external repo-owned capture only becomes `healthy` after the release path can prove sink records are externally retrievable; otherwise it stays `degraded` and the monitoring handoff falls back to the basic release-evidence contract.
- Added bounded `REPO_OWNED_SCORER_INIT=` / `REPO_OWNED_SCORER_RUN=` proof markers plus deployment metadata for scorer-contract inspection, then validated one fresh Azure release on `modest_caravan_mc3tt6sltd`.
- Fixed the managed-online scorer bootstrap and feature-name reconstruction so the fresh Azure release on `modest_caravan_mc3tt6sltd` now classifies the live runtime as `repo_owned_scoring_proven` and surfaces both repo-owned proof markers in cloud logs.
- Added `run_release_failure_probe.py` plus the bounded `sample-data-invalid-feature-count.json` fixture, then proved the same Azure deployment fails cleanly inside the repo-owned scorer with `ValueError: Endpoint payload input_data row 0 expected 10 features, got 9.` while the original release artifact stays `status = succeeded`.
- Added `run_monitor_handoff.py` as the repeatable post-release monitoring wrapper that starts from a saved `release_record.json`, reruns bounded caller-side capture plus exact retrieval, and hands the evidence back into `run_monitor.py` without requiring a fresh deploy.
- Verified that the same saved release can be re-used for monitoring without redeploying; when capture sink credentials are absent, the handoff now stays truthfully in the `release_evidence_only` lane instead of overclaiming caller-side evidence.
- Completed the saved-release Blob capture proof for the same `modest_caravan_mc3tt6sltd` deployment: after fixing the caller-capture writer/reader URI contract, the repeatable handoff wrapper retrieved three exact JSONL records, kept deployment-owned capture truth at `disabled`, and still reached `capture_backed_monitoring_ready` via caller-side evidence.
- Added the first monitor-triggered retraining policy bridge on top of release-monitoring outputs, so saved release truth can now feed a bounded retraining recommendation without implying that deployment itself auto-starts new training or bypasses later gates.
- Added the next monitor-to-training handoff boundary: release and monitor evidence can now feed `run_retraining_candidate.py`, which freezes operator-declared dataset identities for validation without treating deployment evidence or capture logs as the retraining dataset itself.
- Extended that same release-to-monitor provenance chain into the first validated-candidate fixed-train smoke proof, so saved release truth can now flow through candidate freeze, candidate validation, and `run_retraining_fixed_train_smoke.py` into `run_pipeline.py` without requiring a fresh deployment or production asset mutation.
- Extended the same provenance chain one step further with `run_retraining_path_selection.py`, so a passed candidate validation can now become an explicit post-validation train-path choice while still distinguishing proven fixed-train execution from prepared-only HPO handoff.
- Added an opt-in retraining-loop release continuation so a promotable downstream retraining result can now flow into `run_release.py` and optionally `run_monitor_handoff.py` without changing `run_release.py` into a loop-owned surface or blurring `release_record.json` truth.
- Completed the first bounded end-to-end retraining-loop release-and-monitor proof: the loop continued promoted fixed-train job `modest_sponge_hxk5b23vty` into a successful release record for model version `15`, redeployed `churn-endpoint/churn-deployment`, and then reached caller-side `capture_backed_monitoring_ready` through the repeatable post-release handoff wrapper.
- Extended the same release depth to the `model_sweep` branch at the child-surface level: after HPO continuation yielded promotable child `heroic_boniato_jk85w0dvkg`, the release path reused approved model `17`, deployed it to `churn-endpoint/churn-deployment`, preserved `repo_owned_scoring_proven`, and produced a deployable release record that could feed capture-backed post-release monitoring.
- Closed the deterministic top-level `model_sweep` deployment-loop hardening follow-up: the resumed retraining loop reused promotable continuation child `heroic_boniato_jk85w0dvkg`, continued through a succeeded release, and reached a truthful post-release monitor handoff without weakening release or promotion gates.
