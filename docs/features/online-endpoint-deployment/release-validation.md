---
doc_id: online-endpoint-release-validation
doc_type: feature-guide
explains:
  features:
    - online-endpoint-deployment
    - release-monitoring-evaluator
---

# Release Validation

Use this checklist when validating `run_release.py` against Azure ML. It keeps
release evidence discoverable without committing downloaded runtime artifacts.

Helper-path note: keep root filenames in command examples here. The broader root-versus-tools ownership rule lives in `docs/pipeline_guide.md`.

## Scope

Release validation checks that a promoted fixed-training output can be resolved
to an approved Azure ML model, optionally deployed to the managed online
endpoint, and smoke-tested with the checked-in `sample-data.json` payload.

The release path should:

- refuse rejected promotion decisions
- validate declared config lineage against source job manifests when available
- reuse a matching approved registered model by default
- write `release_record.json` for both successful releases and failures after
  model resolution
- require `--force-reregister` when intentionally creating another approved
  model version for the same validated release lineage
- validate the configured endpoint smoke payload locally before invoking Azure
  ML
- preserve canary inference metadata in the release record when deployment smoke
  succeeds
- derive an explicit `monitoring_handoff` summary in the release record so the
  current monitor-stage evidence is operator-readable
- configure the repo-owned online scoring surface as deployment intent, while
  treating `release_record.json > deployment.repo_owned_scoring_*` plus Azure
  logs as the source of truth for which runtime actually executed in cloud
- only upgrade repo-owned inference capture from `configured` to `healthy` when
  the configured external sink is actually retrievable in cloud
- honor `canonical_train_config` from fixed-train manifests when HPO winner
  configs were materialized through Azure ML as a generic `train_config.yaml`
- classify deployment finalization as delayed success, terminal failure, or
  explicit timeout instead of assuming the first synchronous SDK result is the
  final Azure truth
- leave a local `release_record.json` for all post-model-resolution deploy
  outcomes, including explicit deployment-finalization timeout

## Local Checks

Before cloud validation, run the focused release tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_azureml_adapters.py tests\test_release_workflow.py tests\test_release_record.py tests\test_run_release.py tests\test_run_pipeline_config.py -q
```

When validating HPO-to-release handoff behavior, also include the winner-export
tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_export_hpo_winner_config.py tests\test_release_workflow.py tests\test_run_release.py -q
```

When validating endpoint payload and repo-owned inference-capture changes,
include the focused inference and deployment tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_inference_payloads.py tests\test_online_scoring.py tests\test_azureml_adapters.py tests\test_run_release.py tests\test_release_record.py -q
```

## Canary Payload Validation

The release deploy path validates the configured `deployment.smoke_payload`
from `configs/assets.yaml` before calling Azure ML endpoint invoke. The current
endpoint consumes the
processed-feature MLflow payload shape:

```json
{
  "input_data": [[0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]]
}
```

The validator rejects:

- missing or invalid JSON files
- payloads without `input_data`
- non-list or empty `input_data`
- rows that are not lists
- rows with a feature count other than `10`
- non-numeric or boolean feature values

Successful deploy smoke records include `canary_inference` in
`release_record.json`. This is a monitor-stage handoff record, not full
production data collection. It stores payload path, payload format, row count,
feature count, validation status, endpoint/deployment coordinates, registered
model coordinates, and a bounded response preview.

Managed online deployment is configured with the repo-owned
`src/inference/score.py` entrypoint, and the successful 2026-04-14 fresh
release-plus-monitor smoke proof surfaced both bounded
`REPO_OWNED_SCORER_INIT=` and `REPO_OWNED_SCORER_RUN=` markers in Azure logs.

That means the current supported interpretation is:

- `code_configuration.scoring_script="src/inference/score.py"` is still just
  configured intent until cloud logs confirm it
- `release_record.json > deployment.repo_owned_scoring_status` remains the
  repo’s machine-readable classification of the observed cloud contract
- `repo_owned_scoring_proven` means the managed online runtime executed the
  repo-owned scorer successfully in cloud
- future endpoint payload or capture changes should preserve this proof path by
  keeping the `REPO_OWNED_SCORER_*` markers visible in Azure logs

The repo now also has a bounded failure-path proof for the same scorer
contract. Using [sample-data-invalid-feature-count.json](../../../sample-data-invalid-feature-count.json),
the negative cloud probe reached Azure scoring, emitted a live
`REPO_OWNED_SCORER_RUN=` marker, and surfaced the exact scorer-owned failure:

- `ValueError: Endpoint payload input_data row 0 expected 10 features, got 9.`

That result matters because the normal release and caller-capture wrappers still
validate payloads locally first. The dedicated negative probe exists only to
prove scorer failure semantics in Azure; it does not weaken the default safety
path.

The scorer now derives payload column names from the deployed model when the
wrapped sklearn model exposes `feature_names_in_`. This is required because the
fixed-train model is fitted on named columns such as `Age`, `Balance`, and
`CreditScore`, while the checked-in canary payload still uses positional
`input_data`. The scorer must reconstruct the trained feature names before
calling the model, or Azure canary scoring will fail with a feature-name
mismatch even when container startup is healthy.

The canonical repo-owned capture path still targets an external sink rather
than container-local files, but it remains subordinate to the runtime proof:

- `configs/assets.yaml` defaults to `workspaceblobstore_jsonl`
- capture stays disabled by default until storage credentials are intentionally
  configured
- capture cannot be called active in cloud unless the runtime contract is
  compatible with repo-owned scoring hooks and the sink is externally proven
- the serving container expects:
  - `INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING`
  - `INFERENCE_CAPTURE_STORAGE_CONTAINER`
- the release runner resolves those values from the env-var names declared in
  `configs/assets.yaml`

Successful and failed deploy records now also include
`monitoring_handoff` in `release_record.json`. This is the bounded operator
summary of what the repo can prove today:

- `ready_for_basic_monitoring_handoff`
  - deployment reached `Succeeded`
  - traffic updated
  - canary payload validated
  - canary scoring invoked successfully
- `ready_for_repo_owned_inference_capture_handoff`
  - deployment reached `Succeeded`
  - traffic updated
  - canary scoring invoked successfully
  - repo-owned inference capture is enabled, externally verified, and reported
    healthy
- `capture_degraded_after_deploy`
  - deployment reached `Succeeded`
  - canary scoring passed
  - repo-owned inference capture reported degraded state or external sink proof
    could not be established
- `canary_failed_after_deploy`
  - deployment reached a usable state
  - traffic may already be updated
  - canary validation or scoring then failed
- `deploy_incomplete_or_timed_out`
  - deployment never reached a confirmed healthy terminal state within the
    release finalization budget
- `deploy_failed_before_handoff`
  - deployment failed before a usable monitoring handoff was established

The current implementation keeps these evidence tiers:

- `evidence_level = release_evidence_only`
  - release evidence is trustworthy, but collector-backed capture is not a
    supported healthy contract in the current Azure deployment path
- `evidence_level = repo_owned_inference_capture`
  - release evidence is trustworthy and repo-owned sampled inference capture is
    externally verified for later monitoring

That wording is intentional. It means the repo has trustworthy release and
canary evidence, but still does not claim full production dashboards, alerts,
or long-horizon production observability. The bounded collector-binding spike
also confirmed that the current managed-online repo-owned path should not claim
collector-backed monitoring support.

## Deployment Capture vs Caller Capture

The artifact contract now separates three truths instead of overloading one
generic capture status:

- `release_record.json > deployment_capture`
  - deployment-owned truth only
  - whether endpoint-side capture was enabled, which mode was configured, and
    whether deployment itself reported that sink as healthy, degraded, or
    disabled
- `monitor_summary.json > caller_capture`
  - post-release caller-side evidence only
  - whether bounded sampled request/response records were actually retrieved
    and evaluated
- `release_monitor_summary.json > handoff`
  - the operator-facing reconciliation of release truth and caller-capture
    truth

Concrete example:

- before
  - release record said capture was disabled
  - monitor summary said `capture_backed`
  - operators had to infer that those referred to different evidence planes
- now
  - release record can truthfully say `deployment_capture.status = disabled`
  - monitor summary can truthfully say
    `capture_evidence_source = caller_side`
  - wrapper summary can truthfully say
    `handoff.status = capture_backed_monitoring_ready`

These are not contradictory. They mean the endpoint itself was not configured
to emit capture, but the approved caller wrapper still captured bounded
request/response evidence and the monitor evaluator used that evidence
successfully.

That saved release truth is now also the upstream provenance for the bounded
monitor-triggered retraining path:

```text
release_record.json
  -> run_monitor.py / retraining_decision.json
  -> run_retraining_candidate.py
  -> validation_summary.json (passed)
  -> run_retraining_path_selection.py
  -> fixed_train or model_sweep
  -> run_retraining_fixed_train_smoke.py + run_pipeline.py when fixed_train is selected
  -> run_retraining_hpo_smoke.py + run_hpo_pipeline.py when model_sweep is selected
```

This remains smoke-scoped. It proves that validated retraining candidates can
reconnect to an explicit post-validation train-path choice while preserving
release and monitor lineage, but it does not imply automatic winner export,
promotion, release, or deployment of the retrained model. Both downstream
branches are now executable in smoke scope while keeping `run_pipeline.py` and
`run_hpo_pipeline.py` as the authoritative child submission surfaces.

The retraining loop can now continue one step further, but only when operators
opt in with `run_retraining_loop.py --release-mode ...` and only when the
selected downstream training path yields promotable evidence. That continuation
remains composition-only:

- promotion truth stays in `promotion_decision.json`
- release truth stays in `release_record.json`
- optional post-release monitoring truth stays in `handoff_summary.json`

`run_retraining_loop.py` may summarize those child artifacts, but it must not
rewrite their meaning or claim a healthy release when the child release or
handoff surface disagrees.

For deterministic `model_sweep` proof closure after a promotable continuation
child already exists, the loop also supports
`--resume-continuation-summary <retraining_hpo_to_fixed_train_summary.json>`.
That resume seam reuses already-proven continuation evidence instead of
rerunning the stochastic HPO branch, but release and monitor truth still remain
child-owned in `release_record.json` and `handoff_summary.json`.

The repo now includes a separate monitor evaluator that consumes this release
handoff truth:

```powershell
.\.venv\Scripts\python.exe run_monitor.py `
  --release-record release_artifacts_scoring_contract_proof\modest_caravan_mc3tt6sltd\release_record.json `
  --config configs\monitor.yaml `
  --output-dir monitored_release_scoring_contract_proof
```

Expected healthy result for the current proven cloud contract:

- `monitor_summary.json > monitor_status = limited_but_healthy`
- `monitor_summary.json > evidence_level = release_evidence_only`
- `monitor_summary.json > runtime_contract = repo_owned_scoring_proven`
- `monitor_manifest/step_manifest.json > status = succeeded`

For repeated post-release checks against a saved release record, use the
monitoring-first handoff wrapper instead of the release-plus-monitor smoke
wrapper:

```powershell
.\.venv\Scripts\python.exe run_monitor_handoff.py `
  --release-record monitored_release_monitor_smoke_fresh_current_retry4\release\modest_caravan_mc3tt6sltd\release_record.json `
  --azure-config config.env `
  --capture-config configs\inference_capture_blob.yaml `
  --monitor-config configs\monitor.yaml `
  --probe-request sample-data.json `
  --probe-request tmp_endpoint_probe\payload_b.json `
  --probe-request tmp_endpoint_probe\payload_c.json `
  --output-dir monitor_handoff_modest_caravan
```

Use this wrapper when release truth already exists and the operator goal is to
gather fresh bounded monitoring evidence without redeploying. It should leave a
stable bundle with:

- `handoff_summary.json`
- `capture\capture_manifest_*.json`
- `downloaded_capture\`
- `monitor\retraining_decision.json`
- `monitor\monitor_summary.json`

The 2026-04-14 repeatable Blob-backed proof for
`modest_caravan_mc3tt6sltd` now succeeds on this exact path. With
`INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING` and
`INFERENCE_CAPTURE_STORAGE_CONTAINER` set, the wrapper:

- invoked the proven `churn-endpoint/churn-deployment` three times
- wrote caller-side manifests with exact `azureblob://...` record locations
- downloaded the exact three referenced JSONL blobs
- finished with:
  - `handoff_summary.json > caller_capture.status = retrieved`
  - `handoff_summary.json > handoff.status = capture_backed_monitoring_ready`
  - `monitor_summary.json > monitor_status = capture_backed`
  - `monitor_summary.json > capture_evidence_source = caller_side`

That proof also matters because deployment-owned capture stayed disabled in the
saved `release_record.json`. The healthy result comes from caller-side evidence,
not from endpoint-side collector binding.

The distinction is intentional:

- `run_release_monitor_smoke.py`
  - release-proof oriented
  - can deploy or reuse
  - good for lifecycle validation
- `run_monitor_handoff.py`
  - monitoring-first
  - requires a saved release record
  - good for repeatable post-release monitoring checks

Caller-side capture is the preferred next production-evidence path for this
managed-online deployment model. It keeps Azure ML as the prediction service,
but moves request/response evidence writing into the invoking wrapper or a
future gateway instead of depending on endpoint-side `score.py` hooks:

```powershell
.\.venv\Scripts\python.exe run_inference_capture.py `
  --endpoint-name churn-endpoint `
  --deployment-name churn-deployment `
  --request-file sample-data.json `
  --config configs\inference_capture_blob.yaml `
  --azure-config config.env `
  --output-manifest captured_inference\capture_manifest.json `
  --model-name churn-prediction-model `
  --model-version 12
```

The safe local mode uses `configs/inference_capture.yaml`. Blob mode uses
`configs/inference_capture_blob.yaml` and reads storage credentials only from:

- `INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING`
- `INFERENCE_CAPTURE_STORAGE_CONTAINER`

Blob-backed retrieval is exact-path based. The current proven URI contract is:

- writer emits `azureblob://<account>/<container>/<blob>`
- downloader strips the account and container segments, then downloads that
  exact blob from the configured container
- retrieval may retry briefly for fresh-write visibility, but it does not fall
  back to prefix search or fuzzy matching

Direct calls to `az ml online-endpoint invoke` or
`MLClient.online_endpoints.invoke` do not automatically write caller-side
capture records unless they are routed through this wrapper or an equivalent
gateway.

When sampled production capture has been retrieved from the configured external
sink, pass it back into the same evaluator rather than creating a second
monitoring path:

```powershell
.\.venv\Scripts\python.exe run_monitor.py `
  --release-record release_artifacts_scoring_contract_proof\modest_caravan_mc3tt6sltd\release_record.json `
  --config configs\monitor.yaml `
  --capture-path downloaded_capture\monitoring\inference_capture `
  --output-dir monitored_release_capture_backed
```

The current production-evidence checks remain bounded. `capture_backed` should
only be considered healthy when:

- records are retrievable
- feature counts are consistent
- prediction distribution is available
- the max single-class share stays within the configured threshold

The repo now also supports a thin end-to-end operator wrapper that composes the
same release, caller-capture, retrieval, and monitor surfaces without creating
a parallel release or monitor implementation:

```powershell
.\.venv\Scripts\python.exe run_release_monitor_smoke.py `
  --release-record release_artifacts_scoring_contract_proof\modest_caravan_mc3tt6sltd\release_record.json `
  --capture-config configs\inference_capture_blob.yaml `
  --monitor-config configs\monitor.yaml `
  --probe-request sample-data.json `
  --probe-request tmp_endpoint_probe\payload_b.json `
  --probe-request tmp_endpoint_probe\payload_c.json `
  --output-dir release_monitor_smoke_proof
```

That command should leave one bounded automation folder containing:

- `release_monitor_summary.json`
- `capture_manifests\`
- `downloaded_capture\`
- `monitor\monitor_summary.json`

The wrapper summary is now intentionally split into:

- `deployment_capture`
  - release-time, deployment-owned capture truth
- `caller_capture`
  - wrapper-executed caller-side capture outcome
- `monitor`
  - evaluator result and evidence level
- `handoff`
  - one operator-readable final state

Use the wrapper when you want one operator-facing proof bundle. Use the child
commands directly when you need to debug release, capture, or monitoring in
isolation.

## Repo-Owned Scorer Failure Probe

Use the dedicated negative probe when you want to prove that Azure is running
the repo-owned scorer and that scorer-side failures stay debuggable:

```powershell
.\.venv\Scripts\python.exe run_release_failure_probe.py `
  --release-record monitored_release_monitor_smoke_fresh_current_retry4\release\modest_caravan_mc3tt6sltd\release_record.json `
  --config config.env `
  --request-file sample-data-invalid-feature-count.json `
  --output-dir negative_probe_modest_caravan
```

Expected truth:

- `failure_probe_summary.json > status = intentional_failure_observed`
- `failure_probe_summary.json > artifact_truth.release_status = succeeded`
- `failure_probe_summary.json > artifact_truth.repo_owned_scoring_status = repo_owned_scoring_proven`
- `failure_probe_summary.json > failure.error_type = ValueError`
- `failure_probe_summary.json > failure.error_message = Endpoint payload input_data row 0 expected 10 features, got 9.`
- `failure_probe_summary.json > failure.log_excerpt` includes `REPO_OWNED_SCORER_RUN=`

This probe is intentionally separate from normal release smoke. Normal release
and caller-capture commands must still reject malformed payloads locally before
Azure invoke.

If payload validation fails after deployment provisioning or traffic update has
already happened, the release should still write a failed record with truthful
deployment metadata, for example:

- `deployment.deployment_state = Succeeded`
- `deployment.traffic_updated = true`
- `deployment.smoke_invoked = false`
- `failure.error_type = ValueError`

## Deployment Finalization Recovery

Azure ML deployment create/update calls can raise locally while Azure keeps
provisioning in the background. The repo now treats Azure deployment state as
the source of truth:

- delayed Azure `Succeeded`
  - release writes `status=succeeded`
  - deployment metadata includes `recovery_used=true`
- terminal Azure `Failed` or `Canceled`
  - release writes `status=failed`
  - failure details remain under `failure`
- non-terminal state after the finalization budget expires
  - release writes `status=failed`
  - `failure.failure_stage` is `deployment_finalization_timeout`
  - deployment metadata records `finalization_timed_out=true`

This avoids the old operator gap where Azure succeeded but the local command
left no final release record.

The same principle now applies to pre-invoke payload validation failures inside
the deploy path: the record should preserve how far deployment actually got,
not fall back to an empty deployment shell.

## No-Deploy Reuse Check

Use this when validating idempotency without touching endpoint deployments:

```powershell
.\.venv\Scripts\python.exe run_release.py `
  --job-name silver_neck_77x7rny6s4 `
  --config config.env `
  --data-config configs\data_smoke_eval.yaml `
  --train-config configs\train_smoke.yaml `
  --download-dir release_artifacts_idempotency_reuse
```

Expected `release_record.json` fields:

- `status`: `succeeded`
- `model_resolution`: `reused`
- `registered_model.version`: the existing approved model version

Do not commit the `release_artifacts_*` folder. Treat it as local validation
evidence only.

## Quota-Safe Deploy Reuse Check

In quota-limited workspaces, prefer updating the existing smoke deployment
instead of creating another `Standard_D2as_v4` deployment.

Current smoke deployment override:

```powershell
$env:AML_ONLINE_DEPLOYMENT_NAME = "churn-smoke-silver"
.\.venv\Scripts\python.exe run_release.py `
  --job-name silver_neck_77x7rny6s4 `
  --config config.env `
  --data-config configs\data_smoke_eval.yaml `
  --train-config configs\train_smoke.yaml `
  --download-dir release_artifacts_idempotency_reuse_deploy_existing `
  --deploy
Remove-Item Env:AML_ONLINE_DEPLOYMENT_NAME
```

Expected `release_record.json` fields:

- `status`: `succeeded`
- `model_resolution`: `reused`
- `deployment.endpoint_name`: `churn-endpoint`
- `deployment.deployment_name`: `churn-smoke-silver`
- `smoke_test_response`: `"[0]"`
- `deployment.inference_capture_mode`: `release_evidence_only`
- `monitoring_handoff.status`: `ready_for_basic_monitoring_handoff`

After the smoke invoke, inspect deployment logs:

```powershell
az ml online-deployment get-logs `
  --name churn-smoke-silver `
  --endpoint-name churn-endpoint `
  --resource-group rg-churn-ml-project `
  --workspace-name churn-ml-workspace `
  --lines 100
```

Current managed-online deployment should treat Azure collector binding as
unsupported, but the live scoring runtime is now proven on the primary path.
Use the release record, especially `monitoring_handoff`,
`deployment.repo_owned_scoring_*`, `deployment`, and `canary_inference`, plus
endpoint state, endpoint logs, and any externally retrievable capture sink
records as the monitor-stage evidence surface.

If a deploy attempt targets a new deployment name and hits Azure quota, the
release should still write a failed diagnostic record after model resolution.
Expected fields include:

- `status`: `failed`
- `model_resolution`: `reused`
- `failure.failure_stage`: `deployment`
- `failure.error_type`: `HttpResponseError`

## Evidence Handling

Record dated summaries in `history.md`. Keep full JSON records in local
`release_artifacts_*` folders or Azure ML run artifacts, not in docs.
