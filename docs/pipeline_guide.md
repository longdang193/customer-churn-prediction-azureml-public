# Pipeline Guide

This guide describes the intended workflow model for the churn project and the current execution surfaces that exist in the repo today.

Canonical lifecycle entrypoints stay at the repo root. Secondary operator helpers now live under `tools/`, while the old root helper filenames remain as compatibility wrappers during the migration window.

Migration policy:

- use root filenames when the operator-facing point is "run this command"
- use `tools/...` paths when the maintenance-facing point is "this file owns the helper implementation"
- keep the canonical lifecycle surface at root:
  `run_pipeline.py`, `run_hpo_pipeline.py`, `run_hpo.py`, `run_monitor.py`, `run_release.py`, and `run_retraining_loop.py`
- add new helper implementations under `tools/`
- treat helper root wrappers as compatibility surfaces rather than root-owned implementations
- keep non-entrypoint utility modules such as `hpo_utils.py` out of the entrypoint-cleanup scope
- keep `scripts/` as the maintenance and adapter-governance lane
- prefer bounded local proof/output areas such as `.tmp-tests/` for new smoke bundles instead of creating fresh root-level artifact folders by default

## 1. Canonical Lifecycle

The target lifecycle for the project is:

```text
raw data asset
  -> data validation and drift check
  -> data preparation
  -> optimization path OR fixed training path
  -> evaluation and promotion gate
  -> model registry
  -> deployment
  -> monitoring handoff
  -> monitoring
```

This is the architecture the repo should document and evolve toward.

## 2. Current Stage Boundaries

The current stage contracts live under `docs/stages/` and currently cover:

- `data_validate`
- `data_prep`
- `model_sweep`
- `fixed_train`
- `model_promote`
- `online_deploy`
- `monitor`

## 2.1 Smoke Surface

Smoke runs are opt-in and exist to test wiring and gate behavior, not model quality.

Current smoke assets:

- `data/smoke/positive/churn_smoke.csv`
  - positive-path raw fixture with the production churn schema and enough class representation for stratified splitting
- `data/smoke/eval/churn_smoke_eval.csv`
  - alternate current-data fixture for smoke validation; `run_pipeline.py --data-config configs/data_smoke_eval.yaml` compares this asset against the base smoke asset before data prep
- `data/smoke/negative/churn_validation_edge.csv`
  - negative-path validation fixture with intentionally excessive missingness
- `configs/data_smoke.yaml`
  - local data-prep and validation smoke profile that writes to `data/processed_smoke`
- `configs/train_smoke.yaml`
  - fixed-training smoke profile for `run_pipeline.py --data-config configs/data_smoke.yaml --train-config configs/train_smoke.yaml`
- `configs/hpo_smoke.yaml`
  - low-budget sweep profile for `run_hpo_pipeline.py --hpo-config configs/hpo_smoke.yaml` or `run_hpo.py --config configs/hpo_smoke.yaml`
- `sample-data.json`
  - Azure ML endpoint request payload used by deployment smoke checks
- `release_record.json`
  - the repo-owned release and monitoring handoff artifact; now includes detailed deployment evidence, canary evidence, and a derived `monitoring_handoff` summary
  - release records now also carry `deployment_capture`, which is deployment-owned capture truth only
  - the default repo-owned external capture mode is now `workspaceblobstore_jsonl`, which remains disabled until storage credentials are intentionally configured
- `run_inference_capture.py`
  - caller-side or gateway-side endpoint invoke wrapper for writing bounded JSONL capture evidence outside Azure managed-online scoring internals
  - implementation now lives in `tools/monitoring/run_inference_capture.py`
- `run_release_monitor_smoke.py`
  - thin proof wrapper that composes release, caller capture, exact capture retrieval, and monitor evaluation into one bounded output folder
  - its summary now separates `deployment_capture`, `caller_capture`, `monitor`, and `handoff` so operators can see why a bundle succeeded even when endpoint-side capture stayed disabled
  - implementation now lives in `tools/monitoring/run_release_monitor_smoke.py`
- `run_release_failure_probe.py`
  - bounded negative Azure probe for the now-proven repo-owned scorer; writes `failure_probe_summary.json` without mutating the original `release_record.json`
  - implementation now lives in `tools/release/run_release_failure_probe.py`

## 2.2 Monitor-Triggered Retraining Policy

The repo now has a bounded decision bridge between monitoring evidence and
later retraining work.

Current trigger meanings:

- `no_retraining_signal`
  - current evidence does not justify opening a retraining candidate
- `retraining_candidate`
  - current bounded evidence is strong enough to recommend opening a retraining
    candidate
- `investigate_before_retraining`
  - evidence is degraded or incomplete enough that operators should inspect it
    before freezing a dataset snapshot or choosing a train path

The bounded retraining policy now also carries recommendation evidence so the
operator can see how strong the path suggestion is without treating it as an
autonomous controller:

- `path_recommendation`
  - one of `none`, `fixed_train`, or `model_sweep`
- `path_recommendation_reason_codes`
  - the bounded reasons for that path suggestion, for example
    `no_retraining_signal`, `bounded_refresh_candidate`, or
    `investigate_before_retraining`
- `drift_severity`
  - the current bounded severity label the evaluator assigns from the evidence
    it actually has
- `signal_persistence`
  - whether the recommendation is based on a single bounded event or a repeated
    signal policy
- `policy_confidence`
  - the evaluator's confidence in the recommendation, kept bounded as
    `strong`, `moderate`, or `weak`

Important boundary:

- monitor may recommend retraining
- it does not automatically:
  - freeze a dataset
  - run fixed training
  - run HPO
  - promote a model
  - deploy a model

Any later retraining still requires:

- a concrete dataset snapshot
- `validate_data`
- an explicit choice between fixed training and HPO
- the normal promotion and release gates

The next repo-owned bridge after the monitor policy is now:

```text
monitor_summary.json / retraining_decision.json
  -> run_retraining_candidate.py
  -> retraining_candidate_manifest.json
  -> validation_handoff.json
  -> optional validate_data execution
```

That bridge freezes explicit dataset identities. It does not derive the
training dataset from caller-side capture records.

Invocation versus ownership:

- operators still run `run_retraining_candidate.py`
- the helper implementation now lives in `tools/retraining/run_retraining_candidate.py`

The next smoke-scoped bridge after a passed candidate validation is now:

```text
retraining_candidate_manifest.json
  + validation_summary.json (status = passed)
  -> run_retraining_path_selection.py
  -> selected path
  -> fixed-train smoke execution OR HPO smoke execution
```

Important boundary:

- this makes the train-path choice explicit after validation instead of leaving it
  to operator convention
- today the overall smoke-scoped retraining chain has executable downstream
  bridges for both training surfaces:
  - `run_retraining_fixed_train_smoke.py` for `fixed_train`
  - `run_retraining_hpo_smoke.py` for `model_sweep`
- `run_retraining_path_selection.py` executes the fixed-train branch directly
  and emits the explicit HPO handoff artifact for the `model_sweep` branch
- this proves the reconnect from monitor policy back into both training
  surfaces without collapsing them into one opaque retraining wrapper
- it still does not auto-release or auto-deploy
- `run_pipeline.py` remains the authoritative fixed-train submission surface
- `run_hpo_pipeline.py` remains the authoritative HPO submission surface
- the bridges use explicit current/reference dataset overrides only for these
  smoke-scoped retraining proofs
- those helper implementations now live under `tools/retraining/`, while the
  root filenames remain the stable invocation surface during migration

The repo now also has a bounded phase-1 loop coordinator:

```text
release_record.json + monitor_summary.json / retraining_decision.json
  -> run_retraining_loop.py
  -> freeze_only OR freeze_and_validate OR validate_and_select_path OR submit_selected_path
```

Important boundary:

- `run_retraining_loop.py` is orchestration only
- it composes the existing candidate, validation, path-selection, fixed-train,
  and HPO bridges instead of reimplementing their logic
- with `release_mode=disabled`, it can stop after exactly one selected
  downstream smoke bridge
- with release continuation enabled, it may continue only after
  promotion-facing evidence is proven promotable
- even then, release truth remains owned by `run_release.py` /
  `release_record.json`, not by the loop wrapper
- optional post-release monitoring handoff remains owned by
  `run_monitor_handoff.py`
- it still stops before autonomous deployment governance; continuation is
  explicit and opt-in

The next bounded continuation after a completed `model_sweep` proof is now:

```text
retraining_hpo_smoke_summary.json
  + hpo_summary.json
  + winner_train_manifest/step_manifest.json
  + winner_train_config/train_config.yaml (or exported equivalent)
  -> run_retraining_hpo_to_fixed_train.py
  -> fixed-train smoke submission
```

Important boundary:

- this keeps HPO exploratory until a concrete winner is validated and
  continuation-safe
- `run_retraining_hpo_to_fixed_train.py` validates winner consistency before it
  reuses or exports an effective fixed-train config
- continuation still flows back through the authoritative fixed-train bridge
  instead of releasing directly from HPO
- promotion, release, and deployment remain downstream gates

The retraining loop now has one bounded opt-in continuation beyond those
training bridges:

```text
run_retraining_loop.py --mode submit_selected_path
  + --release-mode after_promotion
    -> selected training bridge
    -> promotion gate
    -> run_release.py

run_retraining_loop.py --mode submit_selected_path
  + --release-mode after_release_monitor_handoff
    -> selected training bridge
    -> promotion gate
    -> run_release.py
    -> run_monitor_handoff.py
```

Important boundary:

- `--release-mode` is opt-in and defaults to `disabled`
- the loop never overrides child truth from `promotion_decision.json`,
  `release_record.json`, or `handoff_summary.json`
- failed promotion blocks release continuation instead of letting the loop
  guess at release eligibility

For deterministic proof hardening on `model_sweep`, the loop now also supports:

```text
run_retraining_loop.py --mode submit_selected_path
  + --resume-continuation-summary <retraining_hpo_to_fixed_train_summary.json>
  + --release-mode after_release_monitor_handoff
    -> resume already-promotable continuation evidence
    -> run_release.py
    -> run_monitor_handoff.py
```

Important boundary:

- this resume seam is for deterministic top-level loop closure after the child
  surfaces are already proven
- it does not invent a new training path or bypass promotion, release, or
  monitor truth
- resumed provenance is recorded in `retraining_loop_summary.json`

## 3. Optimization Path

The optimization path is periodic and deliberately exploratory.

Current live execution surfaces:

- `run_hpo_pipeline.py`
- `run_hpo.py`
- `inspect_hpo_run.py`
- `export_hpo_winner_config.py`
- `notebooks/main.ipynb`

Implementation ownership for the two CLI helpers now lives in:

- `tools/hpo/inspect_hpo_run.py`
- `tools/hpo/export_hpo_winner_config.py`

Current supporting code:

- `hpo_utils.py`
- `src/azureml/`
- `src/run_sweep_trial.py`
- `configs/hpo.yaml`
- `configs/hpo_smoke.yaml`

Ownership note:

- operators may still invoke the root wrapper names `inspect_hpo_run.py` and `export_hpo_winner_config.py`
- implementation ownership for those helpers lives under `tools/hpo/`
- `hpo_utils.py` is a reusable utility module, not part of the operator entrypoint surface

Current purpose:

- submit one Azure ML HPO parent pipeline that reuses validation and data prep before per-model sweeps
- submit Azure ML sweeps from a script-first entrypoint
- compare model candidates
- export the selected best configuration back into `configs/train.yaml`

Target shape:

```text
data asset
  -> validate data
  -> data prep
  -> Azure ML sweep branches
  -> collect HPO summary
  -> analyze results
  -> export approved configuration
```

Current implementation direction:

- `run_hpo_pipeline.py` is the canonical end-to-end Azure ML submission surface when you want one graph and one operator summary
- `run_hpo.py` is the lower-level Azure ML SDK v2 direct rerun surface when processed data already exists
- `run_hpo.py` accepts `--config` so alternate sweep profiles such as `configs/hpo_smoke.yaml` are reachable from the direct rerun surface
- the HPO pipeline reuses `validate_data` and `data_prep` instead of duplicating prep logic inside the sweep path
- the HPO pipeline writes `hpo_summary.json`, `hpo_summary.md`, and `hpo_manifest/step_manifest.json` for operator review
- HPO family winner selection is now explicit and deterministic: primary metric first, then configured `selection.secondary_metric`, then configured `selection.family_priority`, then a stable final fallback
- tied-family outcomes now surface `selection_policy`, `tie_break_reason`, and `tie_candidates` in HPO summary and winner-manifest artifacts so operators can tell why a winner was chosen
- Azure ML structured file outputs are exposed as folder outputs that contain extension-bearing files so Studio can render them, for example `hpo_summary/hpo_summary.json`, `hpo_summary_report/hpo_summary_report.md`, `candidate_metrics/candidate_metrics.json`, and `winner_candidate_metrics/candidate_metrics.json`
- `export_hpo_winner_config.py` turns the selected HPO winner back into a fixed train config so the standard train pipeline can retrain the chosen family without manual parameter copying
- `run_retraining_hpo_to_fixed_train.py` is the bounded continuation helper that validates one completed retraining HPO proof and hands the winner back into the fixed-train smoke bridge
- exported winner configs now carry `lineage.canonical_train_config`, and fixed-train manifests persist that alias so `run_release.py` can validate against the stable repo config identity instead of the AML materialized `train_config.yaml` basename
- each sweep branch now exposes the selected `hpo_config` input and emits its own `hpo_manifest/step_manifest.json` in addition to the trial `train_manifest`
- direct `run_hpo.py` now requests the same trial-level outputs as the HPO pipeline branch surface: `model_output`, `mlflow_model`, `candidate_metrics`, `train_manifest`, and `hpo_manifest`
- the HPO parent pipeline now surfaces per-family `*_model_output`, `*_mlflow_model`, `*_candidate_metrics`, `*_train_manifest`, and `*_hpo_manifest` outputs so operators can inspect the right family without opening Azure-generated controller/trial jobs first
- the HPO parent also materializes canonical winner outputs: `winner_model_output`, `winner_mlflow_model`, `winner_candidate_metrics`, `winner_train_manifest`, and `winner_hpo_manifest`
- `inspect_hpo_run.py` is the repo-owned inspection helper for one HPO parent run
- `notebooks/main.ipynb` remains the review surface for interactive analysis and best-config writeback

### 3.3 Exporting The HPO Winner Into Fixed Training

When HPO selects a winner and you want to retrain it through the standard fixed
pipeline, export a concrete train config instead of copying parameters by hand:

```powershell
.\.venv\Scripts\python.exe export_hpo_winner_config.py `
  --run-dir downloaded_hpo_patient_main `
  --base-config configs\train_smoke.yaml `
  --output-config configs\train_smoke_hpo_winner.yaml `
  --experiment-name train-smoke-hpo-winner `
  --display-name train-smoke-hpo-winner
```

Then submit the normal fixed-train pipeline with that generated config. This is
part of the intended HPO-to-train workflow, not a troubleshooting-only path.

If you save the exported config under a stable repo path such as
`configs/train_smoke_hpo_winner_rf.yaml`, that canonical path now survives into
the later fixed-train manifest and release-lineage checks even when Azure ML
materializes the runtime input as a generic `train_config.yaml`.

### 3.1 Azure ML Sweep Hierarchy

Pipeline-embedded sweeps appear with an extra Azure ML hierarchy:

```text
parent pipeline
  -> sweep pipeline node
    -> Azure ML sweep controller job
      -> Azure ML trial jobs
```

That means random names such as `loyal_basin_...` or `heroic_eye_...` are not a
repo bug. They are Azure ML controller and trial jobs under the stable
repo-owned sweep node.

Use these surfaces first:

- parent `hpo_summary`
- parent `hpo_manifest`
- parent `winner_model_output`
- parent `winner_mlflow_model`
- parent `winner_candidate_metrics`
- parent `*_model_output`
- parent `*_mlflow_model`
- parent `*_hpo_manifest`
- parent `*_train_manifest`
- `inspect_hpo_run.py`

Only dig into controller/trial jobs when the parent artifacts point to a real
family-level issue.

### 3.2 Smoke vs Ranking HPO

The repo now treats HPO profiles as two operating modes:

- smoke HPO
  - cheap
  - wiring/artifact validation
  - `configs/hpo_smoke.yaml` with smoke data such as `configs/data_smoke_eval.yaml`
- ranking HPO
  - meaningful family comparison
  - use the default `configs/hpo.yaml` path, or another larger-eval profile when added

Smoke HPO is expected to be unstable for model ranking when the eval split is
tiny. It is still valuable for validating:

- graph shape
- manifests
- branch outputs
- summary generation

## 4. Fixed Training Path

The fixed training path is the repeatable production-oriented path.

Current live execution surface:

- `run_pipeline.py`
- `run_retraining_path_selection.py`
- `run_retraining_fixed_train_smoke.py`

Current supporting code:

- `src/azureml/`
- `src/train.py`
- `src/config/runtime.py`
- `aml/components/data_prep.yaml`
- `aml/components/train.yaml`
- `aml/components/promote_model.yaml`
- `configs/train.yaml`
- `configs/train_smoke.yaml`
- `run_release.py`

Current purpose:

- run the AML data prep and training components
- train using the selected fixed configuration
- emit a deployable MLflow model bundle for release
- emit candidate metrics for the best fixed-training result
- evaluate the candidate against baseline metrics derived from the latest approved Azure ML model
- produce model artifacts plus a promotion decision artifact
- Studio-readable fixed-train scalar artifacts are emitted as folder outputs with explicit filenames, for example `validation_summary/validation_summary.json`, `candidate_metrics/candidate_metrics.json`, `parent_run_id/parent_run_id.txt`, and `promotion_decision/promotion_decision.json`
- accept explicit current/reference dataset override inputs when the smoke
  retraining handoff needs to feed a validated candidate into the normal fixed
  pipeline without mutating production asset defaults

The repo now also has a monitor-to-train smoke proof bridge:

- `run_retraining_path_selection.py`
  - consumes a passed candidate validation plus retraining recommendation
  - writes the explicit selected-path artifact
  - can invoke the proven fixed-train smoke bridge
  - writes the explicit HPO handoff artifact when `model_sweep` is selected
  - stops before release and deployment
- `run_retraining_fixed_train_smoke.py`
  - consumes a frozen retraining candidate plus a passed validation summary
  - writes the exact fixed-train invocation contract for dry-run inspection
  - can submit the normal fixed-train pipeline with explicit current/reference
    dataset overrides
  - preserves candidate and validation lineage
  - stops before release and deployment
- `run_retraining_hpo_smoke.py`
  - consumes a frozen retraining candidate, a passed validation summary, and a
    `model_sweep` path-selection artifact
  - writes the exact HPO invocation contract for dry-run inspection
  - can submit the normal HPO pipeline with explicit current/reference dataset
    overrides
  - preserves release, candidate, validation, and path-selection lineage
  - stops before winner export, promotion, release, and deployment

Target shape:

```text
data asset
  -> validate data
  -> data prep
  -> fixed training pipeline
  -> evaluation and promotion gate
  -> register model
  -> deploy
```

Current implementation direction:

- `run_pipeline.py` now executes validation, data prep, fixed training, and promotion in one AML pipeline submission
- `run_pipeline.py` now uses the shared quiet AML submission adapter so known benign SDK console noise is filtered the same way as the HPO entrypoints
- `run_pipeline.py` now materializes the promotion baseline from Azure ML approved-model tags at submission time
- `run_pipeline.py --data-config ... --train-config ...` passes selected policy files into validation, data prep, training, and promotion components
- `run_pipeline.py` attaches lightweight lineage tags from runtime data assets, selected configs, and `configs/assets.yaml` to submitted jobs
- `src/azureml/` now owns the Azure ML SDK boundary used by the fixed training and release entrypoints
- runtime config resolution for Azure, training defaults, and promotion thresholds now lives in `src/config/runtime.py`
- `run_release.py` is the script-first release handoff for registration and optional deployment
- fixed-train and validation manifests are release-lineage evidence; `run_release.py` validates CLI-declared configs against those manifests before registering or deploying
- when the fixed-train job came from an exported HPO winner config, release lineage prefers the manifest's `canonical_train_config` over the generic AML runtime filename

## 5. Deployment Path

Current live execution surfaces:

- `run_release.py`
- `notebooks/deploy_online_endpoint.ipynb`

Current purpose:

- register a promoted MLflow model bundle
- deploy to an Azure ML managed online endpoint
- invoke the endpoint with `sample-data.json`
- capture release metadata that feeds monitoring follow-up
- configure repo-owned online scoring intent for payload validation and scoring control, while using release evidence and Azure logs to verify whether that scorer actually executed in cloud
- when repo-owned capture is enabled, the canonical sink is an Azure-accessible JSONL path rather than a container-local file path

Current release expectation:

## 6. Monitoring Path

Current live execution surfaces:

- `run_monitor.py`
- `run_inference_capture.py`
- `run_release_monitor_smoke.py`
- `configs/monitor.yaml`
- `configs/inference_capture.yaml`
- `configs/inference_capture_blob.yaml`
- `aml/components/monitor.yaml`

Current purpose:

- evaluate bounded monitoring readiness from `release_record.json`
- invoke the endpoint through a caller-side wrapper when production-evidence capture is required
- support healthy `release_evidence_only` handoff without pretending unsupported runtime hooks exist
- optionally enrich monitoring with externally retrievable sampled inference capture
- emit operator-readable monitoring artifacts
- optionally bundle release reuse, caller capture, exact retrieval, and monitor evaluation into one repo-owned smoke proof folder

Current outputs:

- `monitor_summary.json`
- `monitor_report.md`
- `monitor_manifest/step_manifest.json`

Current implementation direction:

- `run_monitor.py` is the canonical local monitor evaluator surface
- `src/monitoring/evaluate_release.py` consumes release evidence first and optional capture evidence second
- healthy monitoring can mean `limited_but_healthy` when deployment and canary evidence are good but capture evidence is not yet present; the now-proven default runtime contract is `repo_owned_scoring_proven`, while `generated_runtime_still_in_control` remains a fallback classification for older runs or regressions
- `capture_backed` is only reachable when capture evidence is truly retrievable, schema-consistent, and within bounded prediction-balance thresholds
- `monitor_summary.json` now states `capture_evidence_source` explicitly so `capture_backed` can be read as caller-side proof rather than endpoint-side collector proof
- `run_monitor.py` now also writes `retraining_decision.json`, which persists the bounded monitor-stage policy output without auto-starting downstream train, HPO, promotion, or release work
- `run_monitor.py --capture-path <retrieved-capture-dir>` is the repo-owned bridge from externally retrieved production evidence back into the monitor evaluator
- `run_inference_capture.py --config configs/inference_capture_blob.yaml ...` is the first repo-owned capture writer for managed-online endpoint traffic; direct endpoint calls bypass this evidence path unless routed through the wrapper or a future gateway
- `run_release_monitor_smoke.py` is the thin convenience wrapper when you want one operator-facing proof bundle instead of running release, capture, retrieval, and monitor as separate commands
- `run_monitor_handoff.py` is the repeatable monitoring-first wrapper when a saved `release_record.json` already exists and you want fresh bounded capture plus monitor evaluation without redeploying
- `run_monitor_handoff.py` remains the command operators invoke, while implementation ownership now lives in `tools/monitoring/run_monitor_handoff.py`
- the repeatable handoff path is now cloud-proven in both truthful fallback and healthy Blob-backed modes: without capture credentials it lands in `release_evidence_only_ready`, and with Blob credentials plus exact retrieval it reaches `capture_backed_monitoring_ready`
- `run_release_failure_probe.py` is the dedicated scorer-failure proof path; it intentionally bypasses local payload validation for one known-bad request so Azure scorer errors can be inspected without weakening the default happy-path guardrails
- `run_release_failure_probe.py` remains the command surface, while implementation ownership now lives in `tools/release/run_release_failure_probe.py`
- `aml/components/monitor.yaml` is a thin wrapper around the same CLI contract; it should stay thin and must not fork monitor logic from `run_monitor.py`
- the monitor path is intentionally separate from unsupported Azure collector-binding assumptions
- `run_retraining_candidate.py` is the next thin bridge after monitor policy: it consumes a positive retraining decision plus explicit `current_data` / `reference_data`, writes a frozen candidate manifest, and can optionally reuse `validate_data.py`

- release refuses rejected promotion decisions
- release fails before model registration when source job manifests disagree with CLI-declared data or train configs, unless `--allow-lineage-mismatch` is explicitly passed
- release reuses an existing approved model version by default when the source job, effective data/train lineage, validated lineage status, and candidate model match
- `run_release.py --force-reregister` is the explicit escape hatch when you intentionally want a new approved model version for the same matching release lineage
- registration stamps approved-model metrics back into Azure ML model tags
- registration stamps lightweight data/config/component lineage into Azure ML model tags using validated effective lineage when source manifests are available
- registration uses Azure ML job-output folder-root URIs such as `azureml://jobs/<job>/outputs/mlflow_model/paths/`
- deployment consumes a promoted registered model or approved MLflow bundle with an Azure ML deployable conda environment
- fixed-train MLflow bundles normalize Python to a conda-resolvable minor version and keep the serving environment compatible with the current managed-online deployment path
- smoke-test evidence is recorded immediately after deployment with the configured `deployment.smoke_payload`, which defaults to the checked-in `sample-data.json` payload shape
- deployment finalization now follows real Azure deployment state rather than trusting only the first synchronous SDK wait; delayed Azure `Succeeded`, terminal `Failed`/`Canceled`, and explicit finalization-timeout outcomes are classified separately
- local smoke-payload validation failure inside the deploy path is also classified as a structured deployment failure, so release records preserve any deployment state or traffic-update progress that had already been reached before endpoint invoke was skipped
- release records capture status, model resolution, endpoint name, deployment name, model version, validation timestamp, declared/source/effective lineage, smoke payload, smoke response, deployment state, recovery/timed-out flags, inference-capture metadata, deployment failure details, and a derived `monitoring_handoff` summary when deployment fails after model resolution or succeeds
- the currently supported healthy monitoring states are `ready_for_basic_monitoring_handoff` and `ready_for_repo_owned_inference_capture_handoff`; Azure collector-backed monitoring remains unsupported for the current managed-online path, and the 2026-04-14 fresh-deploy proof run classified live scoring as `repo_owned_scoring_proven`
- repo-owned inference capture is only `healthy` when the external sink is actually retrievable after deploy smoke and the runtime contract does not contradict repo-owned scorer expectations; a merely configured sink must not upgrade monitoring readiness
- release-record construction now lives with the rest of the release workflow helpers rather than in a separate tiny module

Quota-safe smoke deployment pattern:

- when an existing smoke deployment is already serving and Azure quota is constrained, set `AML_ONLINE_DEPLOYMENT_NAME=churn-smoke-silver` before `run_release.py --deploy` so the release updates the existing endpoint path instead of creating another `Standard_D2as_v4` deployment
- inspect `release_record.json` under the selected `--download-dir` for the full local validation record, but do not commit `release_artifacts_*` folders
- see `docs/features/online-endpoint-deployment/release-validation.md` for the focused release validation checklist

## 6. Data Validation And Drift

This stage now exists as a first-class implementation stage in the repo.

Target responsibility:

- compare incoming datasets to a training baseline
- produce a batch validation artifact
- decide whether training should proceed, warn, or fail

Recommended tool placement:

- Evidently for batch drift and validation reports
- Azure ML for orchestration and artifact handling

Current implementation surfaces:

- `src/validate_data.py`
- `aml/components/validate_data.yaml`

Current debugging artifacts:

- `validation_manifest/step_manifest.json`
  - named AML output for resolved validation config, reference/current inputs, summary metrics, and report paths
- `data_prep_manifest/step_manifest.json`
  - named AML output for resolved preprocessing settings, input discovery, split counts, and artifact bundle paths
- `train_manifest/step_manifest.json`
  - named AML output for resolved training config, candidate models, best metrics, and model artifact paths
- `promotion_manifest/step_manifest.json`
  - named AML output for promotion thresholds, candidate-versus-baseline comparison, and canonical promotion manifest pathing
  - named AML output for promotion thresholds, candidate/baseline scores, decision status, and rejection reasons

The same JSON is still mirrored into nested output-folder files such as
`validation_report/step_manifest.json`, `processed_data/step_manifest.json`,
`model_output/step_manifest.json`, and `promotion_decision/step_manifest.json`
for `az ml job download --all` fallback workflows.

For HPO, the declared named manifest outputs are now the canonical paths. If a
training or HPO trial also writes a nested fallback manifest, treat that as a
download compatibility surface rather than the primary path.

The training artifact vocabulary is now intentionally shared across fixed
training and HPO:

- `model_output`
- `mlflow_model`
- `candidate_metrics`
- `train_manifest`

HPO adds:

- `hpo_manifest`
- parent-level winner outputs
- `winner_train_config/train_config.yaml` for fixed-train handoff without
  manual hyperparameter copy/paste
- when that winner config is later exported into a repo-owned config file, the
  fixed-train manifest preserves the repo-facing alias through
  `canonical_train_config`

The same manifest shape is also emitted by `src/smoke_preflight.py`, so local
preflight and AML jobs now share a clearer debugging surface.

## 7. Monitoring

This stage now has a documented contract even though most checks remain operator-driven.

Current monitoring split:

- pre-training monitoring through batch validation and drift checks
- post-deployment monitoring through bounded release handoff evidence today, with Azure ML Model Monitor or fuller production collectors still planned rather than fully implemented

Expected post-deployment checks:

- successful scoring call with `sample-data.json`
- endpoint provisioning state and traffic routing in Azure ML Studio
- container logs and request/error metrics in Azure ML or Azure Monitor
- repo-owned inference capture warnings mean monitoring evidence degraded and should be reviewed
- Azure collector-backed monitoring is not a supported contract on the current managed-online repo-owned path; use release evidence and repo-owned capture instead
- a follow-up monitoring plan for data drift, prediction drift, or quality regression

The monitoring surfaces now expose three different status planes and they
should be read separately:

- `release_record.json > monitoring_handoff.status`
  - release-owned summary only
  - `ready_for_basic_monitoring_handoff` means deployment succeeded, traffic updated, and canary scoring passed
  - `ready_for_repo_owned_inference_capture_handoff` means the release path also proved deploy-time repo-owned capture
  - `capture_degraded_after_deploy`, `canary_failed_after_deploy`, `deploy_incomplete_or_timed_out`, and `deploy_failed_before_handoff` remain release-stage classifications
- `monitor_summary.json > monitor_status`
  - evaluator result only
  - `limited_but_healthy` means release evidence is healthy even without retrieved caller-side capture
  - `capture_backed` means caller-side capture was retrieved and passed bounded checks
  - `degraded` or `blocked` means the evaluator found a real monitoring problem
- wrapper summaries such as `handoff_summary.json > handoff.status` or `release_monitor_summary.json > handoff.status`
  - operator-facing reconciliation only
  - `release_evidence_only_ready` means the wrapper finished with trustworthy release evidence but not retrieved caller capture
  - `capture_backed_monitoring_ready` means the wrapper finished with retrieved caller-side evidence and a healthy evaluator result
  - `blocked` or `needs_attention` means orchestration or evidence gathering stopped short of a healthy handoff

## 8. Shared Core

The sweep path and fixed training path should remain separate at the orchestration level, but they should share:

- data preparation component
- trial training logic
- configuration loading rules
- MLflow and output conventions
- evaluation utilities

This keeps the system easier to explain and safer to operate.

The current refactor direction also favors thicker domain owners over generic barrels:

- `src/azureml/` owns Azure ML SDK client, input, registry, and deployment adapter boundaries shared by the entrypoints
- `src/config/` owns runtime config composition
- `src/models/` owns model-factory behavior
- `src/release/` owns pure release workflow semantics such as release-record construction, baseline interpretation, and promotion gating
- `src/utils/` is limited to low-level reusable helpers rather than domain orchestration

For concrete asset ownership examples, see `docs/asset_management_examples.md`.

## 9. Current Naming Rule

`run_hpo_pipeline.py` is the canonical end-to-end HPO submission entrypoint for
this repo.

`run_hpo.py` remains the lower-level direct rerun surface when processed data
already exists and you want to submit or repeat sweeps without the validation +
data-prep wrapper.

`notebooks/main.ipynb` remains the canonical HPO notebook review surface.
Older references to `notebooks/hpo_manual_trials.ipynb` should be treated as
documentation drift unless and until the file is intentionally renamed back to
that path.
