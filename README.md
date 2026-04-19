# Customer Churn Prediction on Azure ML

> A portfolio-oriented Azure ML MLOps system for tabular churn prediction, with explicit validation, HPO, fixed training, release, deployment, and monitoring handoff workflows.

## Who Uses It

This repo is built for a person or team that wants to run a real cloud ML lifecycle instead of stopping at notebook-only experimentation.

Typical users are:

- an ML engineer building and maintaining the churn pipeline
- an operator submitting Azure ML training, HPO, release, and monitoring workflows
- a reviewer who needs traceable artifacts, bounded diagnostics, and repeatable cloud proofs

## Problem

A churn model pipeline becomes hard to trust when its lifecycle is split across notebooks, ad hoc scripts, and manually interpreted cloud runs.

Typical pain points are:

- incoming datasets can drift or degrade before training starts
- HPO and fixed retraining can get mixed together operationally
- model promotion and deployment can become side effects instead of explicit gates
- managed online scoring is difficult to reason about without clear release and monitoring evidence
- Azure ML runs are noisy unless artifacts, manifests, and wrapper surfaces are intentionally designed

## Solution

This repo turns that into a staged Azure ML workflow:

1. validate incoming data and drift against a reference asset
2. prepare training-ready features through a reusable data-prep stage
3. choose either the optimization path or the fixed training path
4. evaluate and promote the resulting candidate model
5. register and optionally deploy the approved model
6. hand off bounded release evidence into monitoring
7. optionally enrich monitoring with retrievable caller-side capture evidence

The result is a repo that is easier to operate, inspect, and explain as a complete MLOps system rather than a loose collection of training scripts.

The repo root now stays intentionally smaller:

- canonical lifecycle entrypoints remain at the root:
  `run_pipeline.py`, `run_hpo_pipeline.py`, `run_hpo.py`, `run_monitor.py`, `run_release.py`, and `run_retraining_loop.py`
- secondary operator helpers live under `tools/`
- old root helper filenames remain as compatibility wrappers during the migration window
- non-entrypoint utilities such as `hpo_utils.py` are a separate code-organization question, not part of the root command surface
- new local proof bundles should prefer a bounded local artifacts area such as `.tmp-tests/` instead of accumulating at repo root by default

## Key Pipeline Stages

- `data_validate`
  - checks schema, missingness, row-count sanity, and bounded drift before training proceeds
- `data_prep`
  - prepares processed train/test artifacts and stage manifests for downstream training
- `model_sweep`
  - runs Azure ML HPO over per-model sweep branches and emits stable parent-level winner outputs
- `fixed_train`
  - trains the selected fixed configuration and produces candidate metrics plus deployable MLflow output
- `model_promote`
  - compares candidate metrics against the approved baseline and writes the promotion decision
- `online_deploy`
  - registers the approved model, optionally deploys it, and records bounded release evidence
- `monitor`
  - evaluates release evidence first and optional caller-side capture evidence second

## Major Workflow Features

- Validation-gated fixed training through `run_pipeline.py`
- End-to-end Azure ML HPO through `run_hpo_pipeline.py`
- Direct HPO reruns against prepared data through `run_hpo.py`
- Deterministic HPO family tie-break policy recorded in `hpo_summary.json` and winner artifacts
- Promotion-gated release and optional deployment through `run_release.py`
- Helper-based HPO inspection, monitoring handoff, release-monitor proofs, and retraining bridges implemented under `tools/`
- Monitor-triggered retraining recommendation through `run_monitor.py` / `retraining_decision.json`
- Bounded phase-1 retraining loop orchestration through `run_retraining_loop.py`
- Optional bounded retraining-loop continuation into release and post-release monitoring after promotion-facing evidence is proven promotable
- Dedicated negative cloud probe for scorer failure semantics through `run_release_failure_probe.py`

## Smoke Surface

The repo keeps low-cost smoke inputs explicit and opt-in:

- `data/smoke/positive/churn_smoke.csv`
  - positive-path fixture for cheap wiring checks
- `data/smoke/eval/churn_smoke_eval.csv`
  - alternate smoke fixture for less-trivial evaluation behavior
- `data/smoke/negative/churn_validation_edge.csv`
  - negative-path validation fixture
- `configs/data_smoke.yaml`, `configs/data_smoke_eval.yaml`, `configs/train_smoke.yaml`, `configs/hpo_smoke.yaml`
  - smoke profiles for validation, fixed training, and HPO
- `sample-data.json`
  - managed online endpoint smoke payload

Smoke runs are for orchestration, guardrail, and artifact validation, not for claiming production model quality.

## Engineering Highlights

The most important work in this repo is not only “train a churn classifier.” It is the surrounding lifecycle reliability and operator truth:

- **Stage-aware orchestration**
  - validation, prep, HPO, fixed train, promotion, deploy, and monitor are kept as explicit stages with stable artifacts
- **Artifact-first debugging**
  - manifests, summaries, and named outputs are treated as the primary inspection surface instead of raw log-diving
- **Separation of optimization and production retraining**
  - HPO and fixed training share core logic but stay distinct operationally
- **Release truthfulness**
  - `release_record.json` captures deployment state, canary evidence, lineage, and monitoring handoff without pretending unsupported telemetry exists
- **Repo-owned scoring proof**
  - current fresh-cloud evidence proves the repo-owned scorer path through `repo_owned_scoring_proven`
- **Caller-side monitoring evidence**
  - retrievable caller-side capture can upgrade monitoring from `limited_but_healthy` to `capture_backed` without relying on Azure collector binding
- **Truthful retraining policy**
  - monitoring can now recommend “no retraining signal,” “retraining candidate,” or “investigate before retraining” without skipping validation, promotion, or release gates
- **HPO stays exploratory until fixed-train continuation**
  - completed HPO now feeds one validated winner-config continuation back into fixed training instead of pretending HPO outputs are release-ready on their own

## Architecture

```text
Azure ML data asset
  |
  v
Validation + drift gate
  |
  v
Data prep
  |
  +--> HPO path
  |      run_hpo_pipeline.py / run_hpo.py
  |      -> per-family sweeps
  |      -> HPO summary + winner outputs
  |      -> export_hpo_winner_config.py
  |
  +--> Fixed train path
         run_pipeline.py
         -> train_manifest + candidate_metrics + MLflow model
         -> promote_model
         -> promotion_decision
         -> run_release.py
         -> managed online deployment
         -> run_monitor.py / run_monitor_handoff.py
```

Operational invariants:

- workflow configuration stays in `configs/`
- generated feature and stage contracts document the current lifecycle surfaces
- publication and other repo-governance config stay separate from the public product surface
- release and monitoring truth are artifact-driven, not log-guess-driven
- wrapper scripts stay thin and compose authoritative child surfaces instead of duplicating domain logic
- smoke evidence is bounded and explicit
- helper implementations now live under `tools/`, while the root keeps the canonical lifecycle entrypoints and stable wrapper commands

## Main Components

### Azure and orchestration

- `run_pipeline.py`
  - canonical fixed-training pipeline submission surface
- `run_hpo_pipeline.py`
  - canonical end-to-end HPO pipeline submission surface
- `run_hpo.py`
  - lower-level direct sweep surface when prepared data already exists
- `run_release.py`
  - authoritative release and optional deployment surface
- `run_monitor.py`
  - authoritative monitor evaluator
- `tools/`
  - implementation home for secondary operator helpers such as HPO inspection, monitoring handoff, release-monitor smoke proofs, and retraining bridges
  - root helper filenames remain as compatibility wrappers during the refactor window
- `hpo_utils.py`
  - shared HPO utility module kept separate from the operator entrypoint taxonomy

### Domain code

- `src/azureml/`
  - Azure ML SDK adapters for jobs, registry, and deployment
- `src/config/`
  - runtime config and asset-manifest loading
- `src/data/`
  - data-prep logic and config handling
- `src/training/`
  - training orchestration and MLflow-facing behavior
- `src/promotion/`
  - candidate-vs-baseline promotion logic
- `src/release/`
  - release workflow and release-record construction
- `src/monitoring/`
  - monitoring evaluator, handoff helpers, and exact capture retrieval

## Docs

- [pipeline_guide.md](docs/pipeline_guide.md)
  Current workflow model, stage boundaries, artifact surfaces, operator guidance, and the active root-versus-tools helper policy.
- [setup_guide.md](docs/setup_guide.md)
  Local Python setup, Azure setup, asset registration, and first-run commands.
- [model-training-pipeline.yaml](docs/features/model-training-pipeline/model-training-pipeline.yaml)
  Generated current contract for the fixed-training feature.
- [release-validation.md](docs/features/online-endpoint-deployment/release-validation.md)
  Focused release, deploy, canary, and monitoring-handoff validation guidance.
- [online-endpoint-deployment.yaml](docs/features/online-endpoint-deployment/online-endpoint-deployment.yaml)
  Structured contract for the release and deployment feature.
- [release-monitoring-evaluator.yaml](docs/features/release-monitoring-evaluator/release-monitoring-evaluator.yaml)
  Structured contract for the monitoring evaluator feature.

## Source Layout

```text
aml/                  Azure ML component and environment definitions
configs/              asset, data, training, HPO, release, and monitor config
data/                 local data, smoke fixtures, and sample payloads
docs/                 workflow docs, feature contracts, and stage contracts
notebooks/            EDA and HPO review notebooks
tools/                helper implementations for HPO inspection, monitoring, release proofs, and retraining bridges
setup/                workspace, environment, and asset setup scripts
src/                  Azure ML adapters, data, training, promotion, release, monitoring
tests/                automated verification
```

## Getting Started

For setup and first runs, start with:

- [setup_guide.md](docs/setup_guide.md)

Typical first commands:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py `
  --data-config configs\data_smoke.yaml `
  --train-config configs\train_smoke.yaml
```

For the end-to-end HPO smoke path:

```powershell
.\.venv\Scripts\python.exe run_hpo_pipeline.py `
  --data-config configs\data_smoke_eval.yaml `
  --hpo-config configs\hpo_smoke.yaml
```
