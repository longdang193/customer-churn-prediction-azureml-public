---
doc_id: asset-management-examples
doc_type: operator-guide
explains:
  features:
    - workspace-bootstrap
    - churn-data-preparation
    - model-training-pipeline
    - online-endpoint-deployment
  stages:
    - data_validate
    - data_prep
    - fixed_train
    - online_deploy
  configs:
    - configs/assets.yaml
    - configs/data.yaml
    - configs/data_smoke.yaml
  components:
    - aml/components/data_prep.yaml
---

# Asset Management Examples

This repo should keep asset management lightweight: register durable Azure ML assets, keep intermediate outputs as job artifacts, and use tags/metadata for lineage.

## Recommended Asset Ownership

| Stage | Durable asset? | Example | Why |
| --- | --- | --- | --- |
| Workspace bootstrap | Yes | AML workspace, compute, environment | Shared runtime prerequisites. |
| Raw data | Yes | `churn-data:1` | Stable input, versioned and auditable. |
| Smoke data | Yes | `churn-data-smoke:1` | Low-cost e2e wiring checks without changing production defaults. |
| Validation report | No | `validation_report` job output | Tied to one run; keep as artifact unless promoted to evidence. |
| Feature engineering | Yes, as component/config | `data_prep` component + `configs/data.yaml` | Reusable pipeline step without needing Feature Store yet. |
| Processed data | Usually no | `processed_data` job output | Derived artifact; register only if reused across many jobs. |
| HPO trial models | Usually no | Trial job artifacts | Registering every trial creates registry noise. |
| Selected model | Yes, after promotion | `churn-prediction-model:<version>` | Release/rollback/audit boundary. |
| Endpoint payload | Yes, in Git | `sample-data.json` | Stable smoke-test contract for deployment checks. |

## Example 1: Register Durable Data Assets

Production data should be registered as a versioned `uri_folder` asset:

```bash
python setup/create_data_asset.py \
  --data-path data \
  --name churn-data \
  --version 1 \
  --description "Production churn training data"
```

Smoke data should use a separate asset name:

```bash
python setup/create_data_asset.py \
  --data-path data/smoke/positive \
  --name churn-data-smoke \
  --version 1 \
  --description "Small churn smoke fixture for pipeline wiring checks"
```

Then point `config.env` at the right asset for the run:

```dotenv
DATA_ASSET_FULL="churn-data"
DATA_VERSION="1"
DATA_REFERENCE_ASSET_FULL="churn-data"
DATA_REFERENCE_VERSION="1"
```

For a smoke-only run, use a local throwaway `config.env` or temporary environment overrides:

```dotenv
DATA_ASSET_FULL="churn-data-smoke"
DATA_VERSION="1"
DATA_REFERENCE_ASSET_FULL="churn-data-smoke"
DATA_REFERENCE_VERSION="1"
```

## Example 2: Treat Feature Engineering As A Component

For this repo, feature engineering is the `data_prep` component plus its config and code:

```text
aml/components/data_prep.yaml
src/data_prep.py
src/data/*
configs/data.yaml
configs/data_smoke.yaml
```

That is enough for now. Registering a full Azure ML Managed Feature Store would add overhead without much benefit until features are reused across multiple models or need online/offline consistency.

If the component stabilizes, publish it as a versioned AML component:

```bash
az ml component create \
  --file aml/components/data_prep.yaml \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --workspace-name "$AZURE_WORKSPACE_NAME"
```

Use semantic component versions when the interface changes:

```yaml
name: data_prep
version: 2
type: command
```

## Example 3: Keep Processed Data As A Job Output

`processed_data` is derived from raw data, code, config, and component version:

```text
churn-data:1
  + configs/data.yaml
  + aml/components/data_prep.yaml
  + src/data_prep.py
  -> processed_data job output
```

Keep it as a job output by default:

```yaml
outputs:
  processed_data:
    type: uri_folder
```

Only register processed data as a durable AML data asset if several downstream jobs need to reuse the exact same processed snapshot.

## Example 4: Register Only Promoted Models

HPO can create many trial models. Keep trial outputs as artifacts, then register only the selected/promoted model:

```text
HPO trials -> metrics/artifacts -> notebook review -> configs/train.yaml
fixed train -> promotion_decision -> run_release.py -> AML model registry
```

`run_release.py` already follows this pattern by registering the promoted MLflow model output rather than every training artifact.

Recommended model tags:

```json
{
  "approval_status": "approved",
  "source_job_name": "azureml-job-name",
  "data_asset": "churn-data",
  "data_version": "1",
  "reference_data_asset": "churn-data",
  "reference_data_version": "1",
  "train_config": "configs/train.yaml",
  "data_prep_component": "data_prep:1",
  "train_component": "train_model:1",
  "primary_metric": "f1",
  "f1": "0.81",
  "roc_auc": "0.88"
}
```

These tags make rollback and audit easier without a separate spreadsheet.

## Example 5: Use The Asset Manifest

The repo keeps canonical names in `configs/assets.yaml` so pipeline jobs and registered models can carry useful lineage tags without replacing `config.env`.

The current manifest shape is:

```yaml
data_assets:
  production:
    name: churn-data
    version: "1"
  reference:
    name: churn-data
    version: "1"
  smoke:
    name: churn-data-smoke
    version: "1"

components:
  validate_data:
    name: validate_data
    version: "1"
  data_prep:
    name: data_prep
    version: "1"
  train:
    name: train_model
    version: "1"
  promote:
    name: promote_model
    version: "1"

environment:
  name: bank-churn-env
  version: "1"

model:
  name: churn-prediction-model

deployment:
  endpoint_name: churn-endpoint
  deployment_name: churn-deployment
  smoke_payload: sample-data.json
```

Use this manifest for asset ownership and lineage metadata. Keep secrets, active workspace coordinates, and operator overrides in `config.env`.

Example lineage tags produced from the manifest plus runtime config:

```json
{
  "data_asset": "churn-data",
  "data_version": "1",
  "reference_data_asset": "churn-data",
  "reference_data_version": "1",
  "data_config": "configs/data.yaml",
  "train_config": "configs/train.yaml",
  "validate_component": "validate_data:1",
  "data_prep_component": "data_prep:1",
  "train_component": "train_model:1",
  "promote_component": "promote_model:1",
  "environment": "bank-churn-env:1"
}
```

## When To Use Managed Feature Store

Defer Managed Feature Store until at least one of these is true:

- multiple models need the same engineered features
- batch and online inference need the same feature definitions
- feature materialization windows become important
- feature ownership needs access control and lineage beyond this repo
- features become business entities rather than simple preprocessing steps

Until then, the efficient path is: versioned data assets, versioned components/config, job outputs for derived artifacts, and promoted model registration.
