---
doc_id: project-setup-guide
doc_type: operator-guide
explains:
  features:
    - workspace-bootstrap
  stages:
    - data_validate
    - data_prep
    - fixed_train
    - model_sweep
    - online_deploy
  configs:
    - configs/assets.yaml
    - configs/data.yaml
    - configs/data_smoke.yaml
    - configs/train.yaml
---

# Project Setup Guide

This guide covers the setup for both local development and Azure Machine Learning.

For the current system architecture and lifecycle, use `README.md` and `docs/pipeline_guide.md` as the primary sources of truth. This document is setup-focused and should not become a second full workflow guide.

Repo layout note:

- canonical lifecycle entrypoints remain at the repo root:
  `run_pipeline.py`, `run_hpo_pipeline.py`, `run_hpo.py`, `run_monitor.py`, `run_release.py`, and `run_retraining_loop.py`
- secondary operator helpers live under `tools/`, even when a compatibility command still exists at the root
- utility modules such as `hpo_utils.py` are not part of the operator command surface
- setup scripts stay in `setup/` and should not be mixed with the operator-helper lane

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/customer-churn-prediction-azureml.git
cd customer-churn-prediction-azureml
```

### 2. Create the Python 3.9 Environment and Install Dependencies

> Detailed OS-specific instructions live in `docs/python_setup.md`. The quick version is below.

```bash
# Create & activate a Python 3.9 virtual environment
python3.9 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install tooling and sync pinned requirements
pip install --upgrade pip pip-tools
pip install -r requirements.txt
pip install -r dev-requirements.txt  # optional for local development tooling
```

Only run `pip-compile` if you change the `.in` files; see `docs/dependencies.md` for the full workflow.

### 3. Running the EDA Notebook

The EDA notebook is configured to run locally without Azure dependencies.

- **Navigate to `notebooks/eda.ipynb`** in your IDE.
- **Select a Python kernel** and run the cells.
- The notebook will automatically use the local `data/churn.csv` file.

## Azure ML Setup

### 1. Create `config.env`

Copy the example configuration file and fill in your Azure service principal or user credentials.

```bash
cp config.env.example config.env
```

Edit `config.env` with your Azure details:

```bash
AZURE_SUBSCRIPTION_ID="your-subscription-id"
AZURE_TENANT_ID="your-tenant-id"
AZURE_RESOURCE_GROUP="rg-churn-ml-project"
AZURE_WORKSPACE_NAME="churn-ml-workspace"
AZURE_PIPELINE_COMPUTE="cpu-cluster"
```

**Important**: The data asset must be registered as `uri_folder` type (directory containing CSV file(s)). The `data_prep` component will automatically load all CSV files in the folder.

The template also includes optional entries you can customize:

- `AZURE_LOCATION`: Azure region for the workspace (e.g., `southeastasia`)
- `AZURE_COMPUTE_CLUSTER_NAME`: Name used when provisioning compute cluster
- `AZURE_PIPELINE_COMPUTE`: Compute target used when `run_pipeline.py` submits the AML pipeline
- `COMPUTE_CLUSTER_SIZE`: VM size for compute cluster
- `AZURE_COMPUTE_INSTANCE_NAME`: Optional notebook compute instance name created by the setup scripts
- `COMPUTE_INSTANCE_SIZE`: VM size for the optional compute instance
- `CREATE_COMPUTE_INSTANCE`: Set to `false` to skip compute instance creation during setup
- Project asset/model/deployment defaults live in `configs/assets.yaml`
- `DATA_*`, `MODEL_NAME`, `ENDPOINT_NAME`, `DEPLOYMENT_NAME`, and `AML_ONLINE_*`: optional local overrides only

**Note**: `run_pipeline.py` and the setup scripts automatically load `config.env`, so you don't need to source it manually.

The AML runtime image is intentionally not duplicated in `config.env`. Keep the environment identity in `aml/environments/environment.yml`, but register the actual image through `setup/register_environment.py`, which resolves the image URI from `configs/assets.yaml` plus `AZURE_ACR_NAME` in `config.env`.

### 2. Authenticate with Azure

Log in to the Azure CLI:

```bash
az login
```

### 3. Build, Push, and Register the Azure ML Environment

The AML components reference `azureml:bank-churn-env:1`, so the workspace must have that environment asset before `run_pipeline.py` can submit jobs.

1. Build and push the Docker image:

```bash
az acr login --name churnmlarc
docker build -t churnmlarc.azurecr.io/bank-churn:1 .
docker push churnmlarc.azurecr.io/bank-churn:1
```

2. Register the AML environment asset:

```bash
python setup/register_environment.py
```

The image URI is resolved from `configs/assets.yaml` plus `AZURE_ACR_NAME` in `config.env`, so no manual edit to `aml/environments/environment.yml` is required during setup.

### 4. Create the Azure ML Data Asset

The churn pipeline expects the dataset to be registered as a `uri_folder` so the `data_prep` component can automatically load all CSV files. Use the provided helper script to register. It reads canonical names/versions from `configs/assets.yaml`, with optional env overrides:

```bash
python setup/create_data_asset.py \
  --data-path data/churn.csv \
  --name churn-data \
  --version 1
```

If you prefer the CLI, the equivalent command is:

```bash
az ml data create \
  --name churn-data \
  --version 1 \
  --path data/ \
  --type uri_folder \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME
```

Keep the `name`/`version` aligned with `configs/assets.yaml`; `run_pipeline.py` and the notebooks read these values via `get_data_asset_config()`.

### 5. Optional: Register the Smoke Data Asset

For low-cost wiring checks, register the positive smoke fixture as a separate asset instead of replacing the production data asset:

```bash
python setup/create_data_asset.py \
  --data-path data/smoke/positive \
  --name churn-data-smoke \
  --version 2 \
  --description "Small churn smoke fixture for pipeline wiring checks"
```

The smoke data is intentionally small. Register the positive-path smoke asset from `data/smoke/positive` only. The negative fixture in `data/smoke/negative` is for validation-gate checks and should not be mixed into training assets.

For smoke pipeline submissions, `run_pipeline.py --data-config configs/data_smoke.yaml --train-config configs/train_smoke.yaml`
selects the registered smoke data asset from `configs/assets.yaml`. Smoke configs
intentionally ignore ambient `DATA_*` asset variables so a leftover shell session
cannot silently change the smoke fixture being tested.

For a less-biased smoke check, register a second asset from `data/smoke/eval` and align it with the
`smoke_eval` entry in `configs/assets.yaml`. Then submit:

```bash
python setup/create_data_asset.py \
  --data-path data/smoke/eval \
  --name churn-data-smoke-eval \
  --version 1 \
  --description "Overlapping churn smoke fixture for gated pipeline checks"
```

`run_pipeline.py --data-config configs/data_smoke_eval.yaml --train-config configs/train_smoke.yaml`
uses that alternate asset automatically as current data, uses the base `churn-data-smoke`
asset as the validation reference, and gates `data_prep` behind `validate_data`. The eval
profile also enables `fail_on_drift: true`, so dataset drift above the configured share
threshold fails the validation step.

### 6. First Submission Commands

Once the workspace, environment, and data assets exist, use one of these first
commands:

**Fixed training smoke**

```bash
python run_pipeline.py \
  --data-config configs/data_smoke_eval.yaml \
  --train-config configs/train_smoke.yaml
```

**End-to-end HPO smoke**

```bash
python run_hpo_pipeline.py \
  --data-config configs/data_smoke_eval.yaml \
  --hpo-config configs/hpo_smoke.yaml
```

**Local smoke preflight**

```bash
python src/smoke_preflight.py --clean
```

## Where To Go Next

- Use `docs/pipeline_guide.md` for workflow behavior, manifests, HPO hierarchy,
  and artifact surfaces.
- Use `docs/python_setup.md` for deeper Python/runtime setup help.
- Use `inspect_hpo_run.py` after downloading an HPO parent job when you want a
  repo-owned summary instead of manually opening each child job.
  The command surface stays at the root during migration, while implementation
  ownership lives in `tools/hpo/inspect_hpo_run.py`.
