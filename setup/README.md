# Setup Scripts and Common Commands

This directory contains setup scripts for Azure ML workspace and compute resources. If you just need the short version, start here:

1. `cp config.env.example config.env` and fill in subscription/workspace names (see [Configuration Setup](#configuration-setup)).
2. `az login` (or ensure your compute instance already has the right identity).
3. Run `./setup/setup.sh` (or `.\setup\setup.ps1`) to provision the resource group, workspace, ACR, compute cluster, and optional compute instance.
4. Build/push the runtime image, then register the AML environment with `python setup/register_environment.py`.
5. Register your dataset: `python setup/create_data_asset.py --data-path data`.
6. Verify resources in Azure ML Studio (`https://ml.azure.com`).

The remainder of this document provides the detailed commands and optional variations for those steps.

Repo-surface note:

- canonical lifecycle entrypoints stay at the repo root
- secondary operator helpers belong under `tools/`, even if a root compatibility command still exists
- setup scripts remain a separate lane from both lifecycle entrypoints and helper wrappers

## Prerequisites

- Azure CLI installed: [Install Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli)
- Authenticated with Azure: `az login`
- Appropriate permissions to create resources in the subscription

## Configuration Setup

Before running the setup scripts, you need to create a `config.env` file with your Azure credentials. Copy `config.env.example` to `config.env` and fill in the values:

```bash
cp config.env.example config.env
```

The setup scripts automatically load `config.env` from the repository root before resolving defaults, so you do not need to manually source it before running `setup/setup.sh` or `setup/setup.ps1`.

### Get Azure Configuration Values

Use these commands to retrieve the values needed for `config.env`:

#### Get Subscription ID

**Linux/Mac:**

```bash
az account show --query id -o tsv
```

**Windows (PowerShell):**

```powershell
az account show --query id -o tsv
```

#### Get Tenant ID

**Linux/Mac:**

```bash
az account show --query tenantId -o tsv
```

**Windows (PowerShell):**

```powershell
az account show --query tenantId -o tsv
```

#### Get Current Subscription Information

**Linux/Mac:**

```bash
az account show --query "{SubscriptionId:id, TenantId:tenantId, Name:name}" -o table
```

**Windows (PowerShell):**

```powershell
az account show --query "{SubscriptionId:id, TenantId:tenantId, Name:name}" -o table
```

#### List Available Locations

**Linux/Mac:**

```bash
az account list-locations --query "[].{Name:name, DisplayName:displayName}" -o table
```

**Windows (PowerShell):**

```powershell
az account list-locations --query "[].{Name:name, DisplayName:displayName}" -o table
```

#### Get Workspace Information (After Creation)

**Linux/Mac:**

```bash
az ml workspace show \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --name "${AZURE_WORKSPACE_NAME:-churn-ml-workspace}" \
    --query "{Name:name, ResourceGroup:resourceGroup, Location:location}" \
    -o table
```

**Windows (PowerShell):**

```powershell
az ml workspace show `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --name $env:AZURE_WORKSPACE_NAME `
    --query "{Name:name, ResourceGroup:resourceGroup, Location:location}" `
    -o table
```

#### Get Storage Account Information

**Linux/Mac:**

```bash
# List storage accounts in resource group
az storage account list \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --query "[].{Name:name, Location:location}" \
    -o table
```

**Windows (PowerShell):**

```powershell
az storage account list `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --query "[].{Name:name, Location:location}" `
    -o table
```

#### Get Azure Container Registry (ACR) Information

**Check if you have an existing ACR:**

**Linux/Mac:**

```bash
# List all ACRs in subscription
az acr list --output table

# List ACRs in specific resource group
az acr list --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" --output table

# Check if specific ACR exists
az acr show --name <acr-name> --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}"
```

**Windows (PowerShell):**

```powershell
# List all ACRs in subscription
az acr list --output table

# List ACRs in specific resource group
az acr list --resource-group $env:AZURE_RESOURCE_GROUP --output table

# Check if specific ACR exists
az acr show --name <acr-name> --resource-group $env:AZURE_RESOURCE_GROUP
```

**Note:** ACR names must be globally unique (3-50 characters, alphanumeric only). If you don't have an ACR, you can create one using the setup script (set `AZURE_ACR_NAME` in `config.env`) or manually (see below).

### Quick Setup Script

You can also use this one-liner to create `config.env` with your current Azure account information:

**Linux/Mac:**

```bash
cat > config.env << EOF
# Azure Subscription and Resource Configuration
AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
AZURE_TENANT_ID="$(az account show --query tenantId -o tsv)"
AZURE_RESOURCE_GROUP="rg-churn-ml-project"
AZURE_LOCATION="southeastasia"

# Azure ML Workspace Configuration
AZURE_WORKSPACE_NAME="churn-ml-workspace"

# Compute Resources
AZURE_COMPUTE_CLUSTER_NAME="cpu-cluster"
AZURE_PIPELINE_COMPUTE="cpu-cluster"
COMPUTE_CLUSTER_SIZE="Standard_DS2_v2"
MIN_NODES="0"
MAX_NODES="2"
AZURE_COMPUTE_INSTANCE_NAME="ci-notebooks"
COMPUTE_INSTANCE_SIZE="Standard_DS2_v2"
CREATE_COMPUTE_INSTANCE="true"

# Optional storage reference (not required by setup scripts)
AZURE_STORAGE_ACCOUNT="yourstorageaccount"
AZURE_STORAGE_CONTAINER="data"

# Azure Container Registry (ACR) Configuration
# ACR name must be globally unique (3-50 characters, alphanumeric only)
AZURE_ACR_NAME="youracrname"
ACR_SKU="Basic"

# Project asset/model/deployment defaults live in configs/assets.yaml.
# Add DATA_*, MODEL_NAME, ENDPOINT_NAME, DEPLOYMENT_NAME, or AML_ONLINE_* here
# only when you intentionally need a local override.
EOF
```

**Windows (PowerShell):**

```powershell
$subscriptionId = az account show --query id -o tsv
$tenantId = az account show --query tenantId -o tsv

@"
# Azure Subscription and Resource Configuration
AZURE_SUBSCRIPTION_ID="$subscriptionId"
AZURE_TENANT_ID="$tenantId"
AZURE_RESOURCE_GROUP="rg-churn-ml-project"
AZURE_LOCATION="southeastasia"

# Azure ML Workspace Configuration
AZURE_WORKSPACE_NAME="churn-ml-workspace"

# Compute Resources
AZURE_COMPUTE_CLUSTER_NAME="cpu-cluster"
AZURE_PIPELINE_COMPUTE="cpu-cluster"
COMPUTE_CLUSTER_SIZE="Standard_DS2_v2"
MIN_NODES="0"
MAX_NODES="2"
AZURE_COMPUTE_INSTANCE_NAME="ci-notebooks"
COMPUTE_INSTANCE_SIZE="Standard_DS2_v2"
CREATE_COMPUTE_INSTANCE="true"

# Optional storage reference (not required by setup scripts)
AZURE_STORAGE_ACCOUNT="yourstorageaccount"
AZURE_STORAGE_CONTAINER="data"

# Azure Container Registry (ACR) Configuration
# ACR name must be globally unique (3-50 characters, alphanumeric only)
AZURE_ACR_NAME="youracrname"
ACR_SKU="Basic"

# Project asset/model/deployment defaults live in configs/assets.yaml.
# Add DATA_*, MODEL_NAME, ENDPOINT_NAME, DEPLOYMENT_NAME, or AML_ONLINE_* here
# only when you intentionally need a local override.
"@ | Out-File -FilePath config.env -Encoding utf8
```

**Note:** After running the quick setup script, you may still need to update:

- `AZURE_STORAGE_ACCOUNT`: Optional reference for manual storage workflows/troubleshooting; the setup scripts do not create or require this named account
- `AZURE_ACR_NAME`: Set to your desired ACR name (must be globally unique) if you want the setup script to create ACR
- Project asset/model/deployment defaults live in `configs/assets.yaml`; keep `config.env` for Azure workspace/resource settings and local overrides only
- Other values can be customized as needed

## Setup Scripts

### Initial Setup

Run the setup script to create Azure ML workspace and compute resources:

**Linux/Mac:**

```bash
./setup/setup.sh
```

**Note:** If you encounter a "Permission denied" error, add execute permissions to the script:

```bash
chmod +x setup/setup.sh
```

**Windows (PowerShell):**

```powershell
.\setup\setup.ps1
```

**What it does:**

- Creates resource group (if it doesn't exist)
- Creates Azure ML workspace
- Creates Azure Container Registry (ACR) if `AZURE_ACR_NAME` is set in `config.env` (**before** compute cluster)
- Creates compute cluster with system-assigned managed identity (AcrPull role automatically granted if ACR exists)
- Creates an optional compute instance for notebook-driven workflows unless `CREATE_COMPUTE_INSTANCE=false`

**Configuration:**

The scripts load `config.env` automatically and then use environment variables (with defaults):

- `AZURE_RESOURCE_GROUP` (default: `rg-churn-ml-project`)
- `AZURE_LOCATION` (default: `southeastasia`)
- `AZURE_WORKSPACE_NAME` (default: `churn-ml-workspace`)
- `AZURE_COMPUTE_CLUSTER_NAME` (default: `cpu-cluster`)
- `AZURE_PIPELINE_COMPUTE` (default: `cpu-cluster`)
- `COMPUTE_CLUSTER_SIZE` (default: `Standard_DS2_v2`)
- `AZURE_COMPUTE_INSTANCE_NAME` (default: `ci-notebooks`)
- `COMPUTE_INSTANCE_SIZE` (default: `Standard_DS2_v2`)
- `CREATE_COMPUTE_INSTANCE` (default: `true`)
- `MIN_NODES` (default: `0`)
- `MAX_NODES` (default: `2`)
- `AZURE_ACR_NAME` (optional: if set, creates ACR during setup)
- `ACR_SKU` (default: `Basic` - options: Basic, Standard, Premium)

### Create a Compute Instance for Notebook-Driven HPO

The repo now supports a script-first HPO submission path via `run_hpo.py`, but `notebooks/main.ipynb` remains the interactive review surface and is still best run from an **Azure ML compute instance** (not the auto-scaling compute cluster). Create one after the workspace is ready:

**Linux/Mac:**

```bash
COMPUTE_INSTANCE_NAME="${AZURE_COMPUTE_INSTANCE_NAME:-ci-notebooks}"

az ml compute create \
  --name "$COMPUTE_INSTANCE_NAME" \
  --type computeinstance \
  --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
  --workspace-name "${AZURE_WORKSPACE_NAME:-churn-ml-workspace}" \
  --size "${COMPUTE_INSTANCE_SIZE:-Standard_DS2_v2}"
```

**Windows (PowerShell):**

```powershell
$computeInstanceName = $env:AZURE_COMPUTE_INSTANCE_NAME
if (-not $computeInstanceName) { $computeInstanceName = "ci-notebooks" }
$computeInstanceSize = $env:COMPUTE_INSTANCE_SIZE
if (-not $computeInstanceSize) { $computeInstanceSize = "Standard_DS2_v2" }

az ml compute create `
  --name $computeInstanceName `
  --type computeinstance `
  --resource-group $env:AZURE_RESOURCE_GROUP `
  --workspace-name $env:AZURE_WORKSPACE_NAME `
  --size $computeInstanceSize
```

After the compute instance reaches the `Succeeded` state, open Azure ML Studio → **Compute → Compute instances**, start it if stopped, and launch Jupyter/VS Code from there to run the notebook. The instance’s managed identity provides seamless authentication for the Azure ML SDK.

### Create Data Asset

After setting up the workspace, create a data asset from your local dataset:

**Linux/Mac:**

```bash
python setup/create_data_asset.py --data-path data
```

**Windows (PowerShell):**

```powershell
python setup/create_data_asset.py --data-path data
```

**Options:**

- `--data-path`: Path to data directory or file (default: `data`)
- `--name`: Data asset name (default: from `configs/assets.yaml`, with optional `DATA_ASSET_FULL` env override)
- `--version`: Data asset version (default: from `configs/assets.yaml`, with optional `DATA_VERSION` env override)
- `--description`: Description of the data asset

**Example:**

```bash
# Use defaults from config.env
python setup/create_data_asset.py

# Specify custom name and version
python setup/create_data_asset.py --name churn-data --version 1

# Use a specific data directory
python setup/create_data_asset.py --data-path data --name churn-data --version 1
```

**Important:** The data asset is registered as `uri_folder` type, which is required for the `data_prep` component. The script automatically uses the parent directory if you specify a file path.

### Create a Smoke Data Asset

Use a separate smoke asset for low-cost end-to-end wiring checks. Do not point production defaults at the smoke asset unless you are intentionally running a smoke pass.

```bash
python setup/create_data_asset.py \
  --data-path data/smoke/positive \
  --name churn-data-smoke \
  --version 1 \
  --description "Small churn smoke fixture for pipeline wiring checks"
```

The smoke asset should use `data/smoke/positive/churn_smoke.csv` for the positive path. `data/smoke/negative/churn_validation_edge.csv` is a negative-path fixture for validation-gate checks and should not be treated as model-quality training data or mixed into the smoke training asset.

Before submitting the Azure ML smoke pipeline, run the local preflight to catch prep, validation, and training drifts early:

```powershell
.\.venv\Scripts\python.exe src\smoke_preflight.py --clean
```

## Azure Container Registry (ACR) Setup

**Important**: For proper ACR authentication, ACR should be created **before** the compute cluster. The setup script follows this order automatically. If you create ACR after compute cluster, you'll need to manually grant AcrPull role to the compute's managed identity.

### Create ACR

If you didn't create ACR during initial setup, you can create it manually:

**Linux/Mac:**
  
```bash
az acr create \
  --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
  --name <youracrname> \
  --sku Basic \
  --location "${AZURE_LOCATION:-southeastasia}"
```

**Windows (PowerShell):**

```powershell
az acr create `
  --resource-group $env:AZURE_RESOURCE_GROUP `
  --name <youracrname> `
  --sku Basic `
  --location $env:AZURE_LOCATION
```

**Important:**

- ACR name must be globally unique (3-50 characters, alphanumeric only)
- After creating ACR, update `AZURE_ACR_NAME` in `config.env`
- After building and pushing the runtime image, rerun `python setup/register_environment.py` so Azure ML resolves the pushed image from central config
- **If compute cluster was created before ACR**: You need to manually grant `AcrPull` to the compute's managed identity before AML jobs can pull the image.
- **If ACR exists before compute cluster**: AcrPull role is automatically granted when compute is created with `--identity-type system_assigned`

### Verify ACR Creation

**Linux/Mac:**

```bash
az acr show \
  --name <youracrname> \
  --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
  --query "{Name:name, LoginServer:loginServer, Sku:sku.name}" \
  -o table
```

**Windows (PowerShell):**

```powershell
az acr show `
  --name <youracrname> `
  --resource-group $env:AZURE_RESOURCE_GROUP `
  --query "{Name:name, LoginServer:loginServer, Sku:sku.name}" `
  -o table
```

### Register the AML Environment

After creating ACR, build and push the Docker image, then register the AML environment from central config:

```powershell
az acr login --name $env:AZURE_ACR_NAME
docker build -t "$env:AZURE_ACR_NAME.azurecr.io/bank-churn:1" .
docker push "$env:AZURE_ACR_NAME.azurecr.io/bank-churn:1"
.\.venv\Scripts\python.exe setup\register_environment.py
```

The image URI is resolved from `configs/assets.yaml` plus `AZURE_ACR_NAME` in `config.env`, so you do not need to hand-edit `aml/environments/environment.yml` before registration.

### Common ACR Commands

**List repositories in ACR:**

**Linux/Mac:**

```bash
az acr repository list --name <youracrname> --output table
```

**Windows (PowerShell):**

```powershell
az acr repository list --name <youracrname> --output table
```

**List tags for a repository:**

**Linux/Mac:**

```bash
az acr repository show-tags --name <youracrname> --repository bank-churn --output table
```

**Windows (PowerShell):**

```powershell
az acr repository show-tags --name <youracrname> --repository bank-churn --output table
```

**Login to ACR:**

**Linux/Mac:**

```bash
az acr login --name <youracrname>
```

**Windows (PowerShell):**

```powershell
az acr login --name <youracrname>
```

**Delete ACR (if needed):**

**Linux/Mac:**

```bash
az acr delete \
  --name <youracrname> \
  --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
  --yes
```

**Windows (PowerShell):**

```powershell
az acr delete `
  --name <youracrname> `
  --resource-group $env:AZURE_RESOURCE_GROUP `
  --yes
```

## Access Azure ML Studio

After setup, access your workspace at:

```text
https://ml.azure.com
```

Navigate to your workspace: `${AZURE_WORKSPACE_NAME:-churn-ml-workspace}`

## Common Azure ML Commands

### Check Compute Cluster Status

**Linux/Mac:**

```bash
az ml compute show \
    --name "${AZURE_COMPUTE_CLUSTER_NAME:-cpu-cluster}" \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --workspace-name "${AZURE_WORKSPACE_NAME:-churn-ml-workspace}" \
    --query "{Name:name, State:provisioning_state, MinNodes:min_instances, MaxNodes:max_instances}" \
    --output table
```

**Windows (PowerShell):**

```powershell
az ml compute show `
    --name $env:AZURE_COMPUTE_CLUSTER_NAME `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --workspace-name $env:AZURE_WORKSPACE_NAME `
    --query "{Name:name, State:provisioning_state, MinNodes:min_instances, MaxNodes:max_instances}" `
    --output table
```

**Note:** Compute cluster auto-scales to 0 nodes when idle, so no charges when not in use.

### List All Compute Resources

**Linux/Mac:**

```bash
az ml compute list \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --workspace-name "${AZURE_WORKSPACE_NAME:-churn-ml-workspace}" \
    --output table
```

**Windows (PowerShell):**

```powershell
az ml compute list `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --workspace-name $env:AZURE_WORKSPACE_NAME `
    --output table
```

### Delete Compute Cluster

**Linux/Mac:**

```bash
az ml compute delete \
    --name "${AZURE_COMPUTE_CLUSTER_NAME:-cpu-cluster}" \
    --resource-group "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --workspace-name "${AZURE_WORKSPACE_NAME:-churn-ml-workspace}" \
    --yes
```

**Windows (PowerShell):**

```powershell
az ml compute delete `
    --name $env:AZURE_COMPUTE_CLUSTER_NAME `
    --resource-group $env:AZURE_RESOURCE_GROUP `
    --workspace-name $env:AZURE_WORKSPACE_NAME `
    --yes
```

### Delete Resource Group

**Warning:** This will delete the entire resource group and all resources within it (workspace, compute cluster, data assets, etc.). This action cannot be undone.

**Linux/Mac:**

```bash
az group delete \
    --name "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" \
    --yes \
    --no-wait
```

**Or as a one-liner:**

```bash
az group delete --name "${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}" --yes --no-wait
```

**Windows (PowerShell):**

```powershell
az group delete `
    --name $env:AZURE_RESOURCE_GROUP `
    --yes `
    --no-wait
```

**Note:** The `--no-wait` flag allows the deletion to proceed asynchronously. Remove it if you want to wait for confirmation of deletion.

## Cost Management Tips

- **Compute Cluster**: Only charges when nodes are active. Auto-scales to 0 when idle.
- Use `MIN_NODES=0` to ensure cluster scales down completely when idle.

For active setup and deployment troubleshooting, see
[`../docs/setup_guide.md`](../docs/setup_guide.md) and
[`../docs/features/online-endpoint-deployment/release-validation.md`](../docs/features/online-endpoint-deployment/release-validation.md).
