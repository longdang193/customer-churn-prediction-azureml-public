#!/bin/bash

# Azure ML Setup Script - Creates workspace and compute resources
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_ENV="$REPO_ROOT/config.env"

if [ -f "$CONFIG_ENV" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$CONFIG_ENV"
    set +a
else
    echo "[WARNING] config.env not found at $CONFIG_ENV; using existing environment variables and script defaults."
fi

# Configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-rg-churn-ml-project}"
LOCATION="${AZURE_LOCATION:-southeastasia}"
WORKSPACE_NAME="${AZURE_WORKSPACE_NAME:-churn-ml-workspace}"
COMPUTE_CLUSTER_NAME="${AZURE_COMPUTE_CLUSTER_NAME:-cpu-cluster}"
COMPUTE_CLUSTER_SIZE="${COMPUTE_CLUSTER_SIZE:-Standard_DS2_v2}"
MIN_NODES="${MIN_NODES:-0}"
MAX_NODES="${MAX_NODES:-2}"
ACR_NAME="${AZURE_ACR_NAME:-}"
ACR_SKU="${ACR_SKU:-Basic}"
COMPUTE_INSTANCE_NAME="${AZURE_COMPUTE_INSTANCE_NAME:-ci-notebooks}"
COMPUTE_INSTANCE_SIZE="${COMPUTE_INSTANCE_SIZE:-Standard_DS2_v2}"
CREATE_COMPUTE_INSTANCE="${CREATE_COMPUTE_INSTANCE:-true}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_azure_cli() {
    if ! command -v az &> /dev/null; then
        print_error "Azure CLI not installed. Install from: https://docs.microsoft.com/cli/azure/install-azure-cli"
        exit 1
    fi
}

check_azure_login() {
    if ! az account show &> /dev/null; then
        print_warning "Not logged in. Running: az login"
        az login
    fi
    SUBSCRIPTION_ID=$(az account show --query id -o tsv)
    print_info "Using subscription: $SUBSCRIPTION_ID"
}

create_resource_group() {
    print_info "Creating resource group: $RESOURCE_GROUP"
    if az group show --name "$RESOURCE_GROUP" &> /dev/null; then
        print_warning "Resource group already exists"
    else
        az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
        print_info "Resource group created"
    fi
}

create_workspace() {
    print_info "Creating workspace: $WORKSPACE_NAME"
    if az ml workspace show --resource-group "$RESOURCE_GROUP" --name "$WORKSPACE_NAME" &> /dev/null; then
        print_warning "Workspace already exists"
    else
        az ml workspace create --resource-group "$RESOURCE_GROUP" --name "$WORKSPACE_NAME" --location "$LOCATION"
        print_info "Workspace created"
    fi
}

create_compute_cluster() {
    print_info "Creating compute cluster: $COMPUTE_CLUSTER_NAME"
    if az ml compute show --resource-group "$RESOURCE_GROUP" --workspace-name "$WORKSPACE_NAME" --name "$COMPUTE_CLUSTER_NAME" &> /dev/null; then
        print_warning "Compute cluster already exists"
        return
    fi
    print_info "Creating compute cluster with system-assigned managed identity for ACR access"
    az ml compute create --resource-group "$RESOURCE_GROUP" --workspace-name "$WORKSPACE_NAME" \
        --name "$COMPUTE_CLUSTER_NAME" --type AmlCompute --size "$COMPUTE_CLUSTER_SIZE" \
        --min-instances "$MIN_NODES" --max-instances "$MAX_NODES" --idle-time-before-scale-down 1800 \
        --identity-type system_assigned
    print_info "Compute cluster created with managed identity"
    
    # If ACR exists, AcrPull role is automatically granted by Azure ML
    if [ -n "$ACR_NAME" ]; then
        print_info "AcrPull role will be automatically granted to compute managed identity (ACR exists)"
    fi
}

create_compute_instance() {
    if [[ "${CREATE_COMPUTE_INSTANCE,,}" == "false" ]]; then
        print_warning "CREATE_COMPUTE_INSTANCE=false, skipping compute instance creation."
        return
    fi

    if [ -z "$COMPUTE_INSTANCE_NAME" ]; then
        print_warning "AZURE_COMPUTE_INSTANCE_NAME not set. Skipping compute instance creation."
        return
    fi

    print_info "Creating compute instance: $COMPUTE_INSTANCE_NAME"
    if az ml compute show --resource-group "$RESOURCE_GROUP" --workspace-name "$WORKSPACE_NAME" --name "$COMPUTE_INSTANCE_NAME" &> /dev/null; then
        print_warning "Compute instance already exists"
        return
    fi

    az ml compute create --resource-group "$RESOURCE_GROUP" --workspace-name "$WORKSPACE_NAME" \
        --name "$COMPUTE_INSTANCE_NAME" --type computeinstance --size "$COMPUTE_INSTANCE_SIZE"
    print_info "Compute instance created. Start it from Azure ML Studio before launching notebooks."
}

create_acr() {
    if [ -z "$ACR_NAME" ]; then
        print_warning "AZURE_ACR_NAME not set. Skipping ACR creation."
        print_info "You can create ACR later using: az acr create --resource-group $RESOURCE_GROUP --name <acr-name> --sku Basic --location $LOCATION"
        return
    fi
    
    print_info "Creating Azure Container Registry: $ACR_NAME"
    if az acr show --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" &> /dev/null; then
        print_warning "ACR already exists"
    else
        az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku "$ACR_SKU" --location "$LOCATION"
        print_info "ACR created successfully"
        print_info "ACR login server: $ACR_NAME.azurecr.io"
    fi
}

display_info() {
    print_info "=== Setup Complete ==="
    echo "Resource Group: $RESOURCE_GROUP"
    echo "Workspace: $WORKSPACE_NAME"
    echo "Location: $LOCATION"
    echo "Compute Cluster: $COMPUTE_CLUSTER_NAME ($MIN_NODES-$MAX_NODES nodes)"
    if [[ "${CREATE_COMPUTE_INSTANCE,,}" != "false" ]]; then
        echo "Compute Instance: ${COMPUTE_INSTANCE_NAME:-<not set>} (${COMPUTE_INSTANCE_SIZE})"
    fi
    if [ -n "$ACR_NAME" ]; then
        echo "ACR: $ACR_NAME ($ACR_NAME.azurecr.io)"
        echo "  - AcrPull role automatically granted to compute managed identity"
    else
        echo "ACR: Not created (set AZURE_ACR_NAME in config.env to create)"
    fi
    echo ""
    echo "Next steps:"
    echo "1. Access https://ml.azure.com and navigate to workspace: $WORKSPACE_NAME"
    if [ -n "$ACR_NAME" ]; then
        echo "2. After building/pushing the runtime image, update aml/environments/environment.yml to use: $ACR_NAME.azurecr.io/bank-churn:1"
        echo "3. Build and push Docker image to ACR (see docs/setup_guide.md: 'Build, Push, and Register the Azure ML Environment')"
    else
        echo "2. Create ACR and update config.env, then recreate compute cluster for automatic AcrPull"
    fi
}

main() {
    print_info "Starting Azure ML setup..."
    check_azure_cli
    check_azure_login
    create_resource_group
    create_workspace
    # Create ACR BEFORE compute cluster so AcrPull role is automatically granted
    create_acr
    create_compute_cluster
    create_compute_instance
    echo ""
    display_info
    print_info "Setup completed successfully!"
}

main
