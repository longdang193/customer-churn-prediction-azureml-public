# @meta
# name: setup_ps1
# type: script
# features:
#   - workspace-bootstrap
# capabilities:
#   - workspace-bootstrap.create-update-resource-group-azure-ml-workspace
#   - workspace-bootstrap.create-compute-cluster-optional-compute-instance-notebook-workflows
#   - workspace-bootstrap.create-reuse-azure-container-registry-aml-environments
#   - workspace-bootstrap.provide-cloud-prerequisites-sdk-v2-jobs-registry-usage
# lifecycle:
#   status: active
#
# Azure ML Setup Script - Creates workspace and compute resources
$ErrorActionPreference = "Stop"

function Import-ConfigEnv {
    param([string]$Path = "config.env")

    if (-not (Test-Path $Path)) {
        Write-Host "[WARNING] config.env not found at $Path; using existing environment variables and script defaults." -ForegroundColor Yellow
        return
    }

    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }
        $key, $value = $line -split "=", 2
        $key = $key.Trim()
        $value = $value.Trim().Trim('"').Trim("'")
        [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

$RepoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { (Get-Location).Path }
Import-ConfigEnv -Path (Join-Path $RepoRoot "config.env")

# Configuration
$ResourceGroup = if ($env:AZURE_RESOURCE_GROUP) { $env:AZURE_RESOURCE_GROUP } else { "rg-churn-ml-project" }
$Location = if ($env:AZURE_LOCATION) { $env:AZURE_LOCATION } else { "southeastasia" }
$WorkspaceName = if ($env:AZURE_WORKSPACE_NAME) { $env:AZURE_WORKSPACE_NAME } else { "churn-ml-workspace" }
$ComputeClusterName = if ($env:AZURE_COMPUTE_CLUSTER_NAME) { $env:AZURE_COMPUTE_CLUSTER_NAME } else { "cpu-cluster" }
$ComputeClusterSize = if ($env:COMPUTE_CLUSTER_SIZE) { $env:COMPUTE_CLUSTER_SIZE } else { "Standard_DS2_v2" }
$ComputeInstanceName = if ($env:AZURE_COMPUTE_INSTANCE_NAME) { $env:AZURE_COMPUTE_INSTANCE_NAME } else { "ci-notebooks" }
$ComputeInstanceSize = if ($env:COMPUTE_INSTANCE_SIZE) { $env:COMPUTE_INSTANCE_SIZE } else { "Standard_DS2_v2" }
$CreateComputeInstance = if ($env:CREATE_COMPUTE_INSTANCE) { $env:CREATE_COMPUTE_INSTANCE } else { "true" }
$MinNodes = if ($env:MIN_NODES) { [int]$env:MIN_NODES } else { 0 }
$MaxNodes = if ($env:MAX_NODES) { [int]$env:MAX_NODES } else { 2 }
$AcrName = if ($env:AZURE_ACR_NAME) { $env:AZURE_ACR_NAME } else { "" }
$AcrSku = if ($env:ACR_SKU) { $env:ACR_SKU } else { "Basic" }

function Write-Info { param([string]$Message) Write-Host "[INFO] $Message" -ForegroundColor Green }
function Write-Warning-Custom { param([string]$Message) Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
function Write-Error-Custom { param([string]$Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }

function Test-AzureCLI {
    try {
        $null = az --version 2>&1 | Select-Object -First 1
    }
    catch {
        Write-Error-Custom "Azure CLI not installed. Install from: https://docs.microsoft.com/cli/azure/install-azure-cli"
        exit 1
    }
}

function Test-AzureLogin {
    try {
        $null = az account show 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warning-Custom "Not logged in. Running: az login"
            az login
        }
        $subscriptionId = az account show --query id -o tsv
        Write-Info "Using subscription: $subscriptionId"
    }
    catch {
        Write-Error-Custom "Failed to authenticate with Azure"
        exit 1
    }
}

function New-ResourceGroup {
    Write-Info "Creating resource group: $ResourceGroup"
    $null = az group show --name $ResourceGroup 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Warning-Custom "Resource group already exists"
    }
    else {
        az group create --name $ResourceGroup --location $Location
        Write-Info "Resource group created"
    }
}

function New-Workspace {
    Write-Info "Creating workspace: $WorkspaceName"
    $null = az ml workspace show --resource-group $ResourceGroup --name $WorkspaceName 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Warning-Custom "Workspace already exists"
    }
    else {
        az ml workspace create --resource-group $ResourceGroup --name $WorkspaceName --location $Location
        Write-Info "Workspace created"
    }
}

function New-ComputeCluster {
    Write-Info "Creating compute cluster: $ComputeClusterName"
    $null = az ml compute show --resource-group $ResourceGroup --workspace-name $WorkspaceName --name $ComputeClusterName 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Warning-Custom "Compute cluster already exists"
        return
    }
    Write-Info "Creating compute cluster with system-assigned managed identity for ACR access"
    az ml compute create --resource-group $ResourceGroup --workspace-name $WorkspaceName `
        --name $ComputeClusterName --type AmlCompute --size $ComputeClusterSize `
        --min-instances $MinNodes --max-instances $MaxNodes --idle-time-before-scale-down 1800 `
        --identity-type system_assigned
    Write-Info "Compute cluster created with managed identity"
    
    # If ACR exists, AcrPull role is automatically granted by Azure ML
    if (-not [string]::IsNullOrEmpty($AcrName)) {
        Write-Info "AcrPull role will be automatically granted to compute managed identity (ACR exists)"
    }
}

function New-ComputeInstance {
    if ($CreateComputeInstance.ToLower() -eq "false") {
        Write-Warning-Custom "CREATE_COMPUTE_INSTANCE=false, skipping compute instance creation."
        return
    }

    if ([string]::IsNullOrEmpty($ComputeInstanceName)) {
        Write-Warning-Custom "AZURE_COMPUTE_INSTANCE_NAME not set. Skipping compute instance creation."
        return
    }

    Write-Info "Creating compute instance: $ComputeInstanceName"
    $null = az ml compute show --resource-group $ResourceGroup --workspace-name $WorkspaceName --name $ComputeInstanceName 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Warning-Custom "Compute instance already exists"
        return
    }

    az ml compute create --resource-group $ResourceGroup --workspace-name $WorkspaceName `
        --name $ComputeInstanceName --type computeinstance --size $ComputeInstanceSize
    Write-Info "Compute instance created. Start it from Azure ML Studio before launching notebooks."
}

function New-Acr {
    if ([string]::IsNullOrEmpty($AcrName)) {
        Write-Warning-Custom "AZURE_ACR_NAME not set. Skipping ACR creation."
        Write-Info "You can create ACR later using: az acr create --resource-group $ResourceGroup --name <acr-name> --sku Basic --location $Location"
        return
    }
    
    Write-Info "Creating Azure Container Registry: $AcrName"
    $null = az acr show --resource-group $ResourceGroup --name $AcrName 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Warning-Custom "ACR already exists"
    }
    else {
        az acr create --resource-group $ResourceGroup --name $AcrName --sku $AcrSku --location $Location
        Write-Info "ACR created successfully"
        Write-Info "ACR login server: $AcrName.azurecr.io"
    }
}

function Show-Info {
    Write-Info "=== Setup Complete ==="
    Write-Host "Resource Group: $ResourceGroup"
    Write-Host "Workspace: $WorkspaceName"
    Write-Host "Location: $Location"
    Write-Host "Compute Cluster: $ComputeClusterName ($MinNodes-$MaxNodes nodes)"
    if ($CreateComputeInstance.ToLower() -ne "false") {
        Write-Host "Compute Instance: $ComputeInstanceName ($ComputeInstanceSize)"
    }
    if (-not [string]::IsNullOrEmpty($AcrName)) {
        Write-Host "ACR: $AcrName ($AcrName.azurecr.io)"
        Write-Host "  - AcrPull role automatically granted to compute managed identity"
    }
    else {
        Write-Host "ACR: Not created (set AZURE_ACR_NAME in config.env to create)"
    }
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "1. Access https://ml.azure.com and navigate to workspace: $WorkspaceName"
    if (-not [string]::IsNullOrEmpty($AcrName)) {
        Write-Host "2. After building/pushing the runtime image, update aml/environments/environment.yml to use: $AcrName.azurecr.io/bank-churn:1"
        Write-Host "3. Build and push Docker image to ACR (see docs/setup_guide.md: 'Build, Push, and Register the Azure ML Environment')"
    }
    else {
        Write-Host "2. Create ACR and update config.env, then recreate compute cluster for automatic AcrPull"
    }
}

function Main {
    Write-Info "Starting Azure ML setup..."
    Test-AzureCLI
    Test-AzureLogin
    New-ResourceGroup
    New-Workspace
    # Create ACR BEFORE compute cluster so AcrPull role is automatically granted
    New-Acr
    New-ComputeCluster
    New-ComputeInstance
    Write-Host ""
    Show-Info
    Write-Info "Setup completed successfully!"
}

Main
