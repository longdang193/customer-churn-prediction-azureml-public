"""Create Azure ML data asset from local data directory.

This script registers a local data directory or file as a data asset in Azure ML.
The data asset is registered as uri_folder type, which is required for the
data_prep component to automatically load CSV files.

@meta
name: create_data_asset
type: utility
domain: setup
responsibility:
  - Provide setup behavior for `setup/create_data_asset.py`.
inputs: []
outputs: []
tags:
  - setup
features:
  - workspace-bootstrap
capabilities:
  - workspace-bootstrap.register-raw-dataset-azure-ml-data-asset
  - workspace-bootstrap.register-smoke-dataset-separate-azure-ml-data-asset
lifecycle:
  status: active
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data

from src.config.runtime import get_data_asset_config, load_azure_config


def format_success_message(data_asset_name: str, data_asset_version: str) -> str:
    """Return a Windows-console-safe success message."""
    return (
        "\nOK Success! Data asset registered.\n"
        f"  Reference: azureml:{data_asset_name}:{data_asset_version}"
    )


def format_create_error_message(error: Exception) -> str:
    """Return a Windows-console-safe create failure message."""
    return f"\nERROR creating data asset: {error}"


def format_configuration_error_message(error: Exception) -> str:
    """Return a Windows-console-safe configuration failure message."""
    return f"\nERROR configuration error: {error}"


def create_data_asset(
    data_path: Path,
    data_asset_name: str,
    data_asset_version: str = "1",
    description: str = "Bank customer churn dataset",
) -> None:
    """Create and register a data asset in Azure ML.

    @capability workspace-bootstrap.register-raw-dataset-azure-ml-data-asset
    @capability workspace-bootstrap.register-smoke-dataset-separate-azure-ml-data-asset

    Args:
        data_path: Path to the data directory or file
        data_asset_name: Name for the data asset
        data_asset_version: Version of the data asset (default: "1")
        description: Description of the data asset

    Raises:
        FileNotFoundError: If the data path does not exist
        Exception: If data asset creation fails
    """
    # Load Azure configuration
    config = load_azure_config()

    # Initialize ML Client
    print("Connecting to Azure ML workspace...")
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )

    # Validate data path
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data path not found: {data_path}\n"
            f"Please ensure the data file or directory exists."
        )

    # Convert to absolute path
    data_path = data_path.resolve()

    # Handle file vs directory
    if data_path.is_file():
        print(f"Note: File specified, using parent directory: {data_path.parent}")
        data_path = data_path.parent

    # Display information
    print(f"\nCreating data asset:")
    print(f"  Name: {data_asset_name}")
    print(f"  Version: {data_asset_version}")
    print(f"  Source: {data_path}")
    print(f"  Type: uri_folder")

    # Create data asset entity
    data_asset = Data(
        name=data_asset_name,
        version=data_asset_version,
        description=description,
        path=str(data_path),
        type="uri_folder",  # Required for data_prep component
    )

    # Register the data asset
    try:
        ml_client.data.create_or_update(data_asset)
        print(format_success_message(data_asset_name, data_asset_version))
    except Exception as e:
        print(format_create_error_message(e))
        sys.exit(1)


def main() -> None:
    """
    Main function to create data asset.

    @capability workspace-bootstrap.register-raw-dataset-azure-ml-data-asset
    @capability workspace-bootstrap.register-smoke-dataset-separate-azure-ml-data-asset
    """
    parser = argparse.ArgumentParser(
        description="Create Azure ML data asset from local data directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults from configs/assets.yaml
  python setup/create_data_asset.py

  # Specify custom name and version
  python setup/create_data_asset.py --name churn-data --version 1

  # Use a specific data directory
  python setup/create_data_asset.py --data-path data --name churn-data --version 1
        """,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Path to data directory or file (default: data)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Data asset name (default: from configs/assets.yaml, "
            "with optional DATA_ASSET_FULL env override)"
        ),
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help=(
            "Data asset version (default: from configs/assets.yaml, "
            "with optional DATA_VERSION env override)"
        ),
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Bank customer churn dataset",
        help="Description of the data asset",
    )

    args = parser.parse_args()

    # Get data asset configuration
    data_asset_config = get_data_asset_config()
    data_asset_name = args.name or data_asset_config["data_asset_name"]
    data_asset_version = args.version or data_asset_config["data_asset_version"]

    # Create data asset
    try:
        create_data_asset(
            data_path=Path(args.data_path),
            data_asset_name=data_asset_name,
            data_asset_version=data_asset_version,
            description=args.description,
        )
    except (ValueError, FileNotFoundError) as e:
        print(format_configuration_error_message(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

