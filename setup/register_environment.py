"""Register the Azure ML runtime environment from central config.

This script avoids manual edits to aml/environments/environment.yml by deriving the
environment image URI from configs/assets.yaml plus AZURE_ACR_NAME in config.env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

from src.config.runtime import get_environment_asset_config, load_azure_config


def register_environment(config_path: str | None = None) -> Environment:
    """Create or update the AML environment asset from central config."""
    azure_config = load_azure_config(config_path)
    environment_config = get_environment_asset_config(config_path)

    print("Connecting to Azure ML workspace...")
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=azure_config["subscription_id"],
        resource_group_name=azure_config["resource_group"],
        workspace_name=azure_config["workspace_name"],
    )

    environment = Environment(
        name=environment_config["name"],
        version=environment_config["version"],
        image=environment_config["image"],
        description=(
            "Runtime environment for the churn prediction pipelines, "
            "resolved from configs/assets.yaml and config.env"
        ),
    )

    print("\nRegistering AML environment:")
    print(f"  Name: {environment_config['name']}")
    print(f"  Version: {environment_config['version']}")
    print(f"  Image: {environment_config['image']}")

    registered_environment = ml_client.environments.create_or_update(environment)

    print("\n✓ Success! AML environment registered.")
    print(
        "  Reference: "
        f"azureml:{environment_config['name']}:{environment_config['version']}"
    )
    return registered_environment


def main() -> None:
    """Parse arguments and register the AML environment."""
    parser = argparse.ArgumentParser(
        description="Register the AML runtime environment from central config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register using config.env + configs/assets.yaml
  python setup/register_environment.py

  # Register using an alternate env file
  python setup/register_environment.py --config-path config.env
        """,
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="config.env",
        help="Path to the env file used for Azure workspace and ACR settings",
    )
    args = parser.parse_args()

    try:
        register_environment(args.config_path)
    except ValueError as exc:
        print(f"\n✗ Configuration error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n✗ Error registering AML environment: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
