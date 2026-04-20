"""
@meta
name: download_capture_blob
type: utility
domain: inference-monitoring
responsibility:
  - Download one exact caller-capture record from local disk or Blob-backed capture storage.
inputs:
  - Capture URI or local path
  - Destination file path
  - Blob connection-string/container environment variable names
outputs:
  - Downloaded capture file on local disk
tags:
  - monitoring
  - inference
  - azure-blob
  - utility
capabilities:
  - monitor.treat-blob-backed-caller-capture-exact-path-evidence
  - online-deploy.support-exact-caller-side-blob-capture-retrieval-repeatable
lifecycle:
  status: active
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient

DEFAULT_DOWNLOAD_RETRY_ATTEMPTS = 3
DEFAULT_DOWNLOAD_RETRY_DELAY_SECONDS = 1.0


def parse_capture_blob_path(capture_uri: str) -> str:
    if capture_uri.startswith("azureml://"):
        marker = "/paths/"
        if marker not in capture_uri:
            raise ValueError(f"Unsupported Azure ML capture URI: {capture_uri}")
        return capture_uri.split(marker, maxsplit=1)[1]
    if capture_uri.startswith("azureblob://"):
        parts = capture_uri.split("/", maxsplit=4)
        if len(parts) < 5:
            raise ValueError(f"Unsupported Azure Blob capture URI: {capture_uri}")
        return parts[4]
    raise ValueError(f"Unsupported capture URI: {capture_uri}")


def download_capture_blob(
    *,
    capture_uri: str,
    output_file: Path,
    connection_string_env: str,
    container_env: str,
    retry_attempts: int = DEFAULT_DOWNLOAD_RETRY_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_DOWNLOAD_RETRY_DELAY_SECONDS,
) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    local_path = Path(capture_uri)
    if local_path.exists():
        shutil.copy2(local_path, output_file)
        return output_file

    connection_string = os.getenv(connection_string_env)
    if not connection_string:
        raise RuntimeError(
            f"Blob download requires environment variable '{connection_string_env}'."
        )
    container_name = os.getenv(container_env)
    if not container_name:
        raise RuntimeError(
            f"Blob download requires environment variable '{container_env}'."
        )

    blob_path = parse_capture_blob_path(capture_uri)
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_path)
    for attempt in range(1, retry_attempts + 1):
        try:
            output_file.write_bytes(blob_client.download_blob().readall())
            return output_file
        except ResourceNotFoundError:
            if attempt >= retry_attempts:
                raise
            time.sleep(retry_delay_seconds)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download one exact caller-capture record from its manifest URI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--capture-uri", required=True, help="Local path, azureml://, or azureblob:// capture URI")
    parser.add_argument("--output-file", required=True, help="Destination file path")
    parser.add_argument(
        "--connection-string-env",
        default="INFERENCE_CAPTURE_STORAGE_CONNECTION_STRING",
        help="Environment variable name that holds the Blob connection string",
    )
    parser.add_argument(
        "--container-env",
        default="INFERENCE_CAPTURE_STORAGE_CONTAINER",
        help="Environment variable name that holds the Blob container name",
    )
    args = parser.parse_args()

    destination = download_capture_blob(
        capture_uri=args.capture_uri,
        output_file=Path(args.output_file),
        connection_string_env=args.connection_string_env,
        container_env=args.container_env,
    )
    print(destination)


if __name__ == "__main__":
    main()
