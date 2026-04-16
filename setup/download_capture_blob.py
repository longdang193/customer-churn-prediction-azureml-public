"""
@meta
name: download_capture_blob
type: script
domain: inference-monitoring
responsibility:
  - Preserve the historical setup CLI path for exact capture downloads as a compatibility wrapper.
  - Forward all runtime download behavior to the monitoring-owned implementation under `src/monitoring/`.
  - Steer new runtime callers toward `src/monitoring/download_capture_blob.py` instead of this compatibility path.
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
  - compatibility
lifecycle:
  status: active
"""

from __future__ import annotations

from src.monitoring.download_capture_blob import (
    DEFAULT_DOWNLOAD_RETRY_ATTEMPTS,
    DEFAULT_DOWNLOAD_RETRY_DELAY_SECONDS,
    download_capture_blob,
    main,
    parse_capture_blob_path,
)

# Compatibility-only import surface for older tests and manual scripts.
# New runtime callers should use `src/monitoring/download_capture_blob.py`.

if __name__ == "__main__":
    main()
