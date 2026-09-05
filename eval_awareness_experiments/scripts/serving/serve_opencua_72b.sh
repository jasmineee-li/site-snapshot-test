#!/usr/bin/env bash
# Convenience wrapper: launch vLLM for OpenCUA-72B on default port (8002, TP=2).
set -euo pipefail
exec bash "$(dirname "$0")/../../../models/opencua/serve_72b.sh" "$@"
