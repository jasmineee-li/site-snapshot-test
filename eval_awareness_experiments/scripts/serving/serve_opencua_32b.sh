#!/usr/bin/env bash
# Convenience wrapper: launch vLLM for OpenCUA-32B on the default port (8001).
# See models/opencua/README.md for tuning knobs.
#
# Run in a tmux pane and leave it up; the runners will hit it via
# `--model-name local/opencua-32b`.
set -euo pipefail
exec bash "$(dirname "$0")/../../../models/opencua/serve_32b.sh" "$@"
