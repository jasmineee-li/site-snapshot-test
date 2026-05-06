#!/usr/bin/env bash
# Convenience wrapper: launch vLLM for GUI-Owl-1.5-32B-Think on default port 8003.
set -euo pipefail
exec bash "$(dirname "$0")/../models/gui_owl/serve_32b.sh" "$@"
