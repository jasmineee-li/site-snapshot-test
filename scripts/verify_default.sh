#!/usr/bin/env bash
# Default local verification before shipping normal repo changes.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

"$ROOT_DIR/scripts/verify_fast.sh"
"$ROOT_DIR/scripts/lib/run_silent.sh" \
    "default pytest parallel" \
    "uv run pytest -q -n auto --dist worksteal"
