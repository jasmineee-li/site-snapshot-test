#!/usr/bin/env bash
# Fast, deterministic local verification for agent sessions.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_SILENT="$ROOT_DIR/scripts/lib/run_silent.sh"

"$RUN_SILENT" "ruff scoped source" "uv run ruff check worldsim tests scripts"
"$RUN_SILENT" "pytest collection" "uv run pytest --collect-only -q"
"$RUN_SILENT" "readiness audit" "uv run python scripts/readiness_audit.py --fail-on tracked-generated --fail-on tokens --fail-on legacy-imports"
