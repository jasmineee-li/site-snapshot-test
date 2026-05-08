#!/usr/bin/env bash
# Fast, deterministic local verification for agent sessions.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_SILENT="$ROOT_DIR/scripts/lib/run_silent.sh"
SKIP_COLLECT=0

for arg in "$@"; do
    case "$arg" in
        --skip-collect)
            SKIP_COLLECT=1
            ;;
        *)
            echo "usage: $0 [--skip-collect]" >&2
            exit 2
            ;;
    esac
done

"$RUN_SILENT" "ruff scoped source" "uv run ruff check worldsim tests scripts"
if [[ "$SKIP_COLLECT" -eq 0 ]]; then
    "$RUN_SILENT" "pytest collection" "uv run pytest --collect-only -q"
fi
"$RUN_SILENT" "readiness audit" "uv run python scripts/readiness_audit.py --fail-on tracked-generated --fail-on tokens --fail-on legacy-imports"
