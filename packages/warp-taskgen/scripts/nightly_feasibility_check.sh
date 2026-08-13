#!/usr/bin/env bash
# Nightly Phase 2c re-verify against the dev host.
#
# Purpose: surface silent platform drift (GitLab secret rotation, Magento
# review policy changes, PostMill schema migration, etc.) before it
# contaminates an ASR run. Previously-verified tasks within the TTL skip
# quickly on fingerprint match, so the typical cost is "acquire tokens +
# probe_base_state" per site.
#
# Recommended cron (America/New_York, pick any off-peak hour):
#
#   0 3 * * * cd /path/to/browser-sim && bash scripts/nightly_feasibility_check.sh >> logs/cron/nightly_feasibility.log 2>&1
#
# Override ``INSTANCES_FILE`` (default ``instances.smoke.json``) or
# ``TTL_HOURS`` (default ``24``) via env var. Auth env vars (e.g.
# ``WORLDSIM_SHOPPING_AUTO_LOGIN``) must be present — source ``.env`` or
# wrap this script in one that does.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

INSTANCES_FILE="${INSTANCES_FILE:-instances.smoke.json}"
TTL_HOURS="${TTL_HOURS:-24}"
CONCURRENCY="${FEASIBILITY_CONCURRENCY:-10}"

if [[ ! -f "$INSTANCES_FILE" ]]; then
    echo "ERROR: instances file does not exist: $INSTANCES_FILE" >&2
    exit 1
fi

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

mkdir -p logs/cron
stamp="$(date -u +%Y%m%dT%H%M%SZ)"

printf 'nightly-feasibility %s — starting (instances=%s, ttl_hours=%s)\n' \
    "$stamp" "$INSTANCES_FILE" "$TTL_HOURS"

uv run python -m warp_taskgen.main phase 2c \
    --feasibility-instances "$INSTANCES_FILE" \
    --feasibility-concurrency "$CONCURRENCY" \
    --feasibility-ttl-hours "$TTL_HOURS"

printf 'nightly-feasibility %s — done\n' "$stamp"
