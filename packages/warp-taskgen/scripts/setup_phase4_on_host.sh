#!/usr/bin/env bash
# setup_phase4_on_host.sh — bring a fresh host from bootstrap-complete to
# Phase-4-ready state. Idempotent: safe to rerun; exits 0 if already green.
#
# Prereq: ``bootstrap_r5.sh`` (or equivalent) has run and all benchmark
# containers are up with env-ctrl responding.
#
# The script codifies everything the operator had to do by hand on the
# 2026-04-20 r5 setup. Run order matters: uv/venvs → playwright system
# deps → instances regen → PVPO page-surface-stable check → artifact sync →
# Phase 0d storage_state mint → preflight gate.
#
# Usage:
#   scripts/setup_phase4_on_host.sh \
#       --host-config configs/benchmark_hosts/r5.yaml \
#       --instances instances.scale.json \
#       --artifacts-source s3://benchmark-archives/worldsim-runs/<id>/
#
# If --artifacts-source is omitted, the script looks for phase_0c/2/3
# artifacts locally and fails loudly if any are missing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

HOST_CONFIG=""
INSTANCES="${INSTANCES:-instances.scale.json}"
ARTIFACTS_SOURCE=""
BENCHMARK_ROOT="${WORLDSIM_BENCHMARK_ROOT:-/home/ubuntu/vendors/webarena-verified}"
SCALE_CONFIG="${WORLDSIM_SCALE_CONFIG:-scripts/scale_config.yml}"
SKIP_PVPO_CONTAINER=1
SKIP_MAGENTO_SYNC=0
SKIP_GITLAB_MINT=0

usage() {
    cat <<USAGE
setup_phase4_on_host.sh

Options:
  --host-config <path>       benchmark host YAML (required)
  --instances <path>         instances.json (default: instances.scale.json)
  --scale-config <path>      scale topology YAML for regeneration (default: scripts/scale_config.yml)
  --artifacts-source <uri>   s3://, ssh://, or /local/path for phase_0c/2/3
  --benchmark-root <path>    WebArena Verified checkout (default: /home/ubuntu/vendors/webarena-verified)
  --skip-pvpo-container      deprecated no-op; page-surface-stable PVPO needs no Docker container
  --skip-magento-sync        deprecated no-op (step 6 removed 2026-04-21)
  --skip-gitlab-mint         skip step 5 (Phase 0d storage_state mint)
  -h, --help                 show this help
USAGE
}

while (("$#")); do
    case "$1" in
        --host-config) HOST_CONFIG="$2"; shift 2 ;;
        --instances) INSTANCES="$2"; shift 2 ;;
        --scale-config) SCALE_CONFIG="$2"; shift 2 ;;
        --artifacts-source) ARTIFACTS_SOURCE="$2"; shift 2 ;;
        --benchmark-root) BENCHMARK_ROOT="$2"; shift 2 ;;
        --skip-pvpo-container) SKIP_PVPO_CONTAINER=1; shift ;;
        --skip-magento-sync) SKIP_MAGENTO_SYNC=1; shift ;;
        --skip-gitlab-mint) SKIP_GITLAB_MINT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

log() { printf '==> %s\n' "$*" >&2; }
substep() { printf '    %s\n' "$*" >&2; }
abs_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$REPO_ROOT" "$1" ;;
    esac
}

if [[ -z "$HOST_CONFIG" ]]; then
    echo "ERROR: --host-config required" >&2
    usage
    exit 2
fi
HOST_CONFIG_PATH="$(abs_path "$HOST_CONFIG")"
INSTANCES_PATH="$(abs_path "$INSTANCES")"
SCALE_CONFIG_PATH="$(abs_path "$SCALE_CONFIG")"
if [[ "$SKIP_MAGENTO_SYNC" -eq 1 ]]; then
    substep "--skip-magento-sync is deprecated; Magento left WASP scope on 2026-04-21"
fi

# ---------------------------------------------------------------------------
# Step 1 — uv + deps + evaluator venv (issues #1, #2 fallback, #17)
# ---------------------------------------------------------------------------
log "step 1: uv + deps + evaluator venv"
if ! command -v uv >/dev/null 2>&1; then
    substep "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
# uv lock --check surfaces pyproject.toml / lockfile drift with a clear
# error before sync; otherwise sync failures are cryptic.
uv lock --check >/dev/null 2>&1 || {
    echo "ERROR: pyproject.toml + uv.lock are out of sync; run 'uv lock' first" >&2
    exit 2
}
uv sync --locked --extra dev
(
    cd "$REPO_ROOT/packages/warp-taskgen-webarena-verified"
    uv lock --check >/dev/null 2>&1 || {
        echo "ERROR: packages/warp-taskgen-webarena-verified lock drift" >&2
        exit 2
    }
    uv sync --locked
)

# Resolve orchestrator_host from the host config once (needed by step 5 to
# mint storage_state against the same host site_url uses). Phase 0d cookies
# are domain-scoped — minting against one host and reusing on another is
# the host_bound_storage_state failure class. orchestrator_host falls back
# to advertise_host when unset, matching the host_config.py schema default.
ORCHESTRATOR_HOST="$(uv run python -c '
import sys, yaml, pathlib
data = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
print(str(data.get("orchestrator_host") or data["advertise_host"]).strip())
' "$HOST_CONFIG_PATH")"
substep "orchestrator_host=${ORCHESTRATOR_HOST} (from $HOST_CONFIG)"

# ---------------------------------------------------------------------------
# Step 1b — regen instances.scale.json from scale_config.yml + host config.
# ---------------------------------------------------------------------------
# instances.scale.json is gitignored; regen every time so edits to
# scripts/scale_config.yml or the host config propagate without hand-
# patching the 62 fields that got bandaided on 2026-04-21. Also keeps
# advertise_host ↔ control_host in sync with the host's actual topology.
log "step 1b: regen $INSTANCES"
"$REPO_ROOT/scripts/generate_scale_r5.sh" \
    --host-config "$HOST_CONFIG_PATH" \
    --scale-config "$SCALE_CONFIG_PATH" >/dev/null
INSTANCES_PATH="$(abs_path "$INSTANCES")"
substep "regenerated $INSTANCES from $SCALE_CONFIG"

# ---------------------------------------------------------------------------
# Step 2 — Playwright chromium + system libs (issue #3)
# ---------------------------------------------------------------------------
# Playwright is optional in current Browser Use/AgentLab paths, but when present
# its managed Chromium and system libs are still useful for local BrowserGym
# compatibility checks. Skip gracefully if the module is absent — loud failure
# on an optional dep is noise.
log "step 2: playwright chromium + system libs"
if uv run python -c "import playwright" >/dev/null 2>&1; then
    uv run python -m playwright install chromium
    if [[ "$(uname -s)" == "Linux" ]]; then
        sudo "$(command -v uv)" run python -m playwright install-deps chromium
    else
        substep "non-Linux host: skipping install-deps (macOS / WSL have system libs bundled)"
    fi
else
    substep "playwright not installed; skipping optional browser install"
fi

# ---------------------------------------------------------------------------
# Step 3 — PVPO browser capture
# ---------------------------------------------------------------------------
log "step 3: PVPO page-surface-stable capture"
substep "no dedicated PVPO browser endpoints required"

# ---------------------------------------------------------------------------
# Step 4 — Phase artifact sync (issues #4, #5, #6)
# ---------------------------------------------------------------------------
log "step 4: phase_0c / phase_2 / phase_3 artifact sync"
STATE_DIR="${WORLDSIM_STATE_DIR:-logs}"
mkdir -p "$STATE_DIR"
NEED=(phase_0c phase_2 phase_3)
MISSING=()
for dir in "${NEED[@]}"; do
    if [[ ! -e "$STATE_DIR/$dir" ]]; then
        MISSING+=("$dir")
    fi
done
if [[ "${#MISSING[@]}" -eq 0 ]]; then
    substep "artifacts already present under $STATE_DIR"
elif [[ -z "$ARTIFACTS_SOURCE" ]]; then
    echo "ERROR: missing artifact dirs: ${MISSING[*]}" >&2
    echo "Pass --artifacts-source s3://bucket/prefix OR rsync them locally into $STATE_DIR/" >&2
    exit 2
else
    case "$ARTIFACTS_SOURCE" in
        s3://*)
            command -v aws >/dev/null || { echo "ERROR: aws CLI not found" >&2; exit 2; }
            substep "aws s3 sync $ARTIFACTS_SOURCE $STATE_DIR/"
            aws s3 sync "${ARTIFACTS_SOURCE%/}/" "$STATE_DIR/" --exact-timestamps
            ;;
        ssh://*)
            # ssh://host:/path
            remote="${ARTIFACTS_SOURCE#ssh://}"
            substep "rsync -a $remote/ $STATE_DIR/"
            rsync -a "$remote/" "$STATE_DIR/"
            ;;
        /*)
            substep "cp -r $ARTIFACTS_SOURCE/* $STATE_DIR/"
            cp -r "$ARTIFACTS_SOURCE"/* "$STATE_DIR/"
            ;;
        *)
            echo "ERROR: unsupported --artifacts-source scheme: $ARTIFACTS_SOURCE" >&2
            exit 2
            ;;
    esac
fi
MANIFEST_ARGS=(
    --state-dir "$STATE_DIR"
    --instances "$INSTANCES_PATH"
    --output "$STATE_DIR/artifact_manifest.json"
)
if [[ -n "$ARTIFACTS_SOURCE" ]]; then
    MANIFEST_ARGS+=(--artifacts-source "$ARTIFACTS_SOURCE")
fi
uv run python scripts/write_phase4_artifact_manifest.py "${MANIFEST_ARGS[@]}" >/dev/null
substep "wrote artifact manifest to $STATE_DIR/artifact_manifest.json"

# ---------------------------------------------------------------------------
# Step 5 — Mint Phase 0d storage_state for gitlab (issue #7)
# ---------------------------------------------------------------------------
if [[ "$SKIP_GITLAB_MINT" -eq 0 ]]; then
    if [[ ! -d "$BENCHMARK_ROOT" ]]; then
        echo "ERROR: benchmark root not found: $BENCHMARK_ROOT" >&2
        echo "Set WORLDSIM_BENCHMARK_ROOT or pass --benchmark-root." >&2
        exit 2
    fi
    # Always re-mint, regardless of whether a storage_state file is already
    # on disk. Stale cookies bound to the wrong advertise_host (laptop
    # loopback vs r5 public IP, or a previous host) are invisible at
    # mint-time but fail the runtime preflight with host_bound_storage_state.
    # Re-minting costs ~30s; debugging a stale artifact costs a whole run.
    log "step 5: mint Phase 0d storage_state from $INSTANCES (unconditional)"
    uv run python -m worldsim.main phase 0d \
        --benchmark "$BENCHMARK_ROOT" \
        --instances "$INSTANCES_PATH"
fi

# Step 6 (Magento base_url sync) was removed 2026-04-21 with the
# WASP-aligned scoping decision.

# ---------------------------------------------------------------------------
# Step 7 — Preflight via pytest
# ---------------------------------------------------------------------------
log "step 7: preflight (pytest -m preflight)"
PYTEST_ARGS=(-m preflight tests/preflight -v)
if [[ -n "$HOST_CONFIG" ]]; then
    export WORLDSIM_PREFLIGHT_HOST_CONFIG="$HOST_CONFIG_PATH"
fi
export WORLDSIM_PREFLIGHT_INSTANCES="$INSTANCES_PATH"
uv run pytest "${PYTEST_ARGS[@]}"

log "Phase 4 setup complete. Next: uv run warp-taskgen resume --instances $INSTANCES"
