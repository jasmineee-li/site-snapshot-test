#!/usr/bin/env bash
# setup_phase4_on_host.sh — bring a fresh host from bootstrap-complete to
# Phase-4-ready state. Idempotent: safe to rerun; exits 0 if already green.
#
# Prereq: ``bootstrap_r5.sh`` (or equivalent) has run and all benchmark
# containers are up with env-ctrl responding.
#
# The script codifies everything the operator had to do by hand on the
# 2026-04-20 r5 setup. Run order matters: uv/venvs → playwright system
# deps → pvpo-chrome container → artifact sync → storage_state mint →
# Magento base_url sync → preflight gate.
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
SKIP_PVPO_CONTAINER=0
SKIP_MAGENTO_SYNC=0
SKIP_GITLAB_MINT=0

usage() {
    cat <<USAGE
setup_phase4_on_host.sh

Options:
  --host-config <path>       benchmark host YAML (required)
  --instances <path>         instances.json (default: instances.scale.json)
  --artifacts-source <uri>   s3://, ssh://, or /local/path for phase_0c/2/3
  --skip-pvpo-container      skip step 3 (PVPO Docker container)
  --skip-magento-sync        skip step 6 (sync_magento_base_urls.py)
  --skip-gitlab-mint         skip step 5 (login_gitlab_r5.py)
  -h, --help                 show this help
USAGE
}

while (("$#")); do
    case "$1" in
        --host-config) HOST_CONFIG="$2"; shift 2 ;;
        --instances) INSTANCES="$2"; shift 2 ;;
        --artifacts-source) ARTIFACTS_SOURCE="$2"; shift 2 ;;
        --skip-pvpo-container) SKIP_PVPO_CONTAINER=1; shift ;;
        --skip-magento-sync) SKIP_MAGENTO_SYNC=1; shift ;;
        --skip-gitlab-mint) SKIP_GITLAB_MINT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

log() { printf '==> %s\n' "$*" >&2; }
substep() { printf '    %s\n' "$*" >&2; }

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
uv sync --locked
(
    cd "$REPO_ROOT/packages/worldsim-webarena-verified"
    uv lock --check >/dev/null 2>&1 || {
        echo "ERROR: packages/worldsim-webarena-verified lock drift" >&2
        exit 2
    }
    uv sync --locked
)

# ---------------------------------------------------------------------------
# Step 2 — Playwright chromium + system libs (issue #3)
# ---------------------------------------------------------------------------
log "step 2: playwright chromium + system libs"
uv run python -m playwright install chromium
if [[ "$(uname -s)" == "Linux" ]]; then
    sudo "$(command -v uv)" run python -m playwright install-deps chromium || {
        substep "WARN: playwright install-deps failed; ensure libatk-1.0 et al. are present"
    }
else
    substep "non-Linux host: skipping install-deps (macOS / WSL have system libs bundled)"
fi

# ---------------------------------------------------------------------------
# Step 3 — pvpo-chrome container (issue #15)
# ---------------------------------------------------------------------------
if [[ "$SKIP_PVPO_CONTAINER" -eq 0 ]]; then
    log "step 3: pvpo-chrome container"
    DOCKERFILE="$REPO_ROOT/worldsim/docker/chrome-headless-shell.Dockerfile"
    STAMP_FILE="$REPO_ROOT/.pvpo_docker_build_stamp"
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: docker not found; install docker before rerunning" >&2
        exit 2
    fi
    EXPECTED_STAMP="$(sha256sum "$DOCKERFILE" | awk '{print $1}')"
    CURRENT_STAMP=""
    if [[ -f "$STAMP_FILE" ]]; then
        CURRENT_STAMP="$(cat "$STAMP_FILE")"
    fi
    NEED_BUILD=0
    if ! docker image inspect worldsim/chrome-headless-shell:latest >/dev/null 2>&1; then
        NEED_BUILD=1
    elif [[ "$EXPECTED_STAMP" != "$CURRENT_STAMP" ]]; then
        NEED_BUILD=1
        substep "Dockerfile changed since last build (stamp mismatch); rebuilding"
    fi
    if [[ "$NEED_BUILD" -eq 1 ]]; then
        substep "building worldsim/chrome-headless-shell:latest"
        docker build -t worldsim/chrome-headless-shell:latest -f "$DOCKERFILE" "$REPO_ROOT"
        echo "$EXPECTED_STAMP" > "$STAMP_FILE"
    else
        substep "Dockerfile unchanged; skipping rebuild"
    fi
    if ! docker ps --filter name=pvpo-chrome --format '{{.Names}}' | grep -q '^pvpo-chrome$'; then
        # Remove any stopped container with the same name, then start fresh.
        docker rm -f pvpo-chrome >/dev/null 2>&1 || true
        docker run -d --name pvpo-chrome --restart unless-stopped \
            -p 127.0.0.1:9222:9222 worldsim/chrome-headless-shell:latest
    fi
    substep "waiting for CDP endpoint at 127.0.0.1:9222"
    deadline=$((SECONDS + 30))
    until curl -fsS http://127.0.0.1:9222/json/version >/dev/null 2>&1; do
        if (( SECONDS > deadline )); then
            echo "ERROR: pvpo-chrome CDP endpoint did not respond within 30s" >&2
            docker logs pvpo-chrome | tail -40 >&2 || true
            exit 1
        fi
        sleep 0.5
    done
    substep "pvpo-chrome ready at 127.0.0.1:9222"
    substep "export WORLDSIM_PVPO_CDP_URL=http://127.0.0.1:9222 before launching Phase 4"
fi

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

# ---------------------------------------------------------------------------
# Step 5 — Mint Phase 0d storage_state for gitlab (issue #7)
# ---------------------------------------------------------------------------
if [[ "$SKIP_GITLAB_MINT" -eq 0 ]]; then
    log "step 5: mint gitlab Phase 0d storage_state"
    STATE_FILE="$STATE_DIR/phase_0d/gitlab/storage_state.json"
    if [[ -s "$STATE_FILE" ]]; then
        substep "$STATE_FILE already present; skipping mint"
    else
        GITLAB_HOST="${GITLAB_HOST:-http://127.0.0.1:8023}" \
            GITLAB_STORAGE_STATE_PATH="$STATE_FILE" \
            uv run python "$REPO_ROOT/scripts/login_gitlab_r5.py"
    fi
fi

# ---------------------------------------------------------------------------
# Step 6 — Magento base_url sync across every replica (issues #12, #13)
# ---------------------------------------------------------------------------
if [[ "$SKIP_MAGENTO_SYNC" -eq 0 ]]; then
    log "step 6: sync Magento base_url across every shopping replica"
    if [[ ! -f "$REPO_ROOT/$INSTANCES" ]]; then
        echo "ERROR: instances file not found: $INSTANCES" >&2
        exit 2
    fi
    uv run python "$REPO_ROOT/scripts/sync_magento_base_urls.py" \
        --instances "$REPO_ROOT/$INSTANCES" \
        --verify-after \
        --retry-on-revert 2
fi

# ---------------------------------------------------------------------------
# Step 7 — Preflight via pytest
# ---------------------------------------------------------------------------
log "step 7: preflight (pytest -m preflight)"
PYTEST_ARGS=(-m preflight tests/preflight -v)
if [[ -n "$HOST_CONFIG" ]]; then
    export WORLDSIM_PREFLIGHT_HOST_CONFIG="$REPO_ROOT/$HOST_CONFIG"
fi
export WORLDSIM_PREFLIGHT_INSTANCES="$REPO_ROOT/$INSTANCES"
uv run pytest "${PYTEST_ARGS[@]}"

log "Phase 4 setup complete. Next: export WORLDSIM_PVPO_CDP_URL=http://127.0.0.1:9222 && uv run python -m worldsim.main phase 4 --benchmark $INSTANCES --resume"
