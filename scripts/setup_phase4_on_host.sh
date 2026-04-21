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
  --skip-magento-sync        deprecated no-op (step 6 removed 2026-04-21)
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
# Playwright is only a dep when browser-use is pinned below 0.12.x. browser-use
# 0.12.6+ talks CDP natively, and PVPO runs in a chrome-headless-shell Docker
# container (step 3), so neither path needs a host-side Playwright install. Skip
# gracefully if the module is absent — loud failure on an optional dep is noise.
log "step 2: playwright chromium + system libs"
if uv run python -c "import playwright" >/dev/null 2>&1; then
    uv run python -m playwright install chromium
    if [[ "$(uname -s)" == "Linux" ]]; then
        sudo "$(command -v uv)" run python -m playwright install-deps chromium
    else
        substep "non-Linux host: skipping install-deps (macOS / WSL have system libs bundled)"
    fi
else
    substep "playwright not installed (browser-use 0.12.6+ uses CDP + docker chrome-headless-shell); skipping"
fi

# ---------------------------------------------------------------------------
# Step 3 — pvpo-chrome containers (issue #15)
# ---------------------------------------------------------------------------
if [[ "$SKIP_PVPO_CONTAINER" -eq 0 ]]; then
    log "step 3: pvpo-chrome containers"
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
    if [[ ! -f "$REPO_ROOT/$INSTANCES" ]]; then
        echo "ERROR: instances file not found: $INSTANCES" >&2
        exit 2
    fi
    mapfile -t PVPO_PORTS < <(
        uv run python - "$REPO_ROOT/$INSTANCES" <<'PY'
import json
import sys
from urllib.parse import urlparse

instances_path = sys.argv[1]
with open(instances_path, encoding="utf-8") as handle:
    payload = json.load(handle)

ports: set[int] = set()
missing_labels: list[str] = []
for index, instance in enumerate(payload.get("instances", [])):
    raw_url = str(instance.get("pvpo_cdp_url") or "").strip()
    if not raw_url:
        site_name = str(instance.get("site_name") or f"instance[{index}]").strip() or f"instance[{index}]"
        replica_index = instance.get("replica_index")
        if replica_index is None:
            missing_labels.append(site_name)
        else:
            missing_labels.append(f"{site_name}[{replica_index}]")
        continue
    parsed = urlparse(raw_url)
    host = (parsed.hostname or "").strip().lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit(
            f"setup_phase4_on_host.sh only manages loopback pvpo_cdp_url entries; got {raw_url!r}"
        )
    if parsed.port is None:
        raise SystemExit(f"pvpo_cdp_url must include an explicit port: {raw_url!r}")
    ports.add(parsed.port)

if not ports:
    raise SystemExit("instances file has no pvpo_cdp_url entries; populate one endpoint per instance")
if missing_labels:
    missing_display = ", ".join(sorted(missing_labels))
    raise SystemExit(
        f"instances missing pvpo_cdp_url: {missing_display}. Populate one dedicated endpoint per instance"
    )

for port in sorted(ports):
    print(port)
PY
    )
    for port in "${PVPO_PORTS[@]}"; do
        name="pvpo-chrome-$port"
        internal_port="$((port + 100))"
        network_mode="$(docker inspect --format '{{.HostConfig.NetworkMode}}' "$name" 2>/dev/null || true)"
        external_env="$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$name" 2>/dev/null | grep '^PVPO_EXTERNAL_PORT=' | tail -1 | cut -d= -f2 || true)"
        internal_env="$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$name" 2>/dev/null | grep '^PVPO_INTERNAL_PORT=' | tail -1 | cut -d= -f2 || true)"
        if ! docker ps --filter "name=^/${name}$" --format '{{.Names}}' | grep -q "^${name}$" \
            || [[ "$network_mode" != "host" ]] \
            || [[ "$external_env" != "$port" ]] \
            || [[ "$internal_env" != "$internal_port" ]]; then
            docker rm -f "$name" >/dev/null 2>&1 || true
            docker run -d --name "$name" --restart unless-stopped \
                --network host \
                -e "PVPO_EXTERNAL_PORT=${port}" \
                -e "PVPO_INTERNAL_PORT=${internal_port}" \
                worldsim/chrome-headless-shell:latest
        fi
        substep "waiting for CDP endpoint at 127.0.0.1:${port}"
        deadline=$((SECONDS + 30))
        until curl -fsS "http://127.0.0.1:${port}/json/version" >/dev/null 2>&1; do
            if (( SECONDS > deadline )); then
                echo "ERROR: pvpo-chrome CDP endpoint on port ${port} did not respond within 30s" >&2
                docker logs "$name" | tail -40 >&2 || true
                exit 1
            fi
            sleep 0.5
        done
        substep "pvpo-chrome ready at 127.0.0.1:${port}"
    done
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

# Step 6 (Magento base_url sync) was removed 2026-04-21 with the
# WASP-aligned scoping decision.

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

log "Phase 4 setup complete. Next: uv run python -m worldsim.main phase 4 --instances $INSTANCES --resume"
