#!/usr/bin/env bash
# Sync the local checkout to a benchmark host without secrets or generated logs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/remote_jobs.sh
source "$REPO_ROOT/scripts/lib/remote_jobs.sh"

HOST_CONFIG=""
REMOTE_DIR=""
SSH_KEY_ARG=""
DRY_RUN=0

usage() {
    cat <<'USAGE'
sync_to_r5.sh

Options:
  --host-config <path>      benchmark host YAML (required)
  --remote-dir <path>       remote checkout dir (default: <compose_dir_remote>/browser-sim)
  --ssh-key <path>          SSH private key (default: $SSH_KEY or ~/.ssh/webarena-key.pem)
  --dry-run                 print rsync changes without writing
  -h, --help                show this help
USAGE
}

while (($#)); do
    case "$1" in
        --host-config) HOST_CONFIG="$2"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --ssh-key) SSH_KEY_ARG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) rj_die "unknown arg: $1" ;;
    esac
done

[[ -n "$HOST_CONFIG" ]] || { usage >&2; rj_die "--host-config required"; }
rj_prepare_connection "$HOST_CONFIG" "$SSH_KEY_ARG"
REMOTE_DIR="${REMOTE_DIR:-$(rj_default_remote_dir)}"

excludes=(
    ".env"
    ".env.*"
    ".git"
    ".git/"
    ".venv/"
    ".m5_instance_id"
    ".benchmark_host_id"
    ".benchmark_topology"
    ".proxy_token"
    ".benchmark_proxy_metadata"
    ".benchmark_proxy_ports.conf"
    ".codex-worktrees/"
    "logs/"
    "logs_*/"
    "logs_run*/"
    "tmp/"
    ".pytest_cache/"
    ".ruff_cache/"
    "__pycache__/"
    "*.pyc"
    "node_modules/"
    "agent-tools/"
    "*secret*"
    "*credentials*"
    "*.pem"
    "*.key"
    "id_rsa*"
    "id_ed25519*"
)

if [[ -f "$REPO_ROOT/.git" ]]; then
    printf 'Source checkout uses a linked-worktree .git file; excluding Git metadata from rsync.\n' >&2
fi

rsync_args=(-az --delete)
if [[ "$DRY_RUN" -eq 1 ]]; then
    rsync_args+=(--dry-run --itemize-changes)
fi
for pattern in "${excludes[@]}"; do
    rsync_args+=(--exclude "$pattern")
done

ssh_cmd=("$RJ_SSH_BIN" "${RJ_SSH_OPTS[@]}")
printf 'Syncing %s -> %s:%s\n' "$REPO_ROOT/" "$RJ_SSH_TARGET" "$REMOTE_DIR" >&2
printf 'Using SSH key: %s\n' "$RJ_SSH_KEY" >&2

"$RJ_RSYNC_BIN" "${rsync_args[@]}" \
    -e "$(printf '%q ' "${ssh_cmd[@]}")" \
    "$REPO_ROOT/" \
    "$RJ_SSH_TARGET:$REMOTE_DIR/"
