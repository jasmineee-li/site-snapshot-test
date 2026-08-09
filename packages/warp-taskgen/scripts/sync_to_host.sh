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
ALLOW_ACTIVE_JOBS=0

usage() {
    cat <<'USAGE'
sync_to_host.sh

Options:
  --host-config <path>      benchmark host YAML (required)
  --remote-dir <path>       remote checkout dir (default: <compose_dir_remote>/browser-sim)
  --ssh-key <path>          SSH private key (default: $SSH_KEY or ~/.ssh/webarena-key.pem)
  --dry-run                 print rsync changes without writing
  --allow-active-jobs       sync even if remote_jobs registry has running jobs
  -h, --help                show this help
USAGE
}

while (($#)); do
    case "$1" in
        --host-config) HOST_CONFIG="$2"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --ssh-key) SSH_KEY_ARG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --allow-active-jobs) ALLOW_ACTIVE_JOBS=1; shift ;;
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
    ".venv.*"
    ".venv.*/"
    ".m5_instance_id"
    ".benchmark_host_id"
    ".benchmark_topology"
    ".proxy_token"
    ".benchmark_proxy_metadata"
    ".benchmark_proxy_ports.conf"
    "instances.scale.json"
    "instances.scale.json.fragment"
    "instances.smoke.json"
    "instances.smoke.json.fragment"
    "instances.smoke.local.json"
    "instances.json"
    "configs/benchmark_hosts/*.local.yaml"
    "configs/benchmark_hosts/r5.yaml"
    "compose.scale.yml"
    "compose.smoke.yml"
    "scripts/docker-compose.scale.yml"
    "scripts/docker-compose.smoke.yml"
    "scripts/proxy_ports.conf"
    ".codex-worktrees/"
    ".cursor/"
    "docs/handoffs/codex-handoff-*.md"
    ".claude/worktrees/"
    ".claude/local.md"
    ".claude/settings.local.json"
    "AgentLab/"
    # Every vendor checkout is host-local and gitignored. A selective list
    # lets rsync --delete erase newly added benchmark/vendor trees.
    "vendors/"
    "logs/"
    "logs_*/"
    "logs_run*/"
    "pipeline_outputs/"
    # Runtime datasets and host-local build/operator artifacts must survive
    # rsync --delete on the execution host; the package tracks only data/.gitkeeps.
    "data/"
    "CODEX.local.md"
    ".cache/"
    "dist/"
    ".DS_Store"
    ".worldsim_sync_stamp.json"
    "*.sqlite"
    "*.sqlite3"
    "reports/"
    "tmp/"
    "scripts/smoke_phase_*.py"
    ".modal/"
    ".uv-cache/"
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

guarded_untracked=(
    ".uv-cache"
    ".modal"
    "pipeline_outputs"
    "reports"
    "tmp"
    "*.sqlite"
    "*.sqlite3"
    "scripts/smoke_phase_*.py"
)

if [[ -f "$REPO_ROOT/.git" ]]; then
    printf 'Source checkout uses a linked-worktree .git file; excluding Git metadata from rsync.\n' >&2
fi

if [[ "$DRY_RUN" -eq 0 && "$ALLOW_ACTIVE_JOBS" -eq 0 && "${WORLDSIM_ALLOW_SYNC_DURING_ACTIVE_JOBS:-}" != "1" ]]; then
    rj_ssh_bash "$REMOTE_DIR" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
python3 - "$remote_dir" <<'PY'
import json
from pathlib import Path

remote_dir = Path(__import__("sys").argv[1])
root = remote_dir / "logs" / "remote_jobs"
active: list[str] = []
proc_root = Path("/proc")

if root.exists():
    for metadata_path in root.glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        job_dir = metadata_path.parent
        exit_path = job_dir / "exit.json"
        if exit_path.exists():
            continue
        heartbeat_path = job_dir / "heartbeat.json"
        if not proc_root.exists() and heartbeat_path.exists():
            try:
                heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
            except Exception:
                heartbeat = {}
            if heartbeat.get("status") == "running":
                name = metadata.get("name") or "unknown"
                active.append(f"{job_dir.name} name={name} pid={metadata.get('pid', 'unknown')}")
                continue
        try:
            pid = int(metadata.get("pid") or (job_dir / "pid").read_text().strip())
        except Exception:
            continue
        proc = proc_root / str(pid)
        if not proc.exists():
            continue
        try:
            if proc.joinpath("stat").read_text().split()[2] == "Z":
                continue
        except Exception:
            continue
        start_ticks = str(metadata.get("process_start_ticks") or "").strip()
        if start_ticks:
            try:
                if proc.joinpath("stat").read_text().split()[21] != start_ticks:
                    continue
            except Exception:
                continue
        name = metadata.get("name") or "unknown"
        active.append(f"{job_dir.name} name={name} pid={pid}")

if active:
    print("sync guard blocked: remote WorldSim jobs are still running.", file=__import__("sys").stderr)
    print("Syncing now can mix code versions inside chained phase jobs.", file=__import__("sys").stderr)
    for item in active[:10]:
        print(f"- {item}", file=__import__("sys").stderr)
    print(
        "Use scripts/remote_job_status.sh/stop.sh first, or pass --allow-active-jobs "
        "only for deliberate maintenance.",
        file=__import__("sys").stderr,
    )
    raise SystemExit(2)
PY
REMOTE
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    untracked_blockers=()
    while IFS= read -r path; do
        [[ -n "$path" ]] || continue
        for pattern in "${guarded_untracked[@]}"; do
            if [[ "$path" == $pattern || "$path" == $pattern/* ]]; then
                untracked_blockers+=("$path")
                break
            fi
        done
    done < <(git -C "$REPO_ROOT" ls-files --others --exclude-standard)

    if ((${#untracked_blockers[@]})); then
        printf 'sync guard blocked: local untracked scratch paths are present.\n' >&2
        printf 'Direct sync is intentionally fail-closed so scratch artifacts do not reach r5.\n' >&2
        printf 'Clean, move, or intentionally ignore these paths before syncing:\n' >&2
        printf '  - %s\n' "${untracked_blockers[@]:0:20}" >&2
        if ((${#untracked_blockers[@]} > 20)); then
            printf '  ... %d more\n' "$((${#untracked_blockers[@]} - 20))" >&2
        fi
        exit 2
    fi
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

if [[ "$DRY_RUN" -eq 0 ]]; then
    SYNC_STAMP_B64="$(python3 - "$REPO_ROOT" "$HOST_CONFIG" <<'PY'
import base64
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

repo = Path(sys.argv[1])
host_config = sys.argv[2]

def git_value(args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None

payload = {
    "synced_at": datetime.now(timezone.utc).isoformat(),
    "host_config": host_config,
    "source": str(repo),
    "git_metadata_excluded": True,
    "local_git": {
        "sha": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(git_value(["status", "--porcelain"])),
    },
}
print(base64.b64encode(json.dumps(payload).encode()).decode())
PY
)"
    rj_ssh_bash "$REMOTE_DIR" "$SYNC_STAMP_B64" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
sync_stamp_b64="$2"
python3 - "$remote_dir" "$sync_stamp_b64" <<'PY'
import base64
import json
import sys
from pathlib import Path

remote_dir = Path(sys.argv[1])
payload = json.loads(base64.b64decode(sys.argv[2]).decode())
path = remote_dir / ".worldsim_sync_stamp.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
REMOTE
fi
