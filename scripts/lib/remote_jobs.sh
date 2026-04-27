#!/usr/bin/env bash
# Shared helpers for remote WorldSim job scripts.

set -euo pipefail

RJ_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RJ_REPO_ROOT="$(cd "$RJ_LIB_DIR/../.." && pwd)"

rj_die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

rj_require_cmd() {
    command -v "$1" >/dev/null 2>&1 || rj_die "required command not found: $1"
}

rj_abs_path() {
    python3 - "$1" <<'PY'
import os
import sys
from pathlib import Path

raw = os.path.expandvars(os.path.expanduser(sys.argv[1]))
print(Path(raw).resolve(strict=False))
PY
}

rj_shell_quote() {
    python3 - "$1" <<'PY'
import shlex
import sys

print(shlex.quote(sys.argv[1]))
PY
}

rj_json_b64() {
    python3 - "$@" <<'PY'
import base64
import json
import sys

payload = list(sys.argv[1:])
print(base64.b64encode(json.dumps(payload).encode()).decode())
PY
}

rj_object_json_b64() {
    python3 - "$@" <<'PY'
import base64
import json
import sys

payload = json.loads(sys.argv[1])
print(base64.b64encode(json.dumps(payload).encode()).decode())
PY
}

rj_load_host_config() {
    local host_config="$1"
    [[ -n "$host_config" ]] || rj_die "--host-config required"
    [[ -f "$host_config" ]] || rj_die "host config not found: $host_config"

    # Keep this dependency-free for operator scripts. The checked-in host YAML
    # fields used here are simple top-level scalars.
    local exports
    exports="$(python3 - "$host_config" <<'PY'
import shlex
import sys
from pathlib import Path

values = {}
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.split("#", 1)[0].rstrip()
    if not line or line.startswith(" ") or ":" not in line:
        continue
    key, value = line.split(":", 1)
    values[key.strip()] = value.strip().strip("'\"")

host = values.get("advertise_host") or values.get("orchestrator_host")
if not host:
    raise SystemExit("host config missing advertise_host/orchestrator_host")

out = {
    "RJ_HOST_IP": host,
    "RJ_SSH_USER": values.get("ssh_user") or "ubuntu",
    "RJ_COMPOSE_DIR_REMOTE": values.get("compose_dir_remote") or "/home/ubuntu",
}
for key, value in out.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"
    eval "$exports"
}

rj_prepare_connection() {
    local host_config="$1"
    local ssh_key_raw="${2:-}"
    rj_load_host_config "$host_config"

    if [[ -z "$ssh_key_raw" ]]; then
        ssh_key_raw="${SSH_KEY:-$HOME/.ssh/webarena-key.pem}"
    fi
    RJ_SSH_KEY="$(rj_abs_path "$ssh_key_raw")"
    RJ_SSH_BIN="${REMOTE_JOBS_SSH_BIN:-ssh}"
    RJ_SCP_BIN="${REMOTE_JOBS_SCP_BIN:-scp}"
    RJ_RSYNC_BIN="${REMOTE_JOBS_RSYNC_BIN:-rsync}"
    RJ_SSH_TARGET="${RJ_SSH_USER}@${RJ_HOST_IP}"
    RJ_SSH_OPTS=(
        -i "$RJ_SSH_KEY"
        -o StrictHostKeyChecking=accept-new
        -o ServerAliveInterval=30
        -o ServerAliveCountMax=120
        -o ConnectTimeout=15
    )
}

rj_default_remote_dir() {
    printf '%s/browser-sim\n' "$RJ_COMPOSE_DIR_REMOTE"
}

rj_ssh() {
    "$RJ_SSH_BIN" "${RJ_SSH_OPTS[@]}" "$RJ_SSH_TARGET" "$@"
}

rj_ssh_bash() {
    local remote_cmd
    remote_cmd='tmp=$(mktemp /tmp/worldsim-remote-job.XXXXXX); cat > "$tmp"; bash "$tmp"'
    local arg
    for arg in "$@"; do
        remote_cmd+=" $(rj_shell_quote "$arg")"
    done
    remote_cmd+='; rc=$?; rm -f "$tmp"; exit "$rc"'
    "$RJ_SSH_BIN" "${RJ_SSH_OPTS[@]}" "$RJ_SSH_TARGET" "$remote_cmd"
}

rj_resolve_job_id() {
    local remote_dir="$1"
    local job_id="$2"
    local latest="$3"
    local name="$4"

    if [[ -n "$job_id" ]]; then
        printf '%s\n' "$job_id"
        return
    fi

    rj_ssh_bash "$remote_dir" "$latest" "$name" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
latest="$2"
name="$3"
python3 - "$remote_dir" "$latest" "$name" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "logs" / "remote_jobs"
latest = sys.argv[2] == "1"
name = sys.argv[3]

if not root.exists():
    raise SystemExit("no remote job registry found")

jobs = []
for metadata_path in root.glob("*/metadata.json"):
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if name and data.get("name") != name:
        continue
    jobs.append((metadata_path.stat().st_mtime, metadata_path.parent.name))

if not jobs:
    if name:
        raise SystemExit(f"no remote jobs found for name: {name}")
    raise SystemExit("no remote jobs found")

jobs.sort(reverse=True)
print(jobs[0][1])
PY
REMOTE
}

rj_validate_job_selector() {
    local job_id="$1"
    local latest="$2"
    local name="$3"
    local allow_name="${4:-1}"
    local count=0
    [[ -n "$job_id" ]] && count=$((count + 1))
    [[ "$latest" -eq 1 ]] && count=$((count + 1))
    [[ -n "$name" ]] && count=$((count + 1))
    [[ "$allow_name" -eq 0 && -n "$name" ]] && rj_die "--name is not supported here; use --job-id"
    [[ "$count" -eq 1 ]] || rj_die "select exactly one of --job-id, --latest, or --name"
}
