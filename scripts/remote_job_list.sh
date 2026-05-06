#!/usr/bin/env bash
# List registered remote WorldSim jobs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/remote_jobs.sh
source "$REPO_ROOT/scripts/lib/remote_jobs.sh"

HOST_CONFIG=""
REMOTE_DIR=""
SSH_KEY_ARG=""
NAME=""
LIMIT=20
JSON=0

usage() {
    cat <<'USAGE'
remote_job_list.sh

Options:
  --host-config <path>      benchmark host YAML (required)
  --remote-dir <path>       remote checkout dir (default: <compose_dir_remote>/browser-sim)
  --ssh-key <path>          SSH private key (default: $SSH_KEY or ~/.ssh/webarena-key.pem)
  --name <name>             only list jobs with this human name
  --limit <n>               max rows (default: 20)
  --json                    print machine-readable JSON rows
  -h, --help                show this help
USAGE
}

while (($#)); do
    case "$1" in
        --host-config) HOST_CONFIG="$2"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --ssh-key) SSH_KEY_ARG="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --json) JSON=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) rj_die "unknown arg: $1" ;;
    esac
done

[[ -n "$HOST_CONFIG" ]] || { usage >&2; rj_die "--host-config required"; }
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || rj_die "--limit must be a positive integer"
rj_prepare_connection "$HOST_CONFIG" "$SSH_KEY_ARG"
REMOTE_DIR="${REMOTE_DIR:-$(rj_default_remote_dir)}"

rj_ssh_bash "$REMOTE_DIR" "$NAME" "$LIMIT" "$JSON" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
name="$2"
limit="$3"
json_mode="$4"
python3 - "$remote_dir" "$name" "$limit" "$json_mode" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "logs" / "remote_jobs"
name = sys.argv[2]
limit = int(sys.argv[3])
json_mode = sys.argv[4] == "1"
if not root.exists():
    if json_mode:
        print(json.dumps({"jobs": []}, sort_keys=True))
        raise SystemExit(0)
    print("no remote job registry found")
    raise SystemExit(0)

rows = []
for metadata_path in root.glob("*/metadata.json"):
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if name and data.get("name") != name:
        continue
    status = "unknown"
    exit_path = metadata_path.parent / "exit.json"
    if exit_path.exists():
        status = "exited"
    elif data.get("pid") and Path("/proc", str(data["pid"])).exists():
        try:
            state = Path("/proc", str(data["pid"]), "stat").read_text().split()[2]
        except Exception:
            status = "unknown"
        else:
            status = "stale" if state == "Z" else "running"
    else:
        status = "stale"
    rows.append({
        "mtime": metadata_path.stat().st_mtime,
        "created_at": data.get("created_at", ""),
        "job_id": data.get("job_id", metadata_path.parent.name),
        "name": data.get("name", ""),
        "status": status,
        "state_dir": data.get("state_dir"),
        "expected_outputs": data.get("expected_outputs") or [],
    })

rows.sort(key=lambda row: row["mtime"], reverse=True)
rows = rows[:limit]
if json_mode:
    print(json.dumps({"jobs": rows}, indent=2, sort_keys=True))
    raise SystemExit(0)
print("created_at status  name  job_id")
for row in rows:
    print(f"{row['created_at']} {row['status']:7} {row['name']} {row['job_id']}")
PY
REMOTE
