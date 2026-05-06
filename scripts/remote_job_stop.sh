#!/usr/bin/env bash
# Stop a registered remote WorldSim job by job id only.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/remote_jobs.sh
source "$REPO_ROOT/scripts/lib/remote_jobs.sh"

HOST_CONFIG=""
REMOTE_DIR=""
SSH_KEY_ARG=""
JOB_ID=""
FORCE=0
TIMEOUT=20

usage() {
    cat <<'USAGE'
remote_job_stop.sh

Options:
  --host-config <path>      benchmark host YAML (required)
  --remote-dir <path>       remote checkout dir (default: <compose_dir_remote>/browser-sim)
  --ssh-key <path>          SSH private key (default: $SSH_KEY or ~/.ssh/webarena-key.pem)
  --job-id <id>             job id to stop (required)
  --force                   send KILL after TERM timeout
  --timeout <seconds>       TERM wait before optional KILL (default: 20)
  -h, --help                show this help

Stop is intentionally job-id only. It never accepts command patterns.
USAGE
}

while (($#)); do
    case "$1" in
        --host-config) HOST_CONFIG="$2"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --ssh-key) SSH_KEY_ARG="$2"; shift 2 ;;
        --job-id) JOB_ID="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) rj_die "unknown arg: $1" ;;
    esac
done

[[ -n "$HOST_CONFIG" ]] || { usage >&2; rj_die "--host-config required"; }
[[ -n "$JOB_ID" ]] || { usage >&2; rj_die "--job-id required; command patterns are not accepted"; }
[[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]] || rj_die "--timeout must be a positive integer"

rj_prepare_connection "$HOST_CONFIG" "$SSH_KEY_ARG"
REMOTE_DIR="${REMOTE_DIR:-$(rj_default_remote_dir)}"

rj_ssh_bash "$REMOTE_DIR" "$JOB_ID" "$FORCE" "$TIMEOUT" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
job_id="$2"
force="$3"
timeout="$4"
python3 - "$remote_dir" "$job_id" "$force" "$timeout" <<'PY'
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

remote_dir = Path(sys.argv[1]).resolve()
job_id = sys.argv[2]
force = sys.argv[3] == "1"
timeout = int(sys.argv[4])
job_dir = remote_dir / "logs" / "remote_jobs" / job_id
metadata_path = job_dir / "metadata.json"
if not metadata_path.exists():
    raise SystemExit(f"metadata not found: {metadata_path}")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
pid = int(metadata.get("pid") or (job_dir / "pid").read_text().strip())
pgid = int(metadata.get("pgid") or (job_dir / "pgid").read_text().strip())
fingerprint = str(metadata.get("command_fingerprint") or "")
start_ticks = str(metadata.get("process_start_ticks") or "").strip()
proc = Path("/proc") / str(pid)

def now():
    return datetime.now(timezone.utc).isoformat()

def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def alive() -> bool:
    if not proc.exists():
        return False
    try:
        return proc.joinpath("stat").read_text().split()[2] != "Z"
    except Exception:
        return True

if not alive():
    write_json(job_dir / "stop.json", {"status": "already_exited", "updated_at": now(), "pid": pid, "pgid": pgid})
    print(f"job {job_id} is not running")
    raise SystemExit(0)

if start_ticks:
    actual_start = proc.joinpath("stat").read_text().split()[21]
    if actual_start != start_ticks:
        raise SystemExit("refusing to stop: pid start time no longer matches metadata")

cmdline = proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
if "run_job.py" not in cmdline or fingerprint not in cmdline:
    raise SystemExit("refusing to stop: pid command line does not match job metadata")

try:
    cwd = Path(os.readlink(proc / "cwd")).resolve()
except OSError as exc:
    raise SystemExit(f"refusing to stop: could not read process cwd: {exc}") from exc
if cwd != remote_dir:
    raise SystemExit(f"refusing to stop: process cwd {cwd} != {remote_dir}")

actual_pgid = os.getpgid(pid)
if actual_pgid != pgid:
    raise SystemExit(f"refusing to stop: process group {actual_pgid} != metadata {pgid}")

write_json(job_dir / "stop.json", {"status": "term_sent", "updated_at": now(), "pid": pid, "pgid": pgid})
os.killpg(pgid, signal.SIGTERM)
deadline = time.time() + timeout
while time.time() < deadline:
    if not alive():
        write_json(job_dir / "exit.json", {"status": "stopped", "updated_at": now(), "pid": pid, "pgid": pgid, "signal": "TERM"})
        print(f"stopped job {job_id} with TERM")
        raise SystemExit(0)
    time.sleep(0.5)

if not force:
    write_json(job_dir / "stop.json", {"status": "term_timeout", "updated_at": now(), "pid": pid, "pgid": pgid})
    raise SystemExit(f"TERM timeout after {timeout}s; rerun with --force to send KILL")

os.killpg(pgid, signal.SIGKILL)
deadline = time.time() + 5
while time.time() < deadline:
    if not alive():
        write_json(job_dir / "exit.json", {"status": "killed", "updated_at": now(), "pid": pid, "pgid": pgid, "signal": "KILL"})
        print(f"killed job {job_id} with KILL")
        raise SystemExit(0)
    time.sleep(0.5)
write_json(job_dir / "stop.json", {"status": "kill_timeout", "updated_at": now(), "pid": pid, "pgid": pgid})
raise SystemExit("KILL timeout after 5s; process group still appears alive")
PY
REMOTE
