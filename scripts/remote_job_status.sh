#!/usr/bin/env bash
# Show status for a registered remote WorldSim job.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/remote_jobs.sh
source "$REPO_ROOT/scripts/lib/remote_jobs.sh"

HOST_CONFIG=""
REMOTE_DIR=""
SSH_KEY_ARG=""
JOB_ID=""
LATEST=0
NAME=""

usage() {
    cat <<'USAGE'
remote_job_status.sh

Options:
  --host-config <path>      benchmark host YAML (required)
  --remote-dir <path>       remote checkout dir (default: <compose_dir_remote>/browser-sim)
  --ssh-key <path>          SSH private key (default: $SSH_KEY or ~/.ssh/webarena-key.pem)
  --job-id <id>             job id to inspect
  --latest                  inspect latest job
  --name <name>             inspect latest job with this human name
  -h, --help                show this help
USAGE
}

while (($#)); do
    case "$1" in
        --host-config) HOST_CONFIG="$2"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --ssh-key) SSH_KEY_ARG="$2"; shift 2 ;;
        --job-id) JOB_ID="$2"; shift 2 ;;
        --latest) LATEST=1; shift ;;
        --name) NAME="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) rj_die "unknown arg: $1" ;;
    esac
done

[[ -n "$HOST_CONFIG" ]] || { usage >&2; rj_die "--host-config required"; }
rj_validate_job_selector "$JOB_ID" "$LATEST" "$NAME"
rj_prepare_connection "$HOST_CONFIG" "$SSH_KEY_ARG"
REMOTE_DIR="${REMOTE_DIR:-$(rj_default_remote_dir)}"
JOB_ID="$(rj_resolve_job_id "$REMOTE_DIR" "$JOB_ID" "$LATEST" "$NAME")"

rj_ssh_bash "$REMOTE_DIR" "$JOB_ID" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
job_id="$2"
python3 - "$remote_dir" "$job_id" <<'PY'
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

remote_dir = Path(sys.argv[1])
job_id = sys.argv[2]
job_dir = remote_dir / "logs" / "remote_jobs" / job_id
metadata_path = job_dir / "metadata.json"
if not metadata_path.exists():
    raise SystemExit(f"metadata not found: {metadata_path}")

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
pid = int(metadata.get("pid") or (job_dir / "pid").read_text().strip())
start_ticks = str(metadata.get("process_start_ticks") or "").strip()
proc = Path("/proc") / str(pid)
proc_state = ""
if proc.exists():
    try:
        proc_state = proc.joinpath("stat").read_text().split()[2]
    except Exception:
        proc_state = ""
alive = proc.exists() and proc_state != "Z"
start_matches = False
if alive and start_ticks:
    try:
        start_matches = proc.joinpath("stat").read_text().split()[21] == start_ticks
    except Exception:
        start_matches = False
elif alive:
    start_matches = True

exit_path = job_dir / "exit.json"
exit_data = None
if exit_path.exists():
    try:
        exit_data = json.loads(exit_path.read_text(encoding="utf-8"))
    except Exception:
        exit_data = {"status": "unparseable"}

if alive and start_matches:
    status = "running"
elif exit_data:
    status = "exited"
else:
    status = "stale"

created_raw = metadata.get("created_at")
elapsed = "unknown"
try:
    created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
    seconds = int((datetime.now(timezone.utc) - created).total_seconds())
    elapsed = f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"
except Exception:
    pass

def tail(path: Path, lines: int = 8) -> list[str]:
    if not path.exists():
        return []
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return data[-lines:]

stdout = job_dir / "stdout.log"
stderr = job_dir / "stderr.log"
latest_log_mtime = 0.0
for path in (stdout, stderr):
    if path.exists():
        latest_log_mtime = max(latest_log_mtime, path.stat().st_mtime)

print(f"job_id: {job_id}")
print(f"name: {metadata.get('name')}")
print(f"status: {status}")
if exit_data and "returncode" in exit_data:
    print(f"returncode: {exit_data['returncode']}")
print(f"elapsed: {elapsed}")
print(f"pid: {pid}")
print(f"pgid: {metadata.get('pgid')}")
print(f"remote_dir: {metadata.get('remote_dir')}")
print(f"metadata: {metadata_path}")
print("command: " + " ".join(metadata.get("command", [])[:12]) + (" ..." if len(metadata.get("command", [])) > 12 else ""))

def print_git(label: str, payload) -> str | None:
    if not isinstance(payload, dict):
        return None
    sha = payload.get("sha")
    branch = payload.get("branch") or "unknown"
    dirty = "dirty" if payload.get("dirty") else "clean"
    short_sha = str(sha)[:12] if sha else "unknown"
    print(f"{label}: {short_sha} branch={branch} worktree={dirty}")
    return str(sha) if sha else None

local_sha = print_git("local_git", metadata.get("local_git"))
remote_sha = print_git("remote_git", metadata.get("remote_git"))
sync_stamp = metadata.get("remote_sync_stamp")
sync_sha = None
if isinstance(sync_stamp, dict):
    stamp_git = sync_stamp.get("local_git") if isinstance(sync_stamp.get("local_git"), dict) else {}
    sync_sha = stamp_git.get("sha")
    short_sync_sha = str(sync_sha)[:12] if sync_sha else "unknown"
    stamp_branch = stamp_git.get("branch") or "unknown"
    stamp_dirty = "dirty" if stamp_git.get("dirty") else "clean"
    print(
        "remote_sync_stamp: "
        f"{short_sync_sha} branch={stamp_branch} worktree={stamp_dirty} "
        f"synced_at={sync_stamp.get('synced_at') or 'unknown'}"
    )
if local_sha and sync_sha and local_sha != str(sync_sha):
    print("warning: local_git and remote_sync_stamp differ; rerun sync_to_r5.sh before trusting this job")
elif local_sha and remote_sha and local_sha != remote_sha and not sync_sha:
    print("warning: local_git and remote_git differ; verify the intended code was synced before trusting this job")
elif remote_sha and sync_sha and remote_sha != str(sync_sha):
    print("note: remote_git differs from remote_sync_stamp because sync_to_r5.sh excludes .git; use remote_sync_stamp for deployed code provenance")

if status == "running" and latest_log_mtime:
    quiet_for = int(time.time() - latest_log_mtime)
    print(f"log_progress: latest write {quiet_for}s ago")
    if quiet_for > 900:
        print("warning: process is alive, but logs have not changed for more than 15 minutes")

expected = metadata.get("expected_outputs") or []
if expected:
    print("expected_outputs:")
    for rel in expected:
        path = remote_dir / rel
        print(f"  {'present' if path.exists() else 'missing'} {rel}")

print(f"stdout: {stdout}")
for line in tail(stdout):
    print(f"  {line}")
print(f"stderr: {stderr}")
for line in tail(stderr):
    print(f"  {line}")
PY
REMOTE
