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
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

remote_dir = Path(sys.argv[1])
job_id = sys.argv[2]
job_dir = remote_dir / "logs" / "remote_jobs" / job_id
metadata_path = job_dir / "metadata.json"
if not metadata_path.exists():
    raise SystemExit(f"metadata not found: {metadata_path}")

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
try:
    job_created_ts = datetime.fromisoformat(
        str(metadata.get("created_at", "")).replace("Z", "+00:00")
    ).timestamp()
except Exception:
    job_created_ts = None
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
    exit_status = exit_data.get("status")
    status = exit_status if isinstance(exit_status, str) and exit_status else "exited"
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

def tail(path: Path, lines: int = 8, max_bytes: int = 65536) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        try:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
        except OSError:
            f.seek(0)
        data = f.read().decode("utf-8", errors="replace").splitlines()
    return data[-lines:]

def rel(path: Path) -> str:
    try:
        return str(path.relative_to(remote_dir))
    except ValueError:
        return str(path)

def mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()

def predates_job(path: Path) -> bool:
    return job_created_ts is not None and path.stat().st_mtime < job_created_ts

def count_map_text(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ",".join(f"{key}={value}" for key, value in sorted(counter.items()))

def job_command_text() -> str:
    command = metadata.get("original_command") or metadata.get("command") or []
    if isinstance(command, list):
        return " ".join(str(item) for item in command)
    return str(command)

def job_runs_phase4() -> bool:
    return bool(re.search(r"\bworldsim\.main\s+phase\s+4(?:\s|$)", job_command_text()))

def phase4_run_roots() -> list[Path]:
    roots: list[Path] = []
    configured_results_path = False
    state_dir = metadata.get("state_dir")
    if isinstance(state_dir, str) and state_dir.strip():
        root = Path(state_dir)
        if not root.is_absolute():
            root = remote_dir / root
        roots.append(root)
        configured_results_path = True
    for item in metadata.get("expected_outputs") or []:
        if isinstance(item, str) and item.endswith("phase_4/results.json"):
            path = Path(item)
            path = path if path.is_absolute() else remote_dir / path
            roots.append(path.parent.parent)
            configured_results_path = True
    if not configured_results_path and job_runs_phase4():
        roots.append(remote_dir / "logs")
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped

def phase4_results_candidates() -> list[Path]:
    candidates: list[Path] = []
    for root in phase4_run_roots():
        candidates.append(root / "phase_4" / "results.json")
    return candidates

def phase4_progress_candidates() -> list[Path]:
    return [root / "phase_4" / "progress.json" for root in phase4_run_roots()]

def load_task_lookup_for_results(results_path: Path) -> dict[str, dict[str, object]]:
    task_path = results_path.parent.parent / "phase_2" / "adversarial_tasks.json"
    if not task_path.exists():
        return {}
    try:
        data = json.loads(task_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("tasks"), list):
        items = data["tasks"]
    elif isinstance(data, dict):
        items = [data]
    else:
        return {}
    lookup: dict[str, dict[str, object]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        task_id = item.get("id")
        if isinstance(task_id, str) and task_id:
            lookup[task_id] = item
    return lookup

def summarize_phase4_results(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": path, "error": f"unparseable: {exc}"}
    if not isinstance(data, list):
        return {"path": path, "error": "results.json is not a list"}
    task_lookup = load_task_lookup_for_results(path)
    def task_site(item: dict[str, object]) -> str:
        value = item.get("site")
        if isinstance(value, str) and value:
            return value
        task_id = item.get("task_id")
        task = task_lookup.get(task_id) if isinstance(task_id, str) else None
        if isinstance(task, dict):
            value = task.get("site")
            if isinstance(value, str) and value:
                return value
        return "unknown"

    final_counts = Counter(str(item.get("final_status", "missing")) for item in data if isinstance(item, dict))
    site_counts = Counter(task_site(item) for item in data if isinstance(item, dict))
    trace_dirs = [
        str(item.get("primary_inspection_trace") or item.get("trajectory_dir"))
        for item in data
        if isinstance(item, dict)
        and isinstance(item.get("primary_inspection_trace") or item.get("trajectory_dir"), str)
    ]
    trace_root = None
    if trace_dirs:
        try:
            trace_root = os.path.commonpath(trace_dirs)
        except ValueError:
            trace_root = None
    return {
        "path": path,
        "total": len(data),
        "final_counts": final_counts,
        "site_counts": site_counts,
        "trace_root": trace_root,
    }

def summarize_phase4_progress(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": path, "error": f"unparseable: {exc}"}
    if not isinstance(data, dict):
        return {"path": path, "error": "progress.json is not an object"}

    def int_value(key: str) -> int:
        value = data.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return 0

    total = int_value("total_tasks")
    age_seconds = max(0, int(time.time() - path.stat().st_mtime))
    return {
        "path": path,
        "status": str(data.get("status") or "unknown"),
        "stage": str(data.get("stage") or "unknown"),
        "updated_at": str(data.get("updated_at") or "unknown"),
        "total_tasks": total,
        "completed_initial_tasks": int_value("completed_initial_tasks"),
        "postprocessed_tasks": int_value("postprocessed_tasks"),
        "age_seconds": age_seconds,
    }

def recent_health_warnings(lines: list[str]) -> list[str]:
    patterns = [
        ("host_unreachable", re.compile(r"host_unreachable|connection timeout|connection refused|name or service not known|could not resolve", re.I)),
        ("storage_state_host_mismatch", re.compile(r"storage[-_ ]state.*host.*mismatch|host-bound storage", re.I)),
        ("validation_retry", re.compile(r"failed validation, retrying|repair prompt|self[-_ ]verification failed", re.I)),
        ("phase0_unreachable", re.compile(r"instance is confirmed unreachable|instance was unreachable|unreachable", re.I)),
        ("phase4_variant_quality", re.compile(r"Quality flags:\s*(?!none\b).*|generated_contract_qa_failed|terminal_failures=(?!none\b)", re.I)),
    ]
    warnings: list[str] = []
    seen: set[str] = set()
    for line in lines:
        for label, pattern in patterns:
            if label in seen:
                continue
            if pattern.search(line):
                warnings.append(f"{label}: {line[-220:]}")
                seen.add(label)
    return warnings

stdout = job_dir / "stdout.log"
stderr = job_dir / "stderr.log"
latest_log_mtime = 0.0
for path in (stdout, stderr):
    if path.exists():
        latest_log_mtime = max(latest_log_mtime, path.stat().st_mtime)
recent_stdout = tail(stdout, lines=80)
recent_stderr = tail(stderr, lines=80)

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

health_warnings = recent_health_warnings(recent_stdout + recent_stderr)
if health_warnings:
    print("health_warnings:")
    for warning in health_warnings:
        print(f"  {warning}")

expected = metadata.get("expected_outputs") or []
if expected:
    print("expected_outputs:")
    for rel_path in expected:
        path = Path(rel_path)
        path = path if path.is_absolute() else remote_dir / path
        if path.exists():
            freshness = "stale" if predates_job(path) else "present"
            print(f"  {freshness} {rel_path} size={path.stat().st_size} mtime={mtime_iso(path)}")
        else:
            print(f"  missing {rel_path}")

for candidate in phase4_progress_candidates():
    summary = summarize_phase4_progress(candidate)
    if summary is None:
        continue
    if summary.get("error"):
        print(f"phase4_progress: present {rel(candidate)} error={summary['error']}")
        break
    print(
        "phase4_progress: "
        f"present {rel(candidate)} "
        f"status={summary['status']} "
        f"stage={summary['stage']} "
        f"initial={summary['completed_initial_tasks']}/{summary['total_tasks']} "
        f"postprocessed={summary['postprocessed_tasks']}/{summary['total_tasks']} "
        f"age_seconds={summary['age_seconds']} "
        f"updated_at={summary['updated_at']}"
    )
    break

for candidate in phase4_results_candidates():
    if candidate.exists() and predates_job(candidate):
        print(f"phase4_results: stale {rel(candidate)} mtime={mtime_iso(candidate)}")
        continue
    summary = summarize_phase4_results(candidate)
    if summary is None:
        continue
    if summary.get("error"):
        print(f"phase4_results: present {rel(candidate)} error={summary['error']}")
        break
    print(
        "phase4_results: "
        f"present {rel(candidate)} "
        f"total={summary['total']} "
        f"final_status={count_map_text(summary['final_counts'])} "
        f"sites={count_map_text(summary['site_counts'])}"
    )
    if summary.get("trace_root"):
        print(f"phase4_trace_root: {summary['trace_root']}")
    manifest_path = candidate.parent.parent / "artifact_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        if isinstance(manifest, dict):
            artifacts = manifest.get("artifacts")
            artifact_count = len(artifacts) if isinstance(artifacts, list) else 0
            print(
                "artifact_manifest: "
                f"{rel(manifest_path)} "
                f"source={manifest.get('artifacts_source') or 'unknown'} "
                f"artifacts={artifact_count}"
            )
    print(
        "phase4_summary_command: "
        f"cd {remote_dir} && uv run python scripts/summarize_phase_4_results.py "
        f"{rel(candidate)} --inspect-limit 8"
    )
    break

print(f"stdout: {stdout}")
for line in recent_stdout[-8:]:
    print(f"  {line}")
print(f"stderr: {stderr}")
for line in recent_stderr[-8:]:
    print(f"  {line}")
PY
REMOTE
