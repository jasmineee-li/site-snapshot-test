#!/usr/bin/env bash
# Tail stdout/stderr for a registered remote WorldSim job.

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
LINES=80
FOLLOW=0
STREAM="stdout"
WORKER_ID=""
TASK_ID=""

usage() {
    cat <<'USAGE'
remote_job_tail.sh

Options:
  --host-config <path>      benchmark host YAML (required)
  --remote-dir <path>       remote checkout dir (default: <compose_dir_remote>/browser-sim)
  --ssh-key <path>          SSH private key (default: $SSH_KEY or ~/.ssh/webarena-key.pem)
  --job-id <id>             job id to tail
  --latest                  tail latest job
  --name <name>             tail latest job with this human name
  --lines <n>               lines to show (default: 80)
  --stderr                  tail stderr instead of stdout
  --both                    show stdout then stderr (no --follow)
  --worker-id <n>           tail process-pool worker_NNN stdout/stderr
  --task-id <id>            tail process-pool worker for task id, active or completed
  --follow                  follow the selected stream
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
        --lines) LINES="$2"; shift 2 ;;
        --stderr) STREAM="stderr"; shift ;;
        --both) STREAM="both"; shift ;;
        --worker-id) WORKER_ID="$2"; shift 2 ;;
        --task-id) TASK_ID="$2"; shift 2 ;;
        --follow) FOLLOW=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) rj_die "unknown arg: $1" ;;
    esac
done

[[ -n "$HOST_CONFIG" ]] || { usage >&2; rj_die "--host-config required"; }
[[ "$LINES" =~ ^[1-9][0-9]*$ ]] || rj_die "--lines must be a positive integer"
if [[ "$STREAM" == "both" && "$FOLLOW" -eq 1 ]]; then
    rj_die "--both cannot be combined with --follow"
fi

rj_validate_job_selector "$JOB_ID" "$LATEST" "$NAME"
rj_prepare_connection "$HOST_CONFIG" "$SSH_KEY_ARG"
REMOTE_DIR="${REMOTE_DIR:-$(rj_default_remote_dir)}"
JOB_ID="$(rj_resolve_job_id "$REMOTE_DIR" "$JOB_ID" "$LATEST" "$NAME")"

rj_ssh_bash "$REMOTE_DIR" "$JOB_ID" "$LINES" "$FOLLOW" "$STREAM" "$WORKER_ID" "$TASK_ID" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
job_id="$2"
lines="$3"
follow="$4"
stream="$5"
worker_id="$6"
task_id="$7"
job_dir="$remote_dir/logs/remote_jobs/$job_id"
metadata="$job_dir/metadata.json"
state_dir=""
if [[ -f "$metadata" ]]; then
    state_dir="$(python3 - "$metadata" "$remote_dir" <<'PY'
import json, sys
from pathlib import Path
metadata = json.loads(Path(sys.argv[1]).read_text())
remote_dir = Path(sys.argv[2])
state = metadata.get("state_dir")
if isinstance(state, str) and state:
    path = Path(state)
    print(path if path.is_absolute() else remote_dir / path)
PY
)"
fi
task_lookup_json=""
if [[ -n "$task_id" ]]; then
    [[ -n "$state_dir" ]] || { echo "job has no state_dir metadata; cannot resolve task logs" >&2; exit 2; }
    task_lookup_json="$(python3 - "$state_dir" "$task_id" <<'PY'
import json, sys
from pathlib import Path
state_dir = Path(sys.argv[1])
task_id = sys.argv[2]


def load(path: Path):
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def worker_payload(worker, source):
    return {
        "source": source,
        "worker_id": worker.get("worker_id"),
        "stdout": worker.get("stdout"),
        "stderr": worker.get("stderr"),
    }


progress = load(state_dir / "phase_4" / "progress.json")
for worker in progress.get("process_pool_active_workers") or []:
    if isinstance(worker, dict) and str(worker.get("task_id")) == task_id:
        print(json.dumps(worker_payload(worker, "progress")))
        raise SystemExit(0)

matches = []
summary = load(state_dir / "phase_4" / "process_pool_summary.json")
for worker in summary.get("outcomes") or []:
    if isinstance(worker, dict) and str(worker.get("task_id")) == task_id:
        matches.append(worker_payload(worker, "process_pool_summary"))
if len(matches) == 1:
    print(json.dumps(matches[0]))
    raise SystemExit(0)
if len(matches) > 1:
    print(json.dumps({"error": f"ambiguous task id in process_pool_summary: {task_id}"}))
    raise SystemExit(0)

partial = load(state_dir / "phase_4" / "partial_manifest.json")
for worker in partial.get("workers") or []:
    if isinstance(worker, dict) and str(worker.get("task_id")) == task_id:
        matches.append(worker_payload(worker, "partial_manifest"))
if len(matches) == 1:
    print(json.dumps(matches[0]))
    raise SystemExit(0)
if len(matches) > 1:
    print(json.dumps({"error": f"ambiguous task id in partial_manifest: {task_id}"}))
    raise SystemExit(0)

available = []
for source, rows in (
    ("progress", progress.get("process_pool_active_workers") or []),
    ("process_pool_summary", summary.get("outcomes") or []),
    ("partial_manifest", partial.get("workers") or []),
):
    for worker in rows:
        if isinstance(worker, dict) and worker.get("task_id") is not None:
            available.append(f"{worker.get('task_id')} ({source})")
print(json.dumps({
    "error": f"task id not found in process-pool worker logs: {task_id}",
    "available": available[:12],
}))
PY
)"
    task_error="$(python3 - "$task_lookup_json" <<'PY'
import json, sys
try:
    payload = json.loads(sys.argv[1] or "{}")
except Exception:
    payload = {"error": "could not parse task lookup result"}
print(payload.get("error") or "")
PY
)"
    if [[ -n "$task_error" ]]; then
        echo "$task_error" >&2
        python3 - "$task_lookup_json" <<'PY' >&2
import json, sys
try:
    payload = json.loads(sys.argv[1] or "{}")
except Exception:
    raise SystemExit(0)
available = payload.get("available")
if isinstance(available, list) and available:
    print("available task ids: " + ", ".join(str(item) for item in available))
PY
        exit 2
    fi
    worker_id="$(python3 - "$task_lookup_json" <<'PY'
import json, sys
payload = json.loads(sys.argv[1] or "{}")
value = payload.get("worker_id")
print("" if value is None else value)
PY
)"
    [[ -n "$worker_id" ]] || { echo "task lookup did not include worker_id for task: $task_id" >&2; exit 2; }
fi
if [[ -n "$worker_id" ]]; then
    [[ -n "$state_dir" ]] || { echo "job has no state_dir metadata; cannot resolve worker logs" >&2; exit 2; }
    if [[ "$worker_id" =~ ^[0-9]+$ ]]; then
        printf -v padded_worker "%03d" "$worker_id"
    else
        padded_worker="$worker_id"
    fi
    worker_dir="$state_dir/phase_4/process_pool_workers/worker_$padded_worker"
    stdout_path="$worker_dir/stdout.log"
    stderr_path="$worker_dir/stderr.log"
    if [[ -n "$task_lookup_json" ]]; then
        lookup_stdout="$(python3 - "$task_lookup_json" <<'PY'
import json, sys
payload = json.loads(sys.argv[1] or "{}")
print(payload.get("stdout") or "")
PY
)"
        lookup_stderr="$(python3 - "$task_lookup_json" <<'PY'
import json, sys
payload = json.loads(sys.argv[1] or "{}")
print(payload.get("stderr") or "")
PY
)"
        [[ -n "$lookup_stdout" ]] && stdout_path="$lookup_stdout"
        [[ -n "$lookup_stderr" ]] && stderr_path="$lookup_stderr"
    fi
    case "$stream" in
        stdout) target="$stdout_path" ;;
        stderr) target="$stderr_path" ;;
        both)
            echo "==> worker_$padded_worker stdout"
            tail -n "$lines" "$stdout_path" || true
            echo "==> worker_$padded_worker stderr"
            tail -n "$lines" "$stderr_path" || true
            exit 0
            ;;
        *) echo "unknown stream: $stream" >&2; exit 2 ;;
    esac
    if [[ "$follow" -eq 1 ]]; then
        tail -n "$lines" -f "$target"
    else
        tail -n "$lines" "$target"
    fi
    exit 0
fi
case "$stream" in
    stdout) target="$job_dir/stdout.log" ;;
    stderr) target="$job_dir/stderr.log" ;;
    both)
        echo "==> stdout"
        tail -n "$lines" "$job_dir/stdout.log" || true
        echo "==> stderr"
        tail -n "$lines" "$job_dir/stderr.log" || true
        exit 0
        ;;
    *) echo "unknown stream: $stream" >&2; exit 2 ;;
esac
if [[ "$follow" -eq 1 ]]; then
    tail -n "$lines" -f "$target"
else
    tail -n "$lines" "$target"
fi
REMOTE
