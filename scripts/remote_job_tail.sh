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
  --task-id <id>            tail active process-pool worker for task id
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
if [[ -n "$task_id" && -n "$state_dir" ]]; then
    worker_id="$(python3 - "$state_dir" "$task_id" <<'PY'
import json, sys
from pathlib import Path
progress = Path(sys.argv[1]) / "phase_4" / "progress.json"
task_id = sys.argv[2]
try:
    data = json.loads(progress.read_text())
except Exception:
    data = {}
for worker in data.get("process_pool_active_workers") or []:
    if isinstance(worker, dict) and str(worker.get("task_id")) == task_id:
        print(worker.get("worker_id"))
        break
PY
)"
fi
if [[ -n "$worker_id" ]]; then
    [[ -n "$state_dir" ]] || { echo "job has no state_dir metadata; cannot resolve worker logs" >&2; exit 2; }
    if [[ "$worker_id" =~ ^[0-9]+$ ]]; then
        printf -v padded_worker "%03d" "$worker_id"
    else
        padded_worker="$worker_id"
    fi
    worker_dir="$state_dir/phase_4/process_pool_workers/worker_$padded_worker"
    case "$stream" in
        stdout) target="$worker_dir/stdout.log" ;;
        stderr) target="$worker_dir/stderr.log" ;;
        both)
            echo "==> worker_$padded_worker stdout"
            tail -n "$lines" "$worker_dir/stdout.log" || true
            echo "==> worker_$padded_worker stderr"
            tail -n "$lines" "$worker_dir/stderr.log" || true
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
