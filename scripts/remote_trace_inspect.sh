#!/usr/bin/env bash
# Run the compact Phase 4 trace inspector on a remote WorldSim checkout.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/remote_jobs.sh
source "$REPO_ROOT/scripts/lib/remote_jobs.sh"

HOST_CONFIG=""
REMOTE_DIR=""
SSH_KEY_ARG=""
RUN=""
INSPECT_ARGS=()

usage() {
    cat <<'USAGE'
remote_trace_inspect.sh

Options:
  --host-config <path>      benchmark host YAML (required)
  --remote-dir <path>       remote checkout dir (default: <compose_dir_remote>/browser-sim)
  --ssh-key <path>          SSH private key (default: $SSH_KEY or ~/.ssh/webarena-key.pem)
  --run <path>              remote state dir, phase_4 dir, or results.json path (required)
  -h, --help                show this help

Everything after -- is forwarded to scripts/inspect_phase4_traces.py after the
run path, for example:

  scripts/remote_trace_inspect.sh --host-config configs/benchmark_hosts/r5.yaml \
    --remote-dir /home/ubuntu/browser-sim \
    --run logs/my_run -- summary --action create_issue_note
USAGE
}

while (($#)); do
    case "$1" in
        --host-config) HOST_CONFIG="$2"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --ssh-key) SSH_KEY_ARG="$2"; shift 2 ;;
        --run) RUN="$2"; shift 2 ;;
        --) shift; INSPECT_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) rj_die "unknown arg: $1" ;;
    esac
done

[[ -n "$HOST_CONFIG" ]] || { usage >&2; rj_die "--host-config required"; }
[[ -n "$RUN" ]] || { usage >&2; rj_die "--run required"; }
if [[ ${#INSPECT_ARGS[@]} -eq 0 ]]; then
    INSPECT_ARGS=("summary")
fi

rj_prepare_connection "$HOST_CONFIG" "$SSH_KEY_ARG"
REMOTE_DIR="${REMOTE_DIR:-$(rj_default_remote_dir)}"
ARGS_B64="$(rj_json_b64 "$RUN" "${INSPECT_ARGS[@]}")"

rj_ssh_bash "$REMOTE_DIR" "$ARGS_B64" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
args_b64="$2"
cd "$remote_dir"
python3 - "$args_b64" <<'PY'
import base64
import json
import shlex
import subprocess
import sys

args = json.loads(base64.b64decode(sys.argv[1]).decode())
cmd = ["uv", "run", "python", "scripts/inspect_phase4_traces.py", *args]
print("+ " + " ".join(shlex.quote(item) for item in cmd), file=sys.stderr)
raise SystemExit(subprocess.call(cmd))
PY
REMOTE
