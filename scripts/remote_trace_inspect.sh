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
JOB_ID=""
LATEST=0
NAME=""
JSON=0
INSPECT_ARGS=()

usage() {
    cat <<'USAGE'
remote_trace_inspect.sh

Options:
  --host-config <path>      benchmark host YAML (required)
  --remote-dir <path>       remote checkout dir (default: <compose_dir_remote>/browser-sim)
  --ssh-key <path>          SSH private key (default: $SSH_KEY or ~/.ssh/webarena-key.pem)
  --run <path>              remote state dir, phase_4 dir, or results.json path
  --job-id <id>             derive --run from registered remote job metadata
  --latest                  derive --run from latest registered remote job
  --name <name>             derive --run from latest job with this human name
  --json                    print wrapper metadata as JSON before inspector output
  -h, --help                show this help

Everything after -- is forwarded to `worldsim trace` after the run path, for example:

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
        --job-id) JOB_ID="$2"; shift 2 ;;
        --latest) LATEST=1; shift ;;
        --name) NAME="$2"; shift 2 ;;
        --json) JSON=1; shift ;;
        --) shift; INSPECT_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) rj_die "unknown arg: $1" ;;
    esac
done

[[ -n "$HOST_CONFIG" ]] || { usage >&2; rj_die "--host-config required"; }
if [[ -z "$RUN" ]]; then
    rj_validate_job_selector "$JOB_ID" "$LATEST" "$NAME"
fi
if [[ ${#INSPECT_ARGS[@]} -eq 0 ]]; then
    INSPECT_ARGS=("summary")
fi

rj_prepare_connection "$HOST_CONFIG" "$SSH_KEY_ARG"
REMOTE_DIR="${REMOTE_DIR:-$(rj_default_remote_dir)}"
if [[ -z "$RUN" ]]; then
    JOB_ID="$(rj_resolve_job_id "$REMOTE_DIR" "$JOB_ID" "$LATEST" "$NAME")"
fi
ARGS_B64="$(rj_json_b64 "$RUN" "$JOB_ID" "$JSON" "${INSPECT_ARGS[@]}")"

rj_ssh_bash "$REMOTE_DIR" "$ARGS_B64" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
args_b64="$2"
cd "$remote_dir"
python3 - "$args_b64" "$remote_dir" <<'PY'
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys

args = json.loads(base64.b64decode(sys.argv[1]).decode())
remote_dir = sys.argv[2]
run, job_id, json_flag, *inspect_args = args
if not run:
    job_dir = Path(remote_dir) / "logs" / "remote_jobs" / job_id
    metadata = json.loads((job_dir / "metadata.json").read_text(encoding="utf-8"))
    state_dir = metadata.get("state_dir")
    if isinstance(state_dir, str) and state_dir:
        run = state_dir
    else:
        expected = metadata.get("expected_outputs") or []
        run = next(
            (
                str(Path(item).parent.parent)
                for item in expected
                if isinstance(item, str) and item.endswith("phase_4/results.json")
            ),
            "",
        )
if not run:
    raise SystemExit("could not derive run path; pass --run explicitly")
run_path = Path(run)
if not run_path.is_absolute():
    run_path = Path(remote_dir) / run_path
if not run_path.exists():
    raise SystemExit(f"remote run path not found: {run_path}")
cmd = ["uv", "run", "python", "-m", "worldsim.main", "trace", *inspect_args[:1], str(run_path), *inspect_args[1:]]
metadata = {
    "remote_dir": remote_dir,
    "run": str(run_path),
    "job_id": job_id or None,
    "command": cmd,
}
if json_flag == "1":
    print(json.dumps({"remote_trace_inspect": metadata}, sort_keys=True), file=sys.stderr)
print("+ " + " ".join(shlex.quote(item) for item in cmd), file=sys.stderr)
raise SystemExit(subprocess.call(cmd))
PY
REMOTE
