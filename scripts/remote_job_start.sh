#!/usr/bin/env bash
# Start a detached WorldSim command on a benchmark host with file-backed logs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/remote_jobs.sh
source "$REPO_ROOT/scripts/lib/remote_jobs.sh"

HOST_CONFIG=""
NAME=""
REMOTE_DIR=""
SSH_KEY_ARG=""
STATE_DIR_MODE="none"
STATE_DIR_VALUE=""
EXPECTED_OUTPUTS=()

usage() {
    cat <<'USAGE'
remote_job_start.sh

Options:
  --host-config <path>        benchmark host YAML (required)
  --name <name>               human-readable job name (required)
  --remote-dir <path>         remote checkout dir (default: <compose_dir_remote>/browser-sim)
  --ssh-key <path>            SSH private key (default: $SSH_KEY or ~/.ssh/webarena-key.pem)
  --expected-output <path>    expected output path relative to remote dir (repeatable)
  --state-dir auto|<path>     set WORLDSIM_STATE_DIR; auto uses logs/remote_jobs/<job_id>/state
  --no-state-dir              do not set WORLDSIM_STATE_DIR (default)
  -h, --help                  show this help

Command follows after --, for example:
  scripts/remote_job_start.sh --host-config configs/benchmark_hosts/r5.yaml \
    --name phase1-route-diversity --remote-dir /home/ubuntu/browser-sim \
    --expected-output logs/phase_1/benign_tasks.json -- \
    uv run python -m worldsim.main phase 1 --generate-novel
USAGE
}

while (($#)); do
    case "$1" in
        --host-config) HOST_CONFIG="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --ssh-key) SSH_KEY_ARG="$2"; shift 2 ;;
        --expected-output) EXPECTED_OUTPUTS+=("$2"); shift 2 ;;
        --state-dir)
            STATE_DIR_MODE="set"
            STATE_DIR_VALUE="$2"
            shift 2
            ;;
        --no-state-dir)
            STATE_DIR_MODE="none"
            STATE_DIR_VALUE=""
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        *) rj_die "unknown arg before --: $1" ;;
    esac
done

[[ -n "$HOST_CONFIG" ]] || { usage >&2; rj_die "--host-config required"; }
[[ -n "$NAME" ]] || { usage >&2; rj_die "--name required"; }
(($# > 0)) || { usage >&2; rj_die "missing command after --"; }

ORIGINAL_COMMAND_B64="$(rj_json_b64 "$@")"
rj_guard_runtime_instance_topology "$HOST_CONFIG" "$@"
rj_prepare_connection "$HOST_CONFIG" "$SSH_KEY_ARG"
REMOTE_DIR="${REMOTE_DIR:-$(rj_default_remote_dir)}"

JOB_ID="$(python3 - "$NAME" <<'PY'
import re
import secrets
import sys
from datetime import datetime, timezone

name = re.sub(r"[^A-Za-z0-9_.-]+", "-", sys.argv[1].strip()).strip("-").lower()
name = name[:48] or "job"
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
print(f"{stamp}-{name}-{secrets.token_hex(3)}")
PY
)"

COMMAND_ENVELOPE_B64="$(python3 - "$@" <<'PY'
import base64
import json
import os
import shlex
import sys
from pathlib import Path

original = list(sys.argv[1:])
mode = os.environ.get("WORLDSIM_REMOTE_JOB_EXEC_MODE", "auto").strip().lower() or "auto"
if mode not in {"auto", "direct", "login-shell"}:
    raise SystemExit(
        "WORLDSIM_REMOTE_JOB_EXEC_MODE must be one of auto, direct, or login-shell"
    )

SHELLS = {"bash", "sh", "zsh"}
ENV_MANAGED_COMMANDS = {
    "bun",
    "claude",
    "modal",
    "node",
    "npm",
    "npx",
    "pnpm",
    "poetry",
    "uv",
    "uvx",
}


def basename(value: str) -> str:
    return Path(value).name


def already_shell(argv: list[str]) -> bool:
    return len(argv) >= 3 and basename(argv[0]) in SHELLS and argv[1] in {"-c", "-lc"}


def managed_command_name(argv: list[str]) -> str:
    if not argv:
        return ""
    if basename(argv[0]) == "env":
        for item in argv[1:]:
            if "=" not in item:
                return basename(item)
        return "env"
    return basename(argv[0])


def should_login_shell_wrap(argv: list[str]) -> bool:
    if not argv or already_shell(argv):
        return False
    if os.path.isabs(argv[0]) or "/" in argv[0]:
        return False
    return managed_command_name(argv) in ENV_MANAGED_COMMANDS


normalized = original
reason = "direct"
if mode == "login-shell" and not already_shell(original):
    normalized = ["bash", "-lc", shlex.join(original)]
    reason = "forced_login_shell"
elif mode == "auto" and should_login_shell_wrap(original):
    normalized = ["bash", "-lc", shlex.join(original)]
    reason = f"auto_login_shell_for_{managed_command_name(original)}"
elif already_shell(original):
    reason = "already_shell"

payload = {
    "command": normalized,
    "execution": {
        "mode": mode,
        "normalized": normalized != original,
        "reason": reason,
        "original_command": original,
    },
}
print(base64.b64encode(json.dumps(payload).encode()).decode())
PY
)"
if ((${#EXPECTED_OUTPUTS[@]})); then
    EXPECTED_B64="$(rj_json_b64 "${EXPECTED_OUTPUTS[@]}")"
else
    EXPECTED_B64="$(rj_json_b64)"
fi
LOCAL_META_B64="$(python3 - "$HOST_CONFIG" "$RJ_REPO_ROOT" <<'PY'
import base64
import json
import subprocess
import sys
from pathlib import Path

host_config = sys.argv[1]
repo = Path(sys.argv[2])

def git_value(args):
    try:
        return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

payload = {
    "host_config": host_config,
    "local_git": {
        "sha": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(git_value(["status", "--porcelain"])),
    },
}
print(base64.b64encode(json.dumps(payload).encode()).decode())
PY
)"

rj_ssh_bash "$REMOTE_DIR" "$JOB_ID" "$NAME" "$HOST_CONFIG" "$STATE_DIR_MODE" "$STATE_DIR_VALUE" "$EXPECTED_B64" "$LOCAL_META_B64" "$COMMAND_ENVELOPE_B64" "$ORIGINAL_COMMAND_B64" <<'REMOTE'
set -euo pipefail

remote_dir="$1"
job_id="$2"
name="$3"
host_config="$4"
state_dir_mode="$5"
state_dir_value="$6"
expected_b64="$7"
local_meta_b64="$8"
command_envelope_b64="$9"
original_command_b64="${10}"

job_dir="$remote_dir/logs/remote_jobs/$job_id"
mkdir -p "$job_dir"

python3 - "$job_dir" "$remote_dir" "$job_id" "$name" "$host_config" "$state_dir_mode" "$state_dir_value" "$expected_b64" "$local_meta_b64" "$command_envelope_b64" "$original_command_b64" <<'PY'
import base64
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

job_dir = Path(sys.argv[1])
remote_dir = Path(sys.argv[2])
job_id = sys.argv[3]
name = sys.argv[4]
host_config = sys.argv[5]
state_dir_mode = sys.argv[6]
state_dir_value = sys.argv[7]
expected_outputs = json.loads(base64.b64decode(sys.argv[8]).decode())
local_meta = json.loads(base64.b64decode(sys.argv[9]).decode())
command_envelope = json.loads(base64.b64decode(sys.argv[10]).decode())
original_argv = json.loads(base64.b64decode(sys.argv[11]).decode())
if isinstance(command_envelope, list):
    argv = command_envelope
    command_execution = {
        "mode": "legacy",
        "normalized": False,
        "reason": "legacy_command_envelope",
        "original_command": original_argv,
    }
else:
    argv = command_envelope.get("command")
    command_execution = command_envelope.get("execution")
if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
    raise SystemExit("remote job command envelope must contain a string argv list")
if not isinstance(command_execution, dict):
    command_execution = {
        "mode": "unknown",
        "normalized": argv != original_argv,
        "reason": "missing_execution_metadata",
        "original_command": original_argv,
    }

argv_json = json.dumps(argv, indent=2)
fingerprint = hashlib.sha256(argv_json.encode()).hexdigest()[:16]
(job_dir / "command.argv.json").write_text(argv_json + "\n", encoding="utf-8")

def git_value(args):
    try:
        return subprocess.check_output(["git", *args], cwd=remote_dir, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

state_dir = None
if state_dir_mode == "set":
    if state_dir_value == "auto":
        state_dir = f"logs/remote_jobs/{job_id}/state"
    else:
        state_dir = state_dir_value

metadata = {
    "job_id": job_id,
    "name": name,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "host_config": host_config,
    "remote_dir": str(remote_dir),
    "command_fingerprint": fingerprint,
    "command": argv,
    "original_command": original_argv,
    "command_execution": command_execution,
    "log_paths": {
        "stdout": str(job_dir / "stdout.log"),
        "stderr": str(job_dir / "stderr.log"),
    },
    "expected_outputs": expected_outputs,
    "state_dir": state_dir,
    "local_git": local_meta.get("local_git"),
    "remote_git": {
        "sha": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(git_value(["status", "--porcelain"])),
    },
    "remote_sync_stamp": read_json(remote_dir / ".worldsim_sync_stamp.json"),
}
(job_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(job_dir / "stdout.log").touch()
(job_dir / "stderr.log").touch()
(job_dir / "heartbeat.json").write_text(json.dumps({"status": "starting", "updated_at": metadata["created_at"]}) + "\n", encoding="utf-8")

runner = f'''#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

job_dir = Path({str(job_dir)!r})
remote_dir = Path({str(remote_dir)!r})
fingerprint = {fingerprint!r}
state_dir = {state_dir!r}
argv = json.loads((job_dir / "command.argv.json").read_text(encoding="utf-8"))

try:
    os.setsid()
except OSError:
    pass

def now():
    return datetime.now(timezone.utc).isoformat()

def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")

env = os.environ.copy()
env["WORLDSIM_REMOTE_JOB_ID"] = {job_id!r}
env["WORLDSIM_REMOTE_JOB_DIR"] = str(job_dir)
if state_dir:
    env["WORLDSIM_STATE_DIR"] = state_dir

stdout = open(job_dir / "stdout.log", "ab", buffering=0)
stderr = open(job_dir / "stderr.log", "ab", buffering=0)
try:
    proc = subprocess.Popen(
        argv,
        cwd=remote_dir,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        env=env,
    )
except Exception as exc:
    message = f"remote job child launch failed: {{type(exc).__name__}}: {{exc}}\\n"
    stderr.write(message.encode("utf-8", errors="replace"))
    write_json(job_dir / "exit.json", {{"status": "launch_failed", "updated_at": now(), "pid": os.getpid(), "returncode": 127, "error": message.strip(), "fingerprint": fingerprint}})
    write_json(job_dir / "heartbeat.json", {{"status": "launch_failed", "updated_at": now(), "pid": os.getpid(), "returncode": 127, "error": message.strip(), "fingerprint": fingerprint}})
    sys.exit(127)
write_json(job_dir / "heartbeat.json", {{"status": "running", "updated_at": now(), "pid": os.getpid(), "child_pid": proc.pid, "fingerprint": fingerprint}})

stop = False
def heartbeat():
    while not stop:
        write_json(job_dir / "heartbeat.json", {{"status": "running", "updated_at": now(), "pid": os.getpid(), "child_pid": proc.pid, "fingerprint": fingerprint}})
        time.sleep(30)

thread = threading.Thread(target=heartbeat, daemon=True)
thread.start()
returncode = proc.wait()
stop = True
write_json(job_dir / "exit.json", {{"status": "exited", "updated_at": now(), "pid": os.getpid(), "child_pid": proc.pid, "returncode": returncode, "fingerprint": fingerprint}})
write_json(job_dir / "heartbeat.json", {{"status": "exited", "updated_at": now(), "pid": os.getpid(), "child_pid": proc.pid, "returncode": returncode, "fingerprint": fingerprint}})
sys.exit(returncode)
'''
(job_dir / "run_job.py").write_text(runner, encoding="utf-8")
os.chmod(job_dir / "run_job.py", 0o755)
PY

fingerprint="$(python3 - "$job_dir/metadata.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["command_fingerprint"])
PY
)"

cd "$remote_dir"
python3 "$job_dir/run_job.py" "$fingerprint" </dev/null >/dev/null 2>/dev/null &
pid="$!"
pgid=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    pgid="$(python3 -c 'import os, sys; print(os.getpgid(int(sys.argv[1])))' "$pid" 2>/dev/null || true)"
    [[ "$pgid" == "$pid" ]] && break
    sleep 0.1
done
start_ticks=""
if [[ -r "/proc/$pid/stat" ]]; then
    start_ticks="$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)"
fi

printf '%s\n' "$pid" > "$job_dir/pid"
printf '%s\n' "${pgid:-$pid}" > "$job_dir/pgid"
printf '%s\n' "$start_ticks" > "$job_dir/start_ticks"

python3 - "$job_dir/metadata.json" "$pid" "${pgid:-$pid}" "$start_ticks" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metadata = json.loads(path.read_text(encoding="utf-8"))
metadata["pid"] = int(sys.argv[2])
metadata["pgid"] = int(sys.argv[3])
metadata["process_start_ticks"] = sys.argv[4] or None
path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

cat <<OUT
job_id=$job_id
pid=$pid
pgid=${pgid:-$pid}
metadata=$job_dir/metadata.json
stdout=$job_dir/stdout.log
stderr=$job_dir/stderr.log
OUT
REMOTE

cat <<OUT

Remote job commands:
  scripts/remote_job_status.sh --host-config $HOST_CONFIG --remote-dir $REMOTE_DIR --job-id $JOB_ID
  scripts/remote_job_tail.sh --host-config $HOST_CONFIG --remote-dir $REMOTE_DIR --job-id $JOB_ID --lines 80
  scripts/remote_job_stop.sh --host-config $HOST_CONFIG --remote-dir $REMOTE_DIR --job-id $JOB_ID
OUT
