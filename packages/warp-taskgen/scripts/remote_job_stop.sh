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
GRACEFUL=0
PAUSE_TIMEOUT=300
PAUSE_POLL_INTERVAL=0.25

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
  --graceful                request and read back Run pause before TERM
  --pause-timeout <seconds> bounded remote pause wait (default: 300)
  --pause-poll-interval <seconds>
                            remote pause readback interval (default: 0.25)
  -h, --help                show this help

Stop is intentionally job-id only. It never accepts command patterns.
Without --graceful this preserves the existing abrupt TERM/KILL behavior.
With --graceful, TERM is sent only after the explicit Run root reports paused.
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
        --graceful) GRACEFUL=1; shift ;;
        --pause-timeout) PAUSE_TIMEOUT="$2"; shift 2 ;;
        --pause-poll-interval) PAUSE_POLL_INTERVAL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) rj_die "unknown arg: $1" ;;
    esac
done

[[ -n "$HOST_CONFIG" ]] || { usage >&2; rj_die "--host-config required"; }
[[ -n "$JOB_ID" ]] || { usage >&2; rj_die "--job-id required; command patterns are not accepted"; }
[[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]] || rj_die "--timeout must be a positive integer"
if [[ "$GRACEFUL" -eq 1 ]]; then
    if ! python3 - "$PAUSE_TIMEOUT" "$PAUSE_POLL_INTERVAL" <<'PY'
import math
import sys

try:
    timeout = float(sys.argv[1])
    poll = float(sys.argv[2])
except ValueError:
    raise SystemExit("--pause-timeout and --pause-poll-interval must be numbers")
if not math.isfinite(timeout) or timeout < 0:
    raise SystemExit("--pause-timeout must be finite and non-negative")
if not math.isfinite(poll) or poll < 0:
    raise SystemExit("--pause-poll-interval must be finite and non-negative")
if timeout > 0 and poll <= 0:
    raise SystemExit("--pause-poll-interval must be positive when --pause-timeout is non-zero")
PY
    then
        exit 2
    fi
fi

rj_prepare_connection "$HOST_CONFIG" "$SSH_KEY_ARG"
REMOTE_DIR="${REMOTE_DIR:-$(rj_default_remote_dir)}"

rj_ssh_bash "$REMOTE_DIR" "$JOB_ID" "$FORCE" "$TIMEOUT" "$GRACEFUL" "$PAUSE_TIMEOUT" "$PAUSE_POLL_INTERVAL" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
job_id="$2"
force="$3"
timeout="$4"
graceful="$5"
pause_timeout="$6"
pause_poll_interval="$7"
python3 - "$remote_dir" "$job_id" "$force" "$timeout" "$graceful" "$pause_timeout" "$pause_poll_interval" <<'PY'
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

remote_dir = Path(sys.argv[1]).resolve()
job_id = sys.argv[2]
force = sys.argv[3] == "1"
timeout = int(sys.argv[4])
graceful = sys.argv[5] == "1"
pause_timeout = sys.argv[6]
pause_poll_interval = sys.argv[7]
job_dir = remote_dir / "logs" / "remote_jobs" / job_id
metadata_path = job_dir / "metadata.json"
if not metadata_path.exists():
    raise SystemExit(f"metadata not found: {metadata_path}")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
pid = int(metadata.get("pid") or (job_dir / "pid").read_text().strip())
pgid = int(metadata.get("pgid") or (job_dir / "pgid").read_text().strip())
fingerprint = str(metadata.get("command_fingerprint") or "")
start_ticks = str(metadata.get("process_start_ticks") or "").strip()
# Normal remote control always reads Linux procfs.  The narrow override is
# only useful to deterministic fake-SSH tests on hosts without procfs; when
# procfs exists, the production default and its PID-reuse checks cannot be
# replaced by an environment variable.
if Path("/proc").is_dir():
    proc_root = Path("/proc")
else:
    proc_root = Path(os.environ.get("WARP_REMOTE_JOB_PROC_ROOT", "/proc"))
proc = proc_root / str(pid)
synthetic_proc = proc_root != Path("/proc")

if graceful and (
    not metadata.get("pid")
    or not metadata.get("pgid")
    or not start_ticks
    or not fingerprint
):
    write_payload = {
        "status": "pause_rejected",
        "control": "graceful",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "reason": "complete process identity metadata is required for graceful stop",
        "pid": pid,
        "pgid": pgid,
    }
    (job_dir / "stop.json").write_text(json.dumps(write_payload, indent=2) + "\n", encoding="utf-8")
    raise SystemExit("graceful stop rejected: complete process identity metadata is required")

def now():
    return datetime.now(timezone.utc).isoformat()

def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def alive() -> bool:
    if synthetic_proc:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
    if not proc.exists():
        return False
    try:
        return proc.joinpath("stat").read_text().split()[2] != "Z"
    except Exception:
        return True

def run_root() -> Path:
    raw = metadata.get("state_dir")
    if not isinstance(raw, str) or not raw.strip():
        raise SystemExit(
            "graceful stop rejected: remote job metadata has no explicit state_dir"
        )
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = remote_dir / root
    return root.resolve(strict=False)

def write_graceful_result(status: str, **extra: object) -> None:
    write_json(
        job_dir / "stop.json",
        {"status": status, "updated_at": now(), "pid": pid, "pgid": pgid, **extra},
    )

graceful_proof: dict[str, object] = {}

def run_identity(state: object) -> tuple[str | None, str | None]:
    if not isinstance(state, dict):
        return None, None
    definition = state.get("run_definition")
    if not isinstance(definition, dict):
        definition = {}
    run_id = state.get("run_id") or definition.get("run_id")
    digest = state.get("definition_digest") or definition.get("definition_digest")
    return (
        run_id if isinstance(run_id, str) and run_id else None,
        digest if isinstance(digest, str) and digest else None,
    )

if graceful and not alive():
    write_graceful_result("already_exited", control="graceful")
    print(f"job {job_id} completed before graceful pause; no signal sent")
    raise SystemExit(0)

if not alive():
    write_json(job_dir / "stop.json", {"status": "already_exited", "updated_at": now(), "pid": pid, "pgid": pgid})
    print(f"job {job_id} is not running")
    raise SystemExit(0)

def validate_process_identity() -> None:
    if not alive():
        raise SystemExit("refusing to stop: job process exited")
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

try:
    validate_process_identity()
except (OSError, ValueError, SystemExit) as exc:
    if graceful:
        status = "already_exited" if not alive() else "pause_rejected"
        write_graceful_result(status, control="graceful", reason=str(exc))
    raise

if graceful:
    try:
        state_root = run_root()
    except SystemExit as exc:
        write_graceful_result("pause_rejected", control="graceful", reason=str(exc))
        raise
    state_path = state_root / "pipeline_state.json"
    if not state_path.exists():
        write_graceful_result(
            "pause_rejected", control="graceful", reason="authoritative pipeline state is missing"
        )
        raise SystemExit(
            f"graceful stop rejected: authoritative pipeline state is missing: {state_path}"
        )
    try:
        before_state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        write_graceful_result("pause_rejected", control="graceful", reason=f"state unreadable: {exc}")
        raise SystemExit(f"graceful stop rejected: authoritative state is unreadable: {state_path}") from exc
    before_run_id, before_digest = run_identity(before_state)
    expected_run_id = metadata.get("run_id")
    expected_digest = metadata.get("definition_digest")
    if (
        not before_run_id
        or not before_digest
        or (expected_run_id is not None and expected_run_id != before_run_id)
        or (expected_digest is not None and expected_digest != before_digest)
    ):
        write_graceful_result(
            "pause_rejected",
            control="graceful",
            reason="authoritative Run identity is missing or does not match metadata",
            run_id=before_run_id,
            definition_digest=before_digest,
        )
        raise SystemExit("graceful stop rejected: authoritative Run identity is missing or mismatched")
    pause_command = [
        "uv",
        "run",
        "warp-taskgen",
        "pause",
        "--state-dir",
        str(state_root),
        "--wait",
        "--timeout",
        pause_timeout,
        "--poll-interval",
        pause_poll_interval,
    ]
    try:
        pause_result = subprocess.run(
            pause_command,
            cwd=remote_dir,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            check=False,
        )
    except OSError as exc:
        write_graceful_result("pause_failed", control="graceful", reason=str(exc))
        raise SystemExit(f"graceful stop failed to invoke pause: {exc}") from exc
    if pause_result.stdout:
        print(pause_result.stdout, end="")
    if pause_result.stderr:
        print(pause_result.stderr, end="", file=sys.stderr)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        write_graceful_result("pause_rejected", control="graceful", reason=f"state unreadable: {exc}")
        raise SystemExit(f"graceful stop rejected: authoritative state is unreadable: {state_path}") from exc
    request_match = re.search(r"Pause acknowledged \((pause-[0-9a-f]{32})\)", pause_result.stdout)
    observed_request = state.get("pause_request_id") if isinstance(state, dict) else None
    observed_run_id, observed_digest = run_identity(state)
    paused = isinstance(state, dict) and state.get("status") == "paused"
    if (
        pause_result.returncode != 0
        or request_match is None
        or not paused
        or observed_request != request_match.group(1)
        or observed_run_id != before_run_id
        or observed_digest != before_digest
    ):
        reason = "pause readback was not authoritative paused"
        if pause_result.returncode != 0:
            reason = f"pause command exited {pause_result.returncode}"
        write_graceful_result(
            "pause_rejected",
            control="graceful",
            reason=reason,
            pause_returncode=pause_result.returncode,
            observed_status=state.get("status") if isinstance(state, dict) else None,
            pause_request_id=observed_request,
            run_id=observed_run_id,
            definition_digest=observed_digest,
        )
        raise SystemExit(
            f"graceful stop did not reach authoritative paused state: "
            f"status={state.get('status') if isinstance(state, dict) else 'malformed'}"
        )
    # Re-check every identity field immediately before the existing signal
    # path. A natural exit or PID reuse during the readback must not turn into
    # an unrelated TERM.
    try:
        validate_process_identity()
    except (OSError, ValueError, SystemExit) as exc:
        status = "already_exited" if not alive() else "pause_rejected"
        write_graceful_result(
            status,
            control="graceful",
            reason=str(exc),
            pause_request_id=observed_request,
        )
        raise
    try:
        final_state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        write_graceful_result("pause_rejected", control="graceful", reason=f"final state unreadable: {exc}")
        raise SystemExit("graceful stop rejected: final authoritative state is unreadable") from exc
    final_request = final_state.get("pause_request_id") if isinstance(final_state, dict) else None
    final_run_id, final_digest = run_identity(final_state)
    if (
        not isinstance(final_state, dict)
        or final_state.get("status") != "paused"
        or final_request != observed_request
        or final_run_id != before_run_id
        or final_digest != before_digest
    ):
        write_graceful_result(
            "pause_rejected",
            control="graceful",
            reason="Run changed before TERM; authoritative paused proof is no longer exact",
            pause_request_id=final_request,
            run_id=final_run_id,
            definition_digest=final_digest,
        )
        raise SystemExit("graceful stop rejected: Run changed before TERM")
    # Keep the process identity check as the final read immediately before
    # killpg.  A PID can be reused, or its group/cwd can change, while the
    # authoritative state is being read back.
    try:
        validate_process_identity()
    except (OSError, ValueError, SystemExit) as exc:
        status = "already_exited" if not alive() else "pause_rejected"
        write_graceful_result(
            status,
            control="graceful",
            reason=str(exc),
            pause_request_id=observed_request,
            run_id=final_run_id,
            definition_digest=final_digest,
        )
        raise
    graceful_proof = {
        "control": "graceful",
        "pause_request_id": observed_request,
        "run_id": before_run_id,
        "definition_digest": before_digest,
        "paused_at": now(),
    }
    write_graceful_result(
        "paused",
        control="graceful",
        pause_request_id=observed_request,
        run_id=observed_run_id,
        definition_digest=observed_digest,
    )

try:
    os.killpg(pgid, signal.SIGTERM)
except ProcessLookupError:
    # The process can naturally exit in the small window between the final
    # identity check and killpg.  Do not claim that TERM was sent.
    write_json(job_dir / "stop.json", {"status": "already_exited", "updated_at": now(), "pid": pid, "pgid": pgid, **graceful_proof})
    print(f"job {job_id} completed before TERM; no signal sent")
    raise SystemExit(0)
except OSError as exc:
    write_json(job_dir / "stop.json", {"status": "signal_failed", "updated_at": now(), "pid": pid, "pgid": pgid, "error": str(exc), **graceful_proof})
    raise SystemExit(f"failed to send TERM to job {job_id}: {exc}") from exc

write_json(job_dir / "stop.json", {"status": "term_sent", "updated_at": now(), "pid": pid, "pgid": pgid, **graceful_proof})
deadline = time.time() + timeout
while time.time() < deadline:
    if not alive():
        write_json(job_dir / "exit.json", {"status": "stopped", "updated_at": now(), "pid": pid, "pgid": pgid, "signal": "TERM", **graceful_proof})
        print(f"stopped job {job_id} with TERM")
        raise SystemExit(0)
    time.sleep(0.5)

if not force:
    write_json(job_dir / "stop.json", {"status": "term_timeout", "updated_at": now(), "pid": pid, "pgid": pgid, **graceful_proof})
    raise SystemExit(f"TERM timeout after {timeout}s; rerun with --force to send KILL")

os.killpg(pgid, signal.SIGKILL)
deadline = time.time() + 5
while time.time() < deadline:
    if not alive():
        write_json(job_dir / "exit.json", {"status": "killed", "updated_at": now(), "pid": pid, "pgid": pgid, "signal": "KILL", **graceful_proof})
        print(f"killed job {job_id} with KILL")
        raise SystemExit(0)
    time.sleep(0.5)
write_json(job_dir / "stop.json", {"status": "kill_timeout", "updated_at": now(), "pid": pid, "pgid": pgid, **graceful_proof})
raise SystemExit("KILL timeout after 5s; process group still appears alive")
PY
REMOTE
