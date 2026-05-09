#!/usr/bin/env bash
# Shared helpers for remote WARP Taskgen job scripts.

set -euo pipefail

RJ_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RJ_REPO_ROOT="$(cd "$RJ_LIB_DIR/../.." && pwd)"

rj_die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

rj_require_cmd() {
    command -v "$1" >/dev/null 2>&1 || rj_die "required command not found: $1"
}

rj_abs_path() {
    python3 - "$1" <<'PY'
import os
import sys
from pathlib import Path

raw = os.path.expandvars(os.path.expanduser(sys.argv[1]))
print(Path(raw).resolve(strict=False))
PY
}

rj_shell_quote() {
    python3 - "$1" <<'PY'
import shlex
import sys

print(shlex.quote(sys.argv[1]))
PY
}

rj_shell_quote_many() {
    python3 - "$@" <<'PY'
import shlex
import sys

print(" ".join(shlex.quote(arg) for arg in sys.argv[1:]))
PY
}

rj_json_b64() {
    python3 - "$@" <<'PY'
import base64
import json
import sys

payload = list(sys.argv[1:])
print(base64.b64encode(json.dumps(payload).encode()).decode())
PY
}

rj_object_json_b64() {
    python3 - "$@" <<'PY'
import base64
import json
import sys

payload = json.loads(sys.argv[1])
print(base64.b64encode(json.dumps(payload).encode()).decode())
PY
}

rj_load_host_config() {
    local host_config="$1"
    [[ -n "$host_config" ]] || rj_die "--host-config required"
    [[ -f "$host_config" ]] || rj_die "host config not found: $host_config"

    # Keep this dependency-free for operator scripts. The checked-in host YAML
    # fields used here are simple top-level scalars.
    local exports
    exports="$(python3 - "$host_config" <<'PY'
import shlex
import sys
from pathlib import Path

values = {}
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.split("#", 1)[0].rstrip()
    if not line or line.startswith(" ") or ":" not in line:
        continue
    key, value = line.split(":", 1)
    values[key.strip()] = value.strip().strip("'\"")

host = values.get("advertise_host") or values.get("orchestrator_host")
if not host:
    raise SystemExit("host config missing advertise_host/orchestrator_host")

out = {
    "RJ_HOST_IP": host,
    "RJ_ADVERTISE_HOST": values.get("advertise_host") or host,
    "RJ_ORCHESTRATOR_HOST": values.get("orchestrator_host") or host,
    "RJ_SSH_USER": values.get("ssh_user") or "ubuntu",
    "RJ_COMPOSE_DIR_REMOTE": values.get("compose_dir_remote") or "/home/ubuntu",
}
for key, value in out.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"
    eval "$exports"
}

rj_guard_runtime_instance_topology() {
    local host_config="$1"
    shift

    if [[ "${WORLDSIM_ALLOW_REMOTE_INSTANCE_TOPOLOGY_MISMATCH:-}" == "1" ]]; then
        return 0
    fi

    python3 - "$host_config" "$@" <<'PY'
import os
import re
import shlex
import sys
from pathlib import Path

host_config = Path(sys.argv[1])
argv = sys.argv[2:]

values: dict[str, str] = {}
for raw in host_config.read_text(encoding="utf-8").splitlines():
    line = raw.split("#", 1)[0].rstrip()
    if not line or line.startswith(" ") or ":" not in line:
        continue
    key, value = line.split(":", 1)
    values[key.strip()] = value.strip().strip("'\"")

advertise_host = values.get("advertise_host", "").strip()
orchestrator_host = (values.get("orchestrator_host") or advertise_host).strip()
access_mode = values.get("access_mode", "").strip()

topology_sensitive = (
    bool(advertise_host)
    and bool(orchestrator_host)
    and advertise_host != orchestrator_host
    and access_mode == "remote_direct_restricted"
)

command_text = " ".join(" ".join(argv).split())
_ENTRYPOINTS = {"warp-taskgen", "worldsim", "worldsim.main"}

_KNOWN_PHASES = {"0", "0c", "1", "2", "2c", "3", "4"}
_PHASE_BOOLEAN_OPTIONS = {
    "--skip-feasibility",
    "--generate-novel",
    "--resume",
    "--force",
    "--quiet",
    "--allow-unknown-auth",
    "--skip-host-bound-storage-state-auth",
}


def _command_tokens() -> list[str]:
    if len(argv) >= 3 and argv[0] == "bash" and argv[1] == "-lc":
        try:
            return shlex.split(argv[2])
        except ValueError:
            return argv
    return argv


def _is_python_module_entrypoint(tokens: list[str], index: int) -> bool:
    return (
        tokens[index] == "python"
        and index + 2 < len(tokens)
        and tokens[index + 1] == "-m"
        and tokens[index + 2] == "worldsim.main"
    )


def _entrypoint_at(tokens: list[str], index: int) -> tuple[str, int] | None:
    token = tokens[index]
    if token in _ENTRYPOINTS:
        return token, index + 1
    if _is_python_module_entrypoint(tokens, index):
        return "worldsim.main", index + 3
    return None


def _contains_taskgen_entrypoint(tokens: list[str]) -> bool:
    return any(_entrypoint_at(tokens, index) is not None for index in range(len(tokens)))


def _phase_segments() -> list[list[str]]:
    tokens = _command_tokens()
    segments: list[list[str]] = []
    if not _contains_taskgen_entrypoint(tokens):
        raise SystemExit(0)
    index = 0
    while index < len(tokens):
        entrypoint = _entrypoint_at(tokens, index)
        if entrypoint is None or entrypoint[1] >= len(tokens) or tokens[entrypoint[1]] != "phase":
            index += 1
            continue
        end = entrypoint[1] + 1
        while end < len(tokens):
            if end > entrypoint[1] + 1 and _entrypoint_at(tokens, end) is not None:
                break
            if tokens[end] in {"&&", "||", ";"}:
                break
            end += 1
        segments.append(tokens[index:end])
        index = end
    return segments


def _segment_phase(segment: list[str]) -> str | None:
    entrypoint = _entrypoint_at(segment, 0)
    if entrypoint is None:
        return None
    skip_value = False
    for token in segment[entrypoint[1] + 1 :]:
        if skip_value:
            skip_value = False
            continue
        if token in _KNOWN_PHASES:
            return token
        if token.startswith("--"):
            if "=" not in token and token not in _PHASE_BOOLEAN_OPTIONS:
                skip_value = True
            continue
        if token.startswith("-"):
            continue
    return None


def _runs_phase(phase: str) -> bool:
    return any(_segment_phase(segment) == phase for segment in _phase_segments())


def _phase_command(phase: str) -> str | None:
    for segment in _phase_segments():
        if _segment_phase(segment) == phase:
            return " ".join(shlex.quote(token) for token in segment)
    return None


def _option_value(option: str) -> str | None:
    return _option_value_from(command_text, option)


def _option_value_from(text: str | None, option: str) -> str | None:
    if text is None:
        return None
    name = re.escape(option)
    patterns = (
        rf"{name}=([^\s;&]+)",
        rf"{name}\s+([^\s;&]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip("'\"")
    return None


def _has_option(text: str | None, option: str) -> bool:
    if text is None:
        return False
    name = re.escape(option)
    return bool(re.search(rf"{name}(?:=|\s|$)", text))


def _is_smoke(value: str | None) -> bool:
    if value is None:
        return False
    return Path(value).name == "instances.smoke.json"


def _is_scale(value: str | None) -> bool:
    if value is None:
        return False
    return Path(value).name == "instances.scale.json"


issues: list[str] = []
phase0_live = _runs_phase("0") or _runs_phase("0c")
phase2_live = _runs_phase("2") and "--skip-feasibility" not in command_text
phase2c_live = _runs_phase("2c")
phase4_live = _runs_phase("4")
phase1_command = _phase_command("1")
resume_live = bool(
    re.search(r"\b(?:warp-taskgen|worldsim)\s+resume(?:\s|$)", command_text)
    or re.search(r"\bpython\s+-m\s+worldsim\.main\s+resume(?:\s|$)", command_text)
    or re.search(r"\bworldsim\.main\s+resume(?:\s|$)", command_text)
)
resume_phase2c = (
    resume_live
    and "--feasibility-only" in command_text
    and "--skip-feasibility" not in command_text
)
resume_phase4 = (
    resume_live
    and "--feasibility-only" not in command_text
)

if topology_sensitive and phase0_live:
    phase0_command = _phase_command("0") or _phase_command("0c") or command_text
    phase0_instances = _option_value_from(phase0_command, "--instances")
    if _is_scale(phase0_instances):
        issues.append(
            "Phase 0c runs inside Modal sandboxes and cannot reach "
            "--instances instances.scale.json host-local/orchestrator URLs. "
            "Use an externally reachable/proxied instance file such as "
            "instances.smoke.json for Phase 0/0c."
        )
    if phase1_command and _has_option(phase1_command, "--generate-novel"):
        host_inventory_instances = _option_value_from(
            phase0_command, "--host-inventory-instances"
        )
        if host_inventory_instances is None:
            issues.append(
                "Chained Phase 0 -> Phase 1 novel generation on r5 must pass "
                "--host-inventory-instances instances.scale.json on the Phase 0 "
                "command. Phase 0c browser probes still use --instances "
                "instances.smoke.json, but host-side GitLab/Reddit inventory "
                "enrichment needs the orchestrator-local topology."
            )
        elif _is_smoke(host_inventory_instances):
            issues.append(
                "Phase 0 host-side inventory enrichment uses "
                "--host-inventory-instances instances.smoke.json. Use "
                "instances.scale.json or an equivalent host-local instances file "
                "so Reddit DB and GitLab API inventory reads use orchestrator_host "
                "ports."
            )

if topology_sensitive and (phase2_live or phase2c_live or resume_phase2c):
    feasibility_instances = _option_value("--feasibility-instances")
    if feasibility_instances is None:
        issues.append(
            "Phase 2/2c on this host must pass "
            "--feasibility-instances instances.scale.json explicitly; "
            "the CLI default is instances.smoke.json."
        )
    elif _is_smoke(feasibility_instances):
        issues.append(
            "Phase 2/2c uses --feasibility-instances instances.smoke.json, "
            "which points browser probes at the public advertised host."
        )

if topology_sensitive and phase4_live:
    instances = _option_value("--instances")
    if _is_smoke(instances):
        issues.append(
            "Phase 4 uses --instances instances.smoke.json, which points "
            "Browser Use/PVPO traffic at the public advertised host."
        )

phase4_command = _phase_command("4") or command_text
if phase4_live and _has_option(phase4_command, "--workers"):
    issues.append(
        "Top-level Phase 4 does not use --workers. "
        "Use --phase-4-max-workers for browser-agent concurrency. "
        "--workers is reserved for scripts/run_phase4_process_pool.py."
    )

if (phase4_live or resume_phase4) and not _has_option(phase4_command, "--agent-task-timeout"):
    issues.append(
        "Phase 4 remote jobs must pass --agent-task-timeout explicitly. "
        "Browser Use's default task wall-clock timeout is long enough for "
        "stale CDP/session-start failures to stall a full registered run; "
        "use a bounded infrastructure guard such as --agent-task-timeout 900."
    )

if (
    phase0_live
    and phase1_command
    and _has_option(phase1_command, "--generate-novel")
    and not _has_option(phase1_command, "--benchmark")
    and not _has_option(phase1_command, "--config")
):
    issues.append(
        "Chained Phase 0 -> Phase 1 novel generation must pass --benchmark or "
        "--config on the Phase 1 command. Detached remote jobs should not rely "
        "on implicit manifest discovery after an expensive Phase 0 run."
    )

phase2_command = _phase_command("2")
phase3_command = _phase_command("3")
if phase2_command and phase3_command:
    phase2_task_origin = _option_value_from(phase2_command, "--task-origin") or "all"
    phase3_task_origin = _option_value_from(phase3_command, "--task-origin") or "all"
    if phase2_task_origin in {"existing_task", "new_task"} and (
        phase3_task_origin != phase2_task_origin
    ):
        issues.append(
            "Chained Phase 2 -> Phase 3 with --task-origin "
            f"{phase2_task_origin} must pass the same --task-origin to Phase 3. "
            "Otherwise Phase 3 can mix a scoped Phase 2 adversarial set with "
            "unscoped Phase 1 benign tasks and fail on duplicate benchmark IDs "
            "or write contracts for the wrong cohort."
        )

repo_relative_webarena_verified = bool(
    re.search(
        r"--benchmark(?:=|\s+)(?:\./)?vendors/webarena-verified(?:\s|$)",
        command_text,
    )
)
if (
    topology_sensitive
    and
    repo_relative_webarena_verified
    and os.environ.get("WORLDSIM_ALLOW_REMOTE_REPO_VENDOR_BENCHMARK") != "1"
):
    issues.append(
        "Remote r5 jobs must not use --benchmark vendors/webarena-verified. "
        "sync_to_r5.sh intentionally excludes repo-local vendors/, so that "
        "path can be stale or incomplete while the host-local benchmark source "
        "lives at /home/ubuntu/vendors/webarena-verified. Use the absolute "
        "host-local benchmark path, or set WORLDSIM_ALLOW_REMOTE_REPO_VENDOR_BENCHMARK=1 "
        "only after proving the repo-local vendor tree is complete."
    )

resume_instances = _option_value("--instances") if resume_live else None
if topology_sensitive and resume_instances is not None and _is_smoke(resume_instances):
    issues.append(
        "Resume uses --instances instances.smoke.json on a host whose runtime "
        "traffic must use orchestrator_host."
    )

if not issues:
    raise SystemExit(0)

message = "\n".join(
    [
        "remote job instance-topology guard blocked this command.",
        f"host_config={host_config}",
        f"advertise_host={advertise_host}",
        f"orchestrator_host={orchestrator_host}",
        "On-host browser phases must use the orchestrator host view. Public-IP "
        "instance files can produce false Phase 2c host_unreachable failures, "
        "host-bound storage_state mismatches, and misleading 0-admission artifacts.",
        "Phase 0c is the exception: its profiling sandboxes run outside the host "
        "and must use externally reachable/proxied URLs.",
        *[f"- {issue}" for issue in issues],
        "Use instances.scale.json for on-host Phase 2c/4 and instances.smoke.json "
        "or an equivalent public/proxy instance file for Phase 0c. Set "
        "WORLDSIM_ALLOW_REMOTE_INSTANCE_TOPOLOGY_MISMATCH=1 only for a deliberate "
        "topology experiment.",
    ]
)
print(message, file=sys.stderr)
raise SystemExit(2)
PY
}

rj_prepare_connection() {
    local host_config="$1"
    local ssh_key_raw="${2:-}"
    rj_load_host_config "$host_config"

    if [[ -z "$ssh_key_raw" ]]; then
        ssh_key_raw="${SSH_KEY:-$HOME/.ssh/webarena-key.pem}"
    fi
    RJ_SSH_KEY="$(rj_abs_path "$ssh_key_raw")"
    RJ_SSH_BIN="${REMOTE_JOBS_SSH_BIN:-ssh}"
    RJ_SCP_BIN="${REMOTE_JOBS_SCP_BIN:-scp}"
    RJ_RSYNC_BIN="${REMOTE_JOBS_RSYNC_BIN:-rsync}"
    RJ_SSH_TARGET="${RJ_SSH_USER}@${RJ_HOST_IP}"
    RJ_SSH_OPTS=(
        -i "$RJ_SSH_KEY"
        -o StrictHostKeyChecking=accept-new
        -o ServerAliveInterval=30
        -o ServerAliveCountMax=120
        -o ConnectTimeout=15
    )
}

rj_default_remote_dir() {
    printf '%s/browser-sim\n' "$RJ_COMPOSE_DIR_REMOTE"
}

rj_ssh() {
    "$RJ_SSH_BIN" "${RJ_SSH_OPTS[@]}" "$RJ_SSH_TARGET" "$@"
}

rj_ssh_bash() {
    local remote_cmd
    remote_cmd='tmp=$(mktemp /tmp/worldsim-remote-job.XXXXXX); cat > "$tmp"; bash "$tmp"'
    if (($# > 0)); then
        remote_cmd+=" $(rj_shell_quote_many "$@")"
    fi
    remote_cmd+='; rc=$?; rm -f "$tmp"; exit "$rc"'
    "$RJ_SSH_BIN" "${RJ_SSH_OPTS[@]}" "$RJ_SSH_TARGET" "$remote_cmd"
}

rj_resolve_job_id() {
    local remote_dir="$1"
    local job_id="$2"
    local latest="$3"
    local name="$4"

    if [[ -n "$job_id" ]]; then
        printf '%s\n' "$job_id"
        return
    fi

    rj_ssh_bash "$remote_dir" "$latest" "$name" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
latest="$2"
name="$3"
python3 - "$remote_dir" "$latest" "$name" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "logs" / "remote_jobs"
latest = sys.argv[2] == "1"
name = sys.argv[3]

if not root.exists():
    raise SystemExit("no remote job registry found")

jobs = []
for metadata_path in root.glob("*/metadata.json"):
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if name and data.get("name") != name:
        continue
    jobs.append((metadata_path.stat().st_mtime, metadata_path.parent.name))

if not jobs:
    if name:
        raise SystemExit(f"no remote jobs found for name: {name}")
    raise SystemExit("no remote jobs found")

jobs.sort(reverse=True)
print(jobs[0][1])
PY
REMOTE
}

rj_validate_job_selector() {
    local job_id="$1"
    local latest="$2"
    local name="$3"
    local allow_name="${4:-1}"
    local count=0
    [[ -n "$job_id" ]] && count=$((count + 1))
    [[ "$latest" -eq 1 ]] && count=$((count + 1))
    [[ -n "$name" ]] && count=$((count + 1))
    [[ "$allow_name" -eq 0 && -n "$name" ]] && rj_die "--name is not supported here; use --job-id"
    [[ "$count" -eq 1 ]] || rj_die "select exactly one of --job-id, --latest, or --name"
}
