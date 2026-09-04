#!/usr/bin/env python3
"""Pure command decisions shared by the remote-job shell adapters.

The remote-job scripts deliberately keep transport and process lifecycle work
in shell.  This module owns the deterministic part that is otherwise easy to
duplicate in tests: command normalization, command-shape analysis, and the
host-topology guard.  It has no project imports so the same file can be copied
to a remote checkout and imported by the detached job bootstrap.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

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
ENTRYPOINTS = {"warp-taskgen", "worldsim", "worldsim.main", "warp_taskgen.main"}
KNOWN_PHASES = {"0", "0c", "1", "2", "2c", "3", "4"}
PHASE_BOOLEAN_OPTIONS = {
    "--skip-feasibility",
    "--generate-novel",
    "--resume",
    "--force",
    "--quiet",
    "--allow-unknown-auth",
    "--skip-host-bound-storage-state-auth",
}


@dataclass(frozen=True)
class HostTopology:
    """The host fields needed by command-safety decisions."""

    advertise_host: str = ""
    orchestrator_host: str = ""
    access_mode: str = ""

    @property
    def topology_sensitive(self) -> bool:
        return bool(
            self.advertise_host
            and self.orchestrator_host
            and self.advertise_host != self.orchestrator_host
            and self.access_mode == "remote_direct_restricted"
        )


def parse_host_config(path: str | Path) -> HostTopology:
    """Read the simple scalar host fields used by the shell scripts."""

    values: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line or line.startswith(" ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")

    advertise_host = values.get("advertise_host") or values.get("orchestrator_host") or ""
    orchestrator_host = values.get("orchestrator_host") or advertise_host
    return HostTopology(
        advertise_host=advertise_host.strip(),
        orchestrator_host=orchestrator_host.strip(),
        access_mode=values.get("access_mode", "").strip(),
    )


def basename(value: str) -> str:
    return Path(value).name


def already_shell(argv: Sequence[str]) -> bool:
    return len(argv) >= 3 and basename(argv[0]) in SHELLS and argv[1] in {"-c", "-lc"}


def managed_command_name(argv: Sequence[str]) -> str:
    if not argv:
        return ""
    if basename(argv[0]) == "env":
        for item in argv[1:]:
            if "=" not in item:
                return basename(item)
        return "env"
    return basename(argv[0])


def should_login_shell_wrap(argv: Sequence[str]) -> bool:
    if not argv or already_shell(argv):
        return False
    if os.path.isabs(argv[0]) or "/" in argv[0]:
        return False
    return managed_command_name(argv) in ENV_MANAGED_COMMANDS


def normalize_command(argv: Sequence[str], mode: str = "auto") -> dict[str, object]:
    """Return the canonical command and its execution metadata."""

    original = list(argv)
    mode = mode.strip().lower() or "auto"
    if mode not in {"auto", "direct", "login-shell"}:
        raise ValueError(
            "WORLDSIM_REMOTE_JOB_EXEC_MODE must be one of auto, direct, or login-shell"
        )

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

    return {
        "command": normalized,
        "execution": {
            "mode": mode,
            "normalized": normalized != original,
            "reason": reason,
            "original_command": original,
        },
    }


def command_tokens(argv: Sequence[str]) -> list[str]:
    if len(argv) >= 3 and argv[0] == "bash" and argv[1] == "-lc":
        try:
            return shlex.split(argv[2])
        except ValueError:
            return list(argv)
    return list(argv)


def is_python_module_entrypoint(tokens: Sequence[str], index: int) -> bool:
    return (
        tokens[index] == "python"
        and index + 2 < len(tokens)
        and tokens[index + 1] == "-m"
        and tokens[index + 2] in {"worldsim.main", "warp_taskgen.main"}
    )


def entrypoint_at(tokens: Sequence[str], index: int) -> tuple[str, int] | None:
    token = tokens[index]
    if token in ENTRYPOINTS:
        return token, index + 1
    if is_python_module_entrypoint(tokens, index):
        return tokens[index + 2], index + 3
    return None


def _phase_segments(argv: Sequence[str]) -> list[list[str]]:
    tokens = command_tokens(argv)
    segments: list[list[str]] = []
    index = 0
    while index < len(tokens):
        entrypoint = entrypoint_at(tokens, index)
        if entrypoint is None or entrypoint[1] >= len(tokens) or tokens[entrypoint[1]] != "phase":
            index += 1
            continue
        end = entrypoint[1] + 1
        while end < len(tokens):
            if end > entrypoint[1] + 1 and entrypoint_at(tokens, end) is not None:
                break
            if tokens[end] in {"&&", "||", ";"}:
                break
            end += 1
        segments.append(tokens[index:end])
        index = end
    return segments


def _segment_phase(segment: Sequence[str]) -> str | None:
    entrypoint = entrypoint_at(segment, 0)
    if entrypoint is None:
        return None
    skip_value = False
    for token in segment[entrypoint[1] + 1 :]:
        if skip_value:
            skip_value = False
            continue
        if token in KNOWN_PHASES:
            return token
        if token.startswith("--"):
            if "=" not in token and token not in PHASE_BOOLEAN_OPTIONS:
                skip_value = True
            continue
        if token.startswith("-"):
            continue
    return None


def command_runs_phase(argv: Sequence[str], phase: str) -> bool:
    return any(_segment_phase(segment) == phase for segment in _phase_segments(argv))


def phase_command(argv: Sequence[str], phase: str) -> list[str] | None:
    for segment in _phase_segments(argv):
        if _segment_phase(segment) == phase:
            return segment
    return None


def command_runs_resume(argv: Sequence[str]) -> bool:
    joined = " ".join(argv)
    return bool(
        re.search(r"(^|\s)(?:warp-taskgen|worldsim)\s+resume(?:\s|$)", joined)
        or re.search(
            r"(^|\s)python\s+-m\s+(?:worldsim|warp_taskgen)\.main\s+resume(?:\s|$)",
            joined,
        )
        or re.search(r"(^|\s)(?:worldsim|warp_taskgen)\.main\s+resume(?:\s|$)", joined)
    )


def command_sets_inline_state_dir(argv: Sequence[str]) -> bool:
    # Shell quoting/escaping may split a literal env-name spelling without
    # changing the conservative substring detector's intended signal.
    joined = re.sub(r"""['"\\]""", "", " ".join(argv))
    return "WORLDSIM_STATE_DIR" in joined or "WARP_TASKGEN_STATE_DIR" in joined


def option_value(argv: Sequence[str], option: str) -> str | None:
    tokens = command_tokens(argv)
    for index, value in enumerate(tokens):
        if value == option and index + 1 < len(tokens):
            return tokens[index + 1]
        if value.startswith(option + "="):
            return value.split("=", 1)[1]
    return None


def option_value_from(text: str | None, option: str) -> str | None:
    if text is None:
        return None
    name = re.escape(option)
    patterns = (rf"{name}=([^\s;&]+)", rf"{name}\s+([^\s;&]+)")
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip("'\"")
    return None


def has_option(text: str | None, option: str) -> bool:
    if text is None:
        return False
    name = re.escape(option)
    return bool(re.search(rf"{name}(?:=|\s|$)", text))


def _is_smoke(value: str | None) -> bool:
    return value is not None and Path(value).name == "instances.smoke.json"


def _is_scale(value: str | None) -> bool:
    return value is not None and Path(value).name == "instances.scale.json"


def topology_issues(
    topology: HostTopology | Mapping[str, str],
    argv: Sequence[str],
    *,
    allow_repo_relative_vendor: bool = False,
) -> tuple[str, ...]:
    """Return all topology/remote-run contract violations for ``argv``."""

    if not isinstance(topology, HostTopology):
        advertise_host = topology.get("advertise_host", "") or topology.get("orchestrator_host", "")
        topology = HostTopology(
            advertise_host=advertise_host,
            orchestrator_host=topology.get("orchestrator_host", "") or advertise_host,
            access_mode=topology.get("access_mode", ""),
        )
    command_text = " ".join(" ".join(argv).split())
    issues: list[str] = []
    phase0_live = command_runs_phase(argv, "0") or command_runs_phase(argv, "0c")
    phase2_live = command_runs_phase(argv, "2") and "--skip-feasibility" not in command_text
    phase2c_live = command_runs_phase(argv, "2c")
    phase4_live = command_runs_phase(argv, "4")
    phase1_command = phase_command(argv, "1")
    phase1_text = (
        " ".join(shlex.quote(token) for token in phase1_command) if phase1_command else None
    )
    resume_live = command_runs_resume(argv)
    resume_phase2c = (
        resume_live
        and "--feasibility-only" in command_text
        and "--skip-feasibility" not in command_text
    )
    resume_phase4 = resume_live and "--feasibility-only" not in command_text

    if topology.topology_sensitive and phase0_live:
        phase0_command = phase_command(argv, "0") or phase_command(argv, "0c") or list(argv)
        phase0_text = " ".join(shlex.quote(token) for token in phase0_command)
        phase0_instances = option_value_from(phase0_text, "--instances")
        if _is_scale(phase0_instances):
            issues.append(
                "Phase 0c runs inside Modal sandboxes and cannot reach "
                "--instances instances.scale.json host-local/orchestrator URLs. "
                "Use an externally reachable/proxied instance file such as "
                "instances.smoke.json for Phase 0/0c."
            )
        if phase1_command and has_option(phase1_text, "--generate-novel"):
            host_inventory_instances = option_value_from(phase0_text, "--host-inventory-instances")
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

    if topology.topology_sensitive and (phase2_live or phase2c_live or resume_phase2c):
        feasibility_instances = option_value(argv, "--feasibility-instances")
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

    if topology.topology_sensitive and phase4_live and _is_smoke(option_value(argv, "--instances")):
        issues.append(
            "Phase 4 uses --instances instances.smoke.json, which points "
            "Browser Use/PVPO traffic at the public advertised host."
        )

    phase4_command = phase_command(argv, "4") or list(argv)
    phase4_text = " ".join(shlex.quote(token) for token in phase4_command)
    if phase4_live and has_option(phase4_text, "--workers"):
        issues.append(
            "Top-level Phase 4 does not use --workers. "
            "Use --phase-4-max-workers for browser-agent concurrency. "
            "--workers is reserved for scripts/run_phase4_process_pool.py."
        )

    if (phase4_live or resume_phase4) and not has_option(phase4_text, "--agent-task-timeout"):
        issues.append(
            "Phase 4 remote jobs must pass --agent-task-timeout explicitly. "
            "Browser Use's default task wall-clock timeout is long enough for "
            "stale CDP/session-start failures to stall a full registered run; "
            "use a bounded infrastructure guard such as --agent-task-timeout 900."
        )

    if (
        phase0_live
        and phase1_command
        and has_option(phase1_text, "--generate-novel")
        and not has_option(phase1_text, "--benchmark")
        and not has_option(phase1_text, "--config")
    ):
        issues.append(
            "Chained Phase 0 -> Phase 1 novel generation must pass --benchmark or "
            "--config on the Phase 1 command. Detached remote jobs should not rely "
            "on implicit manifest discovery after an expensive Phase 0 run."
        )

    phase2_command = phase_command(argv, "2")
    phase3_command = phase_command(argv, "3")
    if phase2_command and phase3_command:
        phase2_text = " ".join(shlex.quote(token) for token in phase2_command)
        phase3_text = " ".join(shlex.quote(token) for token in phase3_command)
        phase2_task_origin = option_value_from(phase2_text, "--task-origin") or "all"
        phase3_task_origin = option_value_from(phase3_text, "--task-origin") or "all"
        if (
            phase2_task_origin in {"existing_task", "new_task"}
            and phase3_task_origin != phase2_task_origin
        ):
            issues.append(
                "Chained Phase 2 -> Phase 3 with --task-origin "
                f"{phase2_task_origin} must pass the same --task-origin to Phase 3. "
                "Otherwise Phase 3 can mix a scoped Phase 2 adversarial set with "
                "unscoped Phase 1 benign tasks and fail on duplicate benchmark IDs "
                "or write contracts for the wrong cohort."
            )

    repo_relative_vendor = bool(
        re.search(
            r"--benchmark(?:=|\s+)(?:\./)?vendors/webarena-verified(?:\s|$)",
            command_text,
        )
    )
    if topology.topology_sensitive and repo_relative_vendor and not allow_repo_relative_vendor:
        issues.append(
            "Remote r5 jobs must not use --benchmark vendors/webarena-verified. "
            "sync_to_host.sh intentionally excludes repo-local vendors/, so that "
            "path can be stale or incomplete while the host-local benchmark source "
            "lives at /home/ubuntu/vendors/webarena-verified. Use the absolute "
            "host-local benchmark path, or set WORLDSIM_ALLOW_REMOTE_REPO_VENDOR_BENCHMARK=1 "
            "only after proving the repo-local vendor tree is complete."
        )

    resume_instances = option_value(argv, "--instances") if resume_live else None
    if topology.topology_sensitive and resume_instances is not None and _is_smoke(resume_instances):
        issues.append(
            "Resume uses --instances instances.smoke.json on a host whose runtime "
            "traffic must use orchestrator_host."
        )
    return tuple(issues)


def topology_guard_message(
    host_config: str | Path, topology: HostTopology, issues: Sequence[str]
) -> str:
    return "\n".join(
        [
            "remote job instance-topology guard blocked this command.",
            f"host_config={host_config}",
            f"advertise_host={topology.advertise_host}",
            f"orchestrator_host={topology.orchestrator_host}",
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


def _emit_normalize(argv: Sequence[str]) -> int:
    mode = os.environ.get("WORLDSIM_REMOTE_JOB_EXEC_MODE", "auto")
    try:
        payload = normalize_command(argv, mode)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    print(encoded)
    return 0


def _emit_topology(host_config: str, argv: Sequence[str]) -> int:
    topology = parse_host_config(host_config)
    if os.environ.get("WORLDSIM_ALLOW_REMOTE_INSTANCE_TOPOLOGY_MISMATCH") == "1":
        return 0
    issues = topology_issues(
        topology,
        argv,
        allow_repo_relative_vendor=os.environ.get("WORLDSIM_ALLOW_REMOTE_REPO_VENDOR_BENCHMARK")
        == "1",
    )
    if not issues:
        return 0
    print(topology_guard_message(host_config, topology, issues), file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: remote_job_decisions.py normalize|topology ...", file=sys.stderr)
        return 2
    command = args.pop(0)
    if command == "normalize":
        if "--" in args:
            separator = args.index("--")
            args = args[separator + 1 :]
        return _emit_normalize(args)
    if command == "topology":
        if len(args) < 2 or args[0] != "--host-config" or "--" not in args[1:]:
            print(
                "usage: remote_job_decisions.py topology --host-config PATH -- COMMAND ...",
                file=sys.stderr,
            )
            return 2
        host_config = args[1]
        separator = args.index("--", 1)
        return _emit_topology(host_config, args[separator + 1 :])
    print(f"unknown decision command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
