from __future__ import annotations

import pytest

from scripts.remote_job_decisions import (
    HostTopology,
    normalize_command,
    topology_guard_message,
    topology_issues,
)

R5 = HostTopology(
    advertise_host="203.0.113.10",
    orchestrator_host="172.17.0.1",
    access_mode="remote_direct_restricted",
)
PLAIN = HostTopology(advertise_host="203.0.113.10", orchestrator_host="203.0.113.10")


def _issues(command: str, *, topology: HostTopology = R5) -> tuple[str, ...]:
    return topology_issues(topology, ["bash", "-lc", command])


@pytest.mark.parametrize(
    ("argv", "mode", "expected"),
    [
        (
            ["uv", "run", "warp-taskgen", "phase", "4"],
            "auto",
            ["bash", "-lc", "uv run warp-taskgen phase 4"],
        ),
        (
            ["bash", "-lc", "uv run warp-taskgen phase 4"],
            "auto",
            ["bash", "-lc", "uv run warp-taskgen phase 4"],
        ),
        (
            ["python3", "-c", "print('ok')"],
            "auto",
            ["python3", "-c", "print('ok')"],
        ),
        (
            ["echo", "hello world"],
            "login-shell",
            ["bash", "-lc", "echo 'hello world'"],
        ),
    ],
)
def test_normalize_command_preserves_execution_contract(
    argv: list[str], mode: str, expected: list[str]
) -> None:
    envelope = normalize_command(argv, mode)
    assert envelope["command"] == expected
    assert envelope["execution"]["original_command"] == argv


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            "uv run python -m warp_taskgen.main phase 2c --feasibility-instances instances.smoke.json",
            "--feasibility-instances instances.smoke.json",
        ),
        (
            "uv run warp-taskgen phase 0 --instances instances.scale.json",
            "Phase 0c runs inside Modal sandboxes",
        ),
        (
            "uv run python -m warp_taskgen.main phase 4 "
            "--instances instances.smoke.json --agent-task-timeout 900",
            "Phase 4 uses --instances instances.smoke.json",
        ),
        (
            "uv run python -m warp_taskgen.main phase 4 "
            "--instances instances.scale.json --workers 48 "
            "--agent-task-timeout 900",
            "Top-level Phase 4 does not use --workers",
        ),
        (
            "uv run python -m warp_taskgen.main phase 4 --instances instances.scale.json",
            "--agent-task-timeout explicitly",
        ),
        (
            "uv run python -m warp_taskgen.main resume --instances instances.smoke.json",
            "Resume uses --instances instances.smoke.json",
        ),
        (
            "uv run python -m warp_taskgen.main phase 0 --instances instances.smoke.json && "
            "uv run python -m warp_taskgen.main phase 1 --generate-novel "
            "--sites gitlab,reddit",
            "--host-inventory-instances instances.scale.json",
        ),
        (
            "uv run python -m warp_taskgen.main phase 2 --task-origin new_task "
            "--feasibility-instances instances.scale.json && "
            "uv run python -m warp_taskgen.main phase 3 --sites gitlab,reddit",
            "same --task-origin to Phase 3",
        ),
        (
            "uv run python -m warp_taskgen.main phase 0 "
            "--benchmark vendors/webarena-verified --instances instances.smoke.json",
            "must not use --benchmark vendors/webarena-verified",
        ),
    ],
)
def test_topology_decisions_cover_repeated_remote_guard_variants(
    command: str, expected: str
) -> None:
    assert any(expected in issue for issue in _issues(command))


def test_topology_decisions_accept_valid_chains_and_host_views() -> None:
    assert not _issues(
        "uv run python -m warp_taskgen.main phase 0 "
        "--benchmark /home/ubuntu/vendors/webarena-verified "
        "--instances instances.smoke.json "
        "--host-inventory-instances instances.scale.json && "
        "uv run python -m warp_taskgen.main phase 1 "
        "--benchmark /home/ubuntu/vendors/webarena-verified "
        "--generate-novel --sites gitlab,reddit && "
        "uv run python -m warp_taskgen.main phase 2 "
        "--feasibility-instances instances.scale.json"
    )
    assert not _issues(
        "uv run python -m warp_taskgen.main phase 2 --task-origin new_task "
        "--feasibility-instances instances.scale.json && "
        "uv run python -m warp_taskgen.main phase 3 --task-origin new_task "
        "--sites gitlab,reddit"
    )
    assert not _issues(
        "uv run python -m warp_taskgen.main phase 4 "
        "--instances instances.scale.json --agent-task-timeout 900"
    )


def test_generic_phase4_timeout_guard_applies_without_r5_topology() -> None:
    issues = topology_issues(
        PLAIN,
        ["uv", "run", "python", "-m", "warp_taskgen.main", "phase", "4"],
    )
    assert any("--agent-task-timeout explicitly" in issue for issue in issues)


def test_repo_relative_vendor_guard_is_limited_to_sensitive_topologies() -> None:
    assert not topology_issues(
        PLAIN,
        [
            "uv",
            "run",
            "warp-taskgen",
            "phase",
            "0",
            "--benchmark",
            "vendors/webarena-verified",
        ],
    )


def test_topology_guard_message_retains_operator_context() -> None:
    message = topology_guard_message("/tmp/host.yaml", R5, ("bad command",))
    assert "host_config=/tmp/host.yaml" in message
    assert "Phase 0c is the exception" in message
    assert "- bad command" in message
