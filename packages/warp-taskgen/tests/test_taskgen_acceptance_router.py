from __future__ import annotations

from scripts.taskgen_acceptance_router import requires_acceptance


def test_canonical_package_changes_run_acceptance() -> None:
    assert requires_acceptance(["packages/warp-taskgen/warp_taskgen/main.py"])


def test_acceptance_entrypoints_run_acceptance() -> None:
    assert requires_acceptance(["scripts/accept_taskgen.sh"])
    assert requires_acceptance([".github/workflows/taskgen-acceptance.yml"])


def test_unrelated_changes_skip_acceptance() -> None:
    assert not requires_acceptance(["README.md", "eval_awareness/probe.py"])


def test_router_accepts_checkout_relative_prefixes() -> None:
    assert requires_acceptance(["./packages/warp-taskgen/README.md"])
