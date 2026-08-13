"""Read-only Resume Plan CLI adapter.

The adapter deliberately stops at :func:`warp_taskgen.run_definition.plan_resume`.
It does not resolve a lifecycle transition, clear markers, materialize a child,
or dispatch a phase.  Those operations remain owned by the normal resume
paths; this module is only the structured inspection surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from warp_taskgen.cli.run_identity import _definition_inputs
from warp_taskgen.run_definition import define_run, plan_resume
from warp_taskgen.state import get_state_dir


def dispatch_resume_plan(args: argparse.Namespace, state: dict[str, Any]) -> int:
    """Print a read-only structured Resume Plan and never dispatch a child."""

    try:
        source = define_run(state)
        # Use the immutable definition's effective-input projection as the
        # baseline. Overlaying the persisted envelope itself would cause
        # ``define_run`` to ignore explicit plan-only overrides.
        requested_inputs = source.input_projection()
        explicit = _definition_inputs(args, apply_defaults=False)
        # Parser/control-only fields are not effective Run Definition inputs.
        for field in ("command", "plan", "json"):
            explicit.pop(field, None)
        requested_inputs.update(explicit)
        logs_dir = state.get("logs_dir")
        run_root = (
            Path(str(logs_dir)).expanduser().resolve(strict=False) if logs_dir else get_state_dir()
        )
        plan = plan_resume(
            source,
            state,
            run_root=run_root,
            requested_inputs=requested_inputs,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"Resume plan failed: {exc}", file=sys.stderr)
        return 2

    payload = plan.to_dict()
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "Resume plan: "
            f"mode={plan.mode} action={plan.lifecycle_action} "
            f"target={plan.target_step or 'none'} reason="
            f"{plan.checkpoint_decisions[0].reason_code if plan.checkpoint_decisions else 'unknown'}"
        )
        for decision in plan.checkpoint_decisions:
            print(
                f"  {decision.owner}/{decision.checkpoint_id}: "
                f"{decision.action} ({decision.reason_code})"
            )
        if plan.errors:
            print("  errors: " + ", ".join(plan.errors))
    return 0


__all__ = ["dispatch_resume_plan"]
