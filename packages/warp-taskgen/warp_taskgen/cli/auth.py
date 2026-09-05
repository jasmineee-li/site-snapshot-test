"""CLI auth validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from warp_taskgen.config import has_configured_agent_auth


def _unknown_auth_sites(
    state_dir: Path, *, instances: list[dict[str, Any]] | None = None
) -> list[str]:
    """Return a list of site names whose auth is truly unknown.

    A site is *not* unknown if either:
    - Phase 0c declared ``auth_mechanism.type`` as something other than
      ``"unknown"``, OR
    - ``instances.json`` provides ``agent_auth`` for the site (the static,
      instances.json-driven path supersedes Phase 0c discovery).

    Handles both layouts Phase 0c can produce:
      - flat: ``<state_dir>/phase_0c/AGENT_CONTEXT_<site>.json`` (current)
      - nested: ``<state_dir>/phase_0c/<site>/AGENT_CONTEXT.json`` (future)

    Returns an empty list when the directory is absent or nothing has been
    profiled yet.
    """
    import json as _json

    # Sites with instance-level agent_auth are never unknown.
    instance_auth_sites: set[str] = set()
    if instances:
        for inst in instances:
            if isinstance(inst, dict) and has_configured_agent_auth(inst.get("agent_auth")):
                site_name = inst.get("site_name", "")
                if site_name:
                    instance_auth_sites.add(site_name)

    profiles_dir = state_dir / "phase_0c"
    if not profiles_dir.exists():
        return []

    parse_errors: list[str] = []

    def _check(ctx_path: Path, site_name: str) -> None:
        if site_name in instance_auth_sites:
            return
        if not ctx_path.exists():
            return
        try:
            data = _json.loads(ctx_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            parse_errors.append(f"{ctx_path}: {exc}")
            return
        if not isinstance(data, dict):
            parse_errors.append(f"{ctx_path}: expected JSON object")
            return
        mech = data.get("auth_mechanism")
        if isinstance(mech, dict) and mech.get("type") == "unknown":
            unknown.append(site_name)

    unknown: list[str] = []

    # Flat layout: AGENT_CONTEXT_<site>.json
    for ctx_path in sorted(profiles_dir.glob("AGENT_CONTEXT_*.json")):
        site_name = ctx_path.stem[len("AGENT_CONTEXT_") :]
        _check(ctx_path, site_name)

    # Nested layout: <site>/AGENT_CONTEXT.json
    for site_dir in sorted(profiles_dir.iterdir()):
        if not site_dir.is_dir():
            continue
        _check(site_dir / "AGENT_CONTEXT.json", site_dir.name)

    if parse_errors:
        raise RuntimeError(
            "Failed to read Phase 0c AGENT_CONTEXT artifacts required for the unknown-auth gate:\n"
            + "\n".join(f"  - {error}" for error in parse_errors)
        )

    return sorted(set(unknown))


__all__ = ["_unknown_auth_sites"]
