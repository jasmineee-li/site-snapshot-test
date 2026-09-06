"""Phase 0d site auth specs: the per-site view of an ``auth_mechanism``.

Owns :class:`AuthBootstrapError`, the :class:`_SiteSpec` derived view, and the
parsers that turn Phase 0c ``AGENT_CONTEXT_*.json`` profiles and instance
``agent_auth`` blocks into the specs Phase 0d can act on. See the
``warp_taskgen.phases.phase_0d_auth_bootstrap`` runner for the phase contract.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from warp_taskgen.config import load_benchmark_config

logger = logging.getLogger(__name__)


class AuthBootstrapError(RuntimeError):
    """Raised when a ``storage_state`` artifact cannot be produced."""


@dataclass
class _SiteSpec:
    """Derived view of one site's auth_mechanism + authentication inputs.

    ``mech_type`` is either ``"storage_state"`` or ``"form_login"``. When the
    site declares ``type == "form_login"`` at top level, ``declared_path`` is
    empty because no artifact path was specified by the benchmark — Phase 0d
    still writes the bootstrapped artifact to its canonical
    ``logs/phase_0d/<site>/storage_state.json`` location, but the runtime
    ``form_login`` dispatcher is still stubbed, so operators must also expose
    the site as ``storage_state`` (with the same ``form_login`` recipe nested)
    to have Phase 3/4 pick up the artifact automatically.
    """

    site_name: str
    mech_type: str
    declared_path: str
    generator_script: str | None
    form_login: dict[str, Any] | None
    per_task_refresh: bool
    credentials: Any
    agent_context_source: Path
    # Extra hashable metadata we fold into the input hash so credential
    # rotations trigger regeneration even when the script path is stable.
    notes: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _load_site_urls(instances_path: str | Path | None) -> dict[str, str]:
    """Return ``{site_name: site_url}`` from an optional BenchmarkConfig file."""
    if not instances_path:
        return {}
    path = Path(instances_path)
    if not path.exists():
        logger.warning(
            "Phase 0d: --instances path %s does not exist; generators will receive site_url=''",
            path,
        )
        return {}
    try:
        config = load_benchmark_config(path)
    except Exception as exc:
        logger.warning("Phase 0d: failed to parse instances file %s: %s", path, exc)
        return {}
    urls: dict[str, str] = {}
    for instance in config.instances:
        urls.setdefault(instance.site_name, instance.site_url)
    return urls


def _collect_storage_state_specs(
    profiles_dir: Path,
    instance_agent_auths: dict[str, dict[str, Any]] | None = None,
):
    """Yield one :class:`_SiteSpec` per site that has an actionable auth recipe.

    Sites qualify when they declare ``auth_mechanism.type`` of either
    ``storage_state`` (any, with or without a generator) or ``form_login`` —
    the two types Phase 0d can materialize an artifact for. Sites with other
    types (``http_basic``, ``none``, ``unknown``, etc.) are ignored.

    ``instance_agent_auths`` is an optional dict mapping site_name to the
    ``agent_auth`` block from ``instances.json``. When Phase 0c does not
    produce ``auth_mechanism``, the instance config is consulted instead.
    """
    if instance_agent_auths is None:
        instance_agent_auths = {}
    seen: set[str] = set()

    if profiles_dir.exists():
        for path in sorted(profiles_dir.glob("AGENT_CONTEXT_*.json")):
            site_name = path.stem.replace("AGENT_CONTEXT_", "")
            if site_name in seen:
                continue
            spec = _spec_from_context(
                site_name, path, instance_agent_auth=instance_agent_auths.get(site_name)
            )
            if spec is not None:
                seen.add(site_name)
                yield spec

        for child in sorted(profiles_dir.iterdir()):
            if not child.is_dir():
                continue
            nested = child / "AGENT_CONTEXT.json"
            if not nested.exists():
                continue
            if child.name in seen:
                continue
            spec = _spec_from_context(
                child.name, nested, instance_agent_auth=instance_agent_auths.get(child.name)
            )
            if spec is not None:
                seen.add(child.name)
                yield spec

    for site_name, auth in sorted(instance_agent_auths.items()):
        if site_name in seen:
            continue
        spec = _spec_from_instance_agent_auth(
            site_name,
            auth,
            context_path=profiles_dir / site_name / "AGENT_CONTEXT.json",
        )
        if spec is not None:
            seen.add(site_name)
            yield spec


def _extract_form_login_recipe(mech: dict[str, Any]) -> dict[str, Any] | None:
    """Pull a form_login recipe dict out of an auth_mechanism, if present.

    A recipe can live in two places per the schema:
    1. Top-level ``form_login`` sub-object (when ``type == "form_login"``).
    2. Nested under ``storage_state.form_login`` (when ``type ==
       "storage_state"`` and the operator wants Phase 0d to produce the
       artifact via a built-in bootstrap instead of a generator_script).

    Returns the recipe dict (with ``success_url_substring`` normalized from a
    legacy ``success_substring`` alias when needed) or ``None`` if no usable
    recipe is declared.
    """
    candidate: Any = None
    if mech.get("type") == "form_login":
        candidate = mech.get("form_login")
    else:
        storage_sub = mech.get("storage_state")
        if isinstance(storage_sub, dict):
            candidate = storage_sub.get("form_login")
    if not isinstance(candidate, dict):
        return None
    required = ("login_url", "username_selector", "password_selector", "submit_selector")
    for key in required:
        val = candidate.get(key)
        if not isinstance(val, str) or not val.strip():
            return None
    # Normalize success_url_substring; fall back to legacy success_substring.
    success = candidate.get("success_url_substring") or candidate.get("success_substring")
    if not isinstance(success, str) or not success.strip():
        return None
    # Return a shallow copy so callers (hashing, bootstrap) see a consistent
    # dict with the canonical success_url_substring key populated.
    normalized = {key: candidate[key].strip() for key in required}
    normalized["success_url_substring"] = success.strip()
    return normalized


def _spec_from_agent_auth(
    site_name: str,
    mech: dict[str, Any],
    *,
    context_path: Path,
    fallback_credentials: Any = None,
) -> _SiteSpec | None:
    mech_type = mech.get("type")
    if mech_type not in ("storage_state", "form_login"):
        return None

    auth_block = mech.get("authentication")
    credentials = auth_block.get("credentials") if isinstance(auth_block, dict) else None
    if credentials is None:
        credentials = fallback_credentials
    form_login_recipe = _extract_form_login_recipe(mech)

    if mech_type == "storage_state":
        sub = mech.get("storage_state") or {}
        if not isinstance(sub, dict):
            return None
        declared_path = sub.get("path")
        if not isinstance(declared_path, str) or not declared_path.strip():
            return None
        generator_script = sub.get("generator_script")
        if generator_script is not None and not isinstance(generator_script, str):
            generator_script = None
        per_task_refresh = bool(sub.get("per_task_refresh"))
        notes = sub.get("notes") if isinstance(sub.get("notes"), str) else None
        return _SiteSpec(
            site_name=site_name,
            mech_type="storage_state",
            declared_path=declared_path.strip(),
            generator_script=generator_script.strip() if generator_script else None,
            form_login=form_login_recipe,
            per_task_refresh=per_task_refresh,
            credentials=credentials,
            agent_context_source=context_path,
            notes=notes,
        )

    if form_login_recipe is None:
        return None
    return _SiteSpec(
        site_name=site_name,
        mech_type="form_login",
        declared_path="",
        generator_script=None,
        form_login=form_login_recipe,
        per_task_refresh=False,
        credentials=credentials,
        agent_context_source=context_path,
        notes=None,
    )


def _spec_from_instance_agent_auth(
    site_name: str,
    auth: dict[str, Any],
    *,
    context_path: Path,
) -> _SiteSpec | None:
    return _spec_from_agent_auth(site_name, auth, context_path=context_path)


def _spec_from_context(
    site_name: str,
    context_path: Path,
    *,
    instance_agent_auth: dict[str, Any] | None = None,
) -> _SiteSpec | None:
    """Parse an AGENT_CONTEXT.json into a _SiteSpec.

    Returns ``None`` for sites whose auth_mechanism is not one Phase 0d can
    act on (currently: ``storage_state`` or ``form_login``).

    When *instance_agent_auth* is provided (from ``instances.json``), it takes
    precedence over any Phase 0c-generated ``auth_mechanism``.
    """
    try:
        data = json.loads(context_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Phase 0d: could not read %s: %s", context_path, exc)
        return None
    if not isinstance(data, dict):
        return None

    # Prefer instance agent_auth (static config) over Phase 0c auth_mechanism.
    mech = (
        instance_agent_auth if isinstance(instance_agent_auth, dict) else data.get("auth_mechanism")
    )
    if not isinstance(mech, dict):
        return None
    fallback_credentials = None
    if not isinstance(instance_agent_auth, dict):
        auth_block = data.get("authentication") or {}
        fallback_credentials = (
            auth_block.get("credentials") if isinstance(auth_block, dict) else None
        )
    return _spec_from_agent_auth(
        site_name,
        mech,
        context_path=context_path,
        fallback_credentials=fallback_credentials,
    )
