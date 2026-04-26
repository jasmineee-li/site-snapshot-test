"""Phase 0d: Storage-state auth artifact bootstrap.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Phase 0d"
subsection. Runs once between Phase 0c and Phase 3 to materialize
``storage_state`` artifacts declared in each site's ``auth_mechanism``.

Per-site dispatch precedence (first match wins):

1. ``generator_script`` is declared -> import + invoke the script.
2. A ``form_login`` recipe is declared (either ``auth_mechanism.type ==
   "form_login"`` OR ``auth_mechanism.storage_state.form_login``) -> use the
   built-in Playwright form-login bootstrapper. Requires
   ``authentication.credentials`` to have string ``username`` + ``password``.
3. The declared ``storage_state.path`` already exists in the benchmark tree
   and is non-empty JSON -> trust it and stamp a completion marker so Phase 3
   can proceed.
4. Otherwise -> skip with a warning.

Either dispatch path produces ``logs/phase_0d/<site>/storage_state.json`` and a
content-addressed ``completion.json``. Idempotent: we skip when the marker's
input hash matches the current inputs (credentials + recipe bytes).

Non-goals:

- Does **not** start, stop, or snapshot benchmark instances. The generator is
  expected to log in against a site the user is already running.
- Does **not** cope with flaky interactive login pages (CAPTCHA, OTP, 2FA).
  If the built-in bootstrapper cannot complete, switch the site to a
  ``pre_auth_script`` or author a ``generator_script`` that handles the
  challenge out-of-band.
- Does **not** handle ``per_task_refresh`` (deferred — see plan §6).

Contract for ``generator_script`` (benchmark-agnostic):

- The value is a filesystem path relative to ``benchmark_root`` (absolute paths
  are also accepted and bypass the root).
- The file is a Python module importable in-process. It **must** expose a
  top-level ``generate`` callable with signature::

      def generate(credentials, site_url, output_path, **kwargs) -> None
      # or
      async def generate(credentials, site_url, output_path, **kwargs) -> None

  ``credentials`` is the ``authentication.credentials`` dict verbatim from
  ``AGENT_CONTEXT`` (may be ``None``). ``site_url`` is the live benchmark base
  URL for the site. ``output_path`` is an absolute :class:`pathlib.Path` the
  callable must write to (e.g. via Playwright's
  ``BrowserContext.storage_state(path=output_path)``).

- The callable is expected to produce a non-empty JSON file at
  ``output_path``. Phase 0d verifies the file exists and is non-empty JSON
  before marking the artifact complete; otherwise it raises
  :class:`AuthBootstrapError`.

Storage layout::

    logs/phase_0d/
      <site>/
        storage_state.json        # artifact Playwright will later load
        completion.json           # {input_hash, generated_at, site, script}
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from worldsim.agent_auth import (
    read_storage_state_payload,
    safe_phase_0d_site_name,
    storage_state_preflight_error_for_payload,
    storage_state_recorded_hosts,
)
from worldsim.atomic_io import write_json_atomic
from worldsim.config import BenchmarkConfig, has_configured_agent_auth
from worldsim.state import get_state_dir, save_state

logger = logging.getLogger(__name__)


# Bumped whenever ``probe_authenticated`` semantics or the meta-sidecar
# liveness fields change. Older sidecars with a smaller value force re-mint
# even when the input hash and TTL would otherwise allow reuse.
CURRENT_VALIDATOR_VERSION = 1

# Skip the live ``probe_authenticated`` call when the sidecar's
# ``last_validated_at`` is fresher than this. 30 minutes is conservative
# enough that a server-side session GC inside the window costs at most one
# re-mint per task batch, while avoiding a probe HTTP call on every load.
LIVENESS_SOFT_TTL_SECONDS = 30 * 60

# Operator break-glass: when set to ``true``/``1``, every Phase 0d
# load-or-mint call re-mints regardless of cache state.
FORCE_REMINT_ENV = "WORLDSIM_STORAGE_STATE_FORCE_REMINT"


class AuthBootstrapError(RuntimeError):
    """Raised when a ``storage_state`` artifact cannot be produced."""


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    failpoint_base: str | None = None,
) -> None:
    write_json_atomic(path, payload, failpoint_base=failpoint_base)


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


async def run(args: argparse.Namespace) -> int:
    """Phase 0d entrypoint.

    Args:
        args: argparse namespace. Required fields: ``benchmark`` (Path to the
            benchmark codebase, used as the root for ``generator_script`` path
            resolution). Optional: ``instances`` (BenchmarkConfig JSON) — when
            provided, the live ``site_url`` is looked up per site and passed to
            the generator; otherwise ``site_url`` is ``""`` and the generator
            must resolve its own endpoint.

    Returns:
        ``0`` on success, non-zero on hard failure.
    """
    benchmark: Path | None = getattr(args, "benchmark", None)
    if benchmark is None:
        logger.error("Phase 0d requires --benchmark for generator_script resolution")
        return 1
    benchmark = Path(benchmark).resolve()
    phase_state_metadata = _phase_0d_state_metadata(
        benchmark=benchmark,
        instances=getattr(args, "instances", None),
    )

    state_dir = get_state_dir()
    profiles_dir = state_dir / "phase_0c"

    # Build maps from instances.json before the Phase 0c gate. A generated
    # instances file is now the preferred auth source for rigor runs: it binds
    # site_url, reset_endpoint, and storage_state generation to the same
    # orchestrator-local host view.
    site_urls = _load_site_urls(getattr(args, "instances", None))
    instance_agent_auths: dict[str, dict[str, Any]] = {}
    instances_path = getattr(args, "instances", None)
    if instances_path is not None:
        try:
            config = BenchmarkConfig.model_validate_json(Path(instances_path).read_text())
            for inst in config.instances:
                if has_configured_agent_auth(inst.agent_auth):
                    instance_agent_auths.setdefault(inst.site_name, inst.agent_auth)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Phase 0d: failed to load instances.json agent_auth from %s: %s",
                instances_path,
                exc,
            )

    if not profiles_dir.exists() and not instance_agent_auths:
        logger.error(
            "Phase 0d requires Phase 0c output at %s or an --instances file with "
            "storage_state/form_login agent_auth recipes",
            profiles_dir,
        )
        return 1

    output_dir = state_dir / "phase_0d"
    output_dir.mkdir(parents=True, exist_ok=True)

    site_specs = list(_collect_storage_state_specs(profiles_dir, instance_agent_auths))
    if not site_specs:
        logger.info(
            "Phase 0d: no sites declare auth_mechanism.type in "
            "{'storage_state','form_login'}; nothing to do"
        )
        save_state(
            "phase_0d",
            status="complete",
            output_dir=str(output_dir),
            generated=[],
            skipped=[],
            **phase_state_metadata,
        )
        return 0

    save_state("phase_0d", status="running", output_dir=str(output_dir), **phase_state_metadata)

    generated: list[str] = []
    skipped: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    for spec in site_specs:
        try:
            artifact_path = phase_0d_artifact_path(spec.site_name, state_dir=state_dir)
            completion_path = phase_0d_completion_path(spec.site_name, state_dir=state_dir)
        except ValueError as exc:
            logger.error("Phase 0d failed for site %r: %s", spec.site_name, exc)
            failures.append((spec.site_name, str(exc)))
            continue
        site_output = artifact_path.parent
        site_output.mkdir(parents=True, exist_ok=True)

        site_url = site_urls.get(spec.site_name, "")
        try:
            input_hash = _compute_input_hash(spec, benchmark_root=benchmark, site_url=site_url)
        except AuthBootstrapError as exc:
            logger.error("Phase 0d failed for site %r: %s", spec.site_name, exc)
            failures.append((spec.site_name, str(exc)))
            continue
        if _is_idempotent_skip(artifact_path, completion_path, input_hash) and not _force_remint():
            try:
                _validate_storage_state_artifact(
                    spec=spec,
                    artifact_path=artifact_path,
                    site_url=site_url,
                )
            except AuthBootstrapError as exc:
                logger.warning(
                    "Phase 0d: existing artifact for site %r matches input hash but "
                    "failed host validation; regenerating: %s",
                    spec.site_name,
                    exc,
                )
            else:
                probe_instance = {
                    "site_name": spec.site_name,
                    "site_url": site_url,
                    "agent_auth": instance_agent_auths.get(spec.site_name) or {},
                }
                if _liveness_check_passes(
                    site_name=spec.site_name,
                    artifact_path=artifact_path,
                    instance=probe_instance,
                ):
                    logger.info(
                        "Phase 0d: skipping site %r (artifact present, hash matches, liveness ok)",
                        spec.site_name,
                    )
                    skipped.append({"site": spec.site_name, "reason": "up_to_date"})
                    continue
                logger.warning(
                    "[phase_0d] storage_state for site=%s failed liveness probe; re-minting",
                    spec.site_name,
                )
        try:
            dispatch = _choose_dispatch(spec, benchmark_root=benchmark)
        except AuthBootstrapError as exc:
            logger.error("Phase 0d failed for site %r: %s", spec.site_name, exc)
            failures.append((spec.site_name, str(exc)))
            continue

        if dispatch == "generator_script":
            try:
                await _run_generator(
                    spec=spec,
                    site_url=site_url,
                    benchmark_root=benchmark,
                    output_path=artifact_path,
                )
            except AuthBootstrapError as exc:
                logger.error("Phase 0d failed for site %r: %s", spec.site_name, exc)
                failures.append((spec.site_name, str(exc)))
                continue
            except Exception as exc:
                logger.exception("Phase 0d unexpected failure for site %r", spec.site_name)
                failures.append((spec.site_name, repr(exc)))
                continue
        elif dispatch == "form_login":
            try:
                await _bootstrap_via_form_login(
                    spec=spec,
                    site_url=site_url,
                    output_path=artifact_path,
                )
            except AuthBootstrapError as exc:
                logger.error("Phase 0d form_login failed for site %r: %s", spec.site_name, exc)
                failures.append((spec.site_name, str(exc)))
                continue
            except Exception as exc:
                logger.exception(
                    "Phase 0d form_login unexpected failure for site %r", spec.site_name
                )
                failures.append((spec.site_name, repr(exc)))
                continue
        elif dispatch == "trust_path":
            try:
                _trust_declared_path(
                    spec=spec,
                    benchmark_root=benchmark,
                    output_path=artifact_path,
                )
            except AuthBootstrapError as exc:
                logger.error("Phase 0d trust-path failed for site %r: %s", spec.site_name, exc)
                failures.append((spec.site_name, str(exc)))
                continue
        else:  # dispatch == "skip"
            reason = "no_recipe" if spec.mech_type == "form_login" else "no_generator_script"
            logger.warning(
                "Phase 0d: site %r has no generator_script, no form_login recipe, and no "
                "pre-staged artifact at %r — skipping (Phase 3 will fail with "
                "AuthArtifactMissingError if it attempts to load the missing artifact)",
                spec.site_name,
                spec.declared_path or "<unspecified>",
            )
            skipped.append({"site": spec.site_name, "reason": reason})
            continue

        try:
            _validate_storage_state_artifact(
                spec=spec,
                artifact_path=artifact_path,
                site_url=site_url,
            )
        except AuthBootstrapError as exc:
            logger.error("Phase 0d failed for site %r: %s", spec.site_name, exc)
            failures.append((spec.site_name, str(exc)))
            continue

        _write_json_atomic(
            completion_path,
            {
                "site": spec.site_name,
                "input_hash": input_hash,
                "artifact_path": str(artifact_path),
                "dispatch": dispatch,
                "generator_script": spec.generator_script,
                "form_login": spec.form_login,
                "agent_context_source": str(spec.agent_context_source),
                "site_url": site_url,
            },
            failpoint_base="phase_0d.completion",
        )
        from worldsim.storage_state_preflight import write_storage_state_meta

        write_storage_state_meta(artifact_path, mechanism=dispatch)
        generated.append(spec.site_name)
        logger.info(
            "Phase 0d: generated %s for site %r via %s",
            artifact_path,
            spec.site_name,
            dispatch,
        )

    if failures:
        save_state(
            "phase_0d",
            status="failed",
            output_dir=str(output_dir),
            generated=generated,
            skipped=skipped,
            failures=[{"site": s, "error": e} for s, e in failures],
            **phase_state_metadata,
        )
        return 1

    save_state(
        "phase_0d",
        status="complete",
        output_dir=str(output_dir),
        generated=generated,
        skipped=skipped,
        **phase_state_metadata,
    )
    logger.info(
        "Phase 0d complete — %d generated, %d skipped",
        len(generated),
        len(skipped),
    )
    return 0


async def reacquire_storage_state(
    *,
    site_name: str,
    instance: dict[str, Any],
    benchmark_root: Path | None = None,
) -> Path:
    """Re-mint one site's canonical Phase-0d ``storage_state.json``.

    This is the runtime repair path for consumers that prove an existing
    artifact is no longer accepted by the live benchmark. It deliberately
    reuses Phase 0d's generator/form-login machinery instead of duplicating
    login flows in later phases.
    """
    raw_site = str(site_name).strip().lower()
    site, site_error = safe_phase_0d_site_name(raw_site)
    if site_error is not None or site is None:
        raise AuthBootstrapError(site_error or "site_name must be non-empty")
    if not isinstance(instance, dict):
        raise AuthBootstrapError("instance must be a dict")

    auth = instance.get("agent_auth")
    if not isinstance(auth, dict):
        raise AuthBootstrapError(f"site {site!r} has no agent_auth block")
    auth_type = str(auth.get("type") or "").strip()
    if auth_type not in {"storage_state", "form_login"}:
        raise AuthBootstrapError(
            f"site {site!r} uses agent_auth type {auth_type!r}; cannot reacquire storage_state"
        )

    storage_state = auth.get("storage_state")
    storage_block = storage_state if isinstance(storage_state, dict) else {}
    generator_script = storage_block.get("generator_script")
    if generator_script is not None and not isinstance(generator_script, str):
        generator_script = None
    declared_path = storage_block.get("path")
    if not isinstance(declared_path, str) or not declared_path.strip():
        declared_path = str(phase_0d_artifact_path(site))

    form_login = _extract_form_login_recipe(auth)
    if generator_script:
        script_path = Path(generator_script)
        if benchmark_root is None and not script_path.is_absolute():
            raise AuthBootstrapError(
                f"site {site!r} declares relative generator_script {generator_script!r}; "
                "benchmark_root is required to reacquire storage_state"
            )
    elif form_login is None:
        raise AuthBootstrapError(
            f"site {site!r} has no generator_script or form_login recipe; "
            "cannot reacquire storage_state"
        )

    authentication = auth.get("authentication")
    credentials = authentication.get("credentials") if isinstance(authentication, dict) else None
    output_path = phase_0d_instance_artifact_path(site, instance)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    spec = _SiteSpec(
        site_name=site,
        mech_type=auth_type,
        declared_path=declared_path.strip(),
        generator_script=generator_script.strip() if generator_script else None,
        form_login=form_login,
        per_task_refresh=False,
        credentials=credentials,
        agent_context_source=output_path.parent / "runtime_reacquire_context.json",
    )
    site_url = str(instance.get("site_url") or "").strip()

    try:
        if spec.generator_script:
            await _run_generator(
                spec=spec,
                site_url=site_url,
                benchmark_root=Path(benchmark_root) if benchmark_root is not None else Path("/"),
                output_path=tmp_path,
            )
        else:
            await _bootstrap_via_form_login(
                spec=spec,
                site_url=site_url,
                output_path=tmp_path,
            )
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise

    logger.info("Phase 0d: reacquired storage_state for site %r at %s", site, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


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
        config = BenchmarkConfig.model_validate_json(path.read_text())
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


def _compute_input_hash(spec: _SiteSpec, *, benchmark_root: Path, site_url: str) -> str:
    """Content-hash the inputs that determine whether regeneration is needed.

    Changes in any of the following invalidate the artifact: credentials,
    the generator_script source bytes, the declared path, per_task_refresh,
    and the form_login recipe (selectors + success_url_substring) when present.
    """
    hasher = hashlib.sha256()
    hasher.update(b"phase_0d|v2\n")
    hasher.update(spec.site_name.encode("utf-8") + b"\n")
    hasher.update(spec.mech_type.encode("utf-8") + b"\n")
    hasher.update(spec.declared_path.encode("utf-8") + b"\n")
    hasher.update(b"ptr:" + (b"1" if spec.per_task_refresh else b"0") + b"\n")
    hasher.update(
        b"creds:"
        + json.dumps(spec.credentials, sort_keys=True, default=str).encode("utf-8")
        + b"\n"
    )
    if spec.generator_script:
        hasher.update(b"script:" + spec.generator_script.encode("utf-8") + b"\n")
        script_path = _resolve_generator_path(spec.generator_script, benchmark_root)
        if script_path is not None and script_path.exists():
            try:
                hasher.update(b"bytes:" + script_path.read_bytes())
            except OSError:
                hasher.update(b"bytes:<unreadable>")
        else:
            hasher.update(b"bytes:<missing>")
    if spec.form_login is not None:
        hasher.update(
            b"form_login:"
            + json.dumps(spec.form_login, sort_keys=True, default=str).encode("utf-8")
            + b"\n"
        )
    # When the dispatch falls through to trust-path mode, the trusted artifact
    # bytes are part of the effective runtime input. Hash them so in-place auth
    # rotations invalidate the completion marker instead of silently reusing the
    # old copied storage_state.json.
    if not spec.generator_script and spec.form_login is None and spec.declared_path:
        declared = Path(spec.declared_path)
        resolved = (
            declared
            if declared.is_absolute()
            else _resolve_benchmark_relative_path(declared, benchmark_root)
        )
        hasher.update(b"trusted_path:" + str(resolved).encode("utf-8") + b"\n")
        if resolved.exists():
            try:
                hasher.update(b"trusted_bytes:" + resolved.read_bytes())
            except OSError:
                hasher.update(b"trusted_bytes:<unreadable>")
        else:
            hasher.update(b"trusted_bytes:<missing>")
    hasher.update(b"site_url:" + site_url.encode("utf-8") + b"\n")
    return hasher.hexdigest()


def _resolve_generator_path(generator_script: str, benchmark_root: Path) -> Path | None:
    """Resolve a generator_script path; absolute wins, else joined to benchmark_root."""
    if not generator_script:
        return None
    candidate = Path(generator_script)
    if candidate.is_absolute():
        return candidate
    return _resolve_benchmark_relative_path(candidate, benchmark_root)


def _resolve_benchmark_relative_path(path: Path, benchmark_root: Path) -> Path:
    """Resolve a benchmark-relative path and reject escapes outside benchmark_root."""
    resolved = (benchmark_root / path).resolve()
    try:
        resolved.relative_to(benchmark_root)
    except ValueError as exc:
        raise AuthBootstrapError(
            f"path {str(path)!r} resolves outside benchmark root {benchmark_root}"
        ) from exc
    return resolved


def _is_idempotent_skip(
    artifact_path: Path,
    completion_path: Path,
    current_hash: str,
) -> bool:
    """Return True when the existing artifact matches the current input hash."""
    if not artifact_path.exists() or not completion_path.exists():
        return False
    try:
        raw = artifact_path.read_text(encoding="utf-8")
        if not raw.strip():
            return False
        json.loads(raw)  # must be valid JSON
    except (OSError, json.JSONDecodeError):
        return False
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return completion.get("input_hash") == current_hash


async def _run_generator(
    *,
    spec: _SiteSpec,
    site_url: str,
    benchmark_root: Path,
    output_path: Path,
) -> None:
    """Import and invoke the generator script, awaiting coroutine returns."""
    if spec.generator_script is None:
        raise AuthBootstrapError("generator_script is None")

    script_path = _resolve_generator_path(spec.generator_script, benchmark_root)
    if script_path is None or not script_path.exists():
        raise AuthBootstrapError(
            f"generator_script {spec.generator_script!r} does not exist (resolved to {script_path})"
        )

    module = _load_module(script_path, spec.site_name)
    generate = getattr(module, "generate", None)
    if generate is None or not callable(generate):
        raise AuthBootstrapError(
            f"generator_script {spec.generator_script!r} does not expose a callable 'generate'"
        )

    _verify_generate_signature(generate, spec.generator_script)

    # Remove any stale artifact so a partial write cannot satisfy the
    # post-generation non-empty check.
    if output_path.exists():
        try:
            output_path.unlink()
        except OSError:
            pass

    try:
        result = generate(
            credentials=spec.credentials,
            site_url=site_url,
            output_path=output_path,
        )
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # pragma: no cover - surfaced via AuthBootstrapError
        raise AuthBootstrapError(
            f"generator_script {spec.generator_script!r} raised: {exc!r}"
        ) from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise AuthBootstrapError(
            f"generator_script {spec.generator_script!r} did not write a non-empty "
            f"artifact at {output_path}"
        )

    try:
        json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthBootstrapError(
            f"generator_script {spec.generator_script!r} wrote invalid JSON to {output_path}: {exc}"
        ) from exc


def _verify_generate_signature(generate: Any, script_display_name: str) -> None:
    """Fail fast when the generator does not accept the required kwargs.

    The contract is keyword-only: ``credentials``, ``site_url``, ``output_path``.
    We allow ``**kwargs`` (so generators can grow extra context later without
    breaking the orchestrator) but require the three canonical names to be
    acceptable.
    """
    try:
        sig = inspect.signature(generate)
    except (TypeError, ValueError):
        # Builtins / C-extension callables — assume the caller knows what it is doing.
        return

    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    required = ("credentials", "site_url", "output_path")
    for name in required:
        param = params.get(name)
        if param is not None:
            if param.kind == inspect.Parameter.POSITIONAL_ONLY and not has_var_kw:
                raise AuthBootstrapError(
                    f"generator_script {script_display_name!r} parameter {name!r} is positional-only; "
                    "Phase 0d calls generate(credentials=..., site_url=..., output_path=...). "
                    "Make it keyword-capable or accept **kwargs."
                )
            continue
        if has_var_kw:
            continue
        raise AuthBootstrapError(
            f"generator_script {script_display_name!r} 'generate' is missing required "
            f"parameter {name!r}; signature must accept (credentials, site_url, output_path)"
        )


def _load_module(script_path: Path, site_name: str) -> Any:
    """Load a generator_script as an isolated module by reading + exec'ing the source.

    We deliberately bypass ``importlib.machinery.SourceFileLoader`` because it
    memoizes compiled bytecode under ``<script>/__pycache__/*.pyc`` keyed by
    source ``st_mtime_ns``. Two rapid successive writes to the same script can
    share an mtime_ns, causing subsequent loads to return stale bytecode and
    silently regenerate auth artifacts from the prior script body (verified
    against CPython 3.12's ``SourceFileLoader.source_to_code`` path).

    Instead we read the source bytes directly, ``compile`` them, and ``exec``
    into a fresh ``types.ModuleType``. The synthetic module name embeds a
    content hash so a fresh ``sys.modules`` slot is used per revision — useful
    for diagnostics, even though we never re-resolve via import semantics.
    """
    try:
        source_bytes = script_path.read_bytes()
    except OSError as exc:
        raise AuthBootstrapError(f"cannot read generator_script at {script_path}: {exc}") from exc

    content_digest = hashlib.sha256(source_bytes).hexdigest()[:12]
    module_name = (
        f"worldsim._phase_0d_generator_{site_name}_{abs(hash(str(script_path)))}_{content_digest}"
    )

    import types

    module = types.ModuleType(module_name)
    module.__file__ = str(script_path)
    module.__name__ = module_name

    try:
        code = compile(source_bytes, str(script_path), "exec")
    except SyntaxError as exc:
        raise AuthBootstrapError(
            f"generator_script at {script_path} has a syntax error: {exc}"
        ) from exc

    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise AuthBootstrapError(
            f"generator_script at {script_path} failed to import: {exc!r}"
        ) from exc
    return module


# ---------------------------------------------------------------------------
# Dispatch selection + form_login bootstrap
# ---------------------------------------------------------------------------


def _choose_dispatch(spec: _SiteSpec, *, benchmark_root: Path) -> str:
    """Decide which bootstrap path to run for ``spec``.

    Returns one of:
    - ``"generator_script"`` — run the declared generator module.
    - ``"form_login"`` — run the built-in Playwright bootstrapper.
    - ``"trust_path"`` — the declared ``storage_state.path`` already exists
      and is non-empty JSON; stamp completion without regenerating.
    - ``"skip"`` — nothing actionable; warn and move on.

    Precedence matches the module docstring:
    generator_script > form_login > trust_path > skip.
    """
    if spec.generator_script:
        return "generator_script"
    if spec.form_login is not None:
        return "form_login"
    if spec.mech_type == "storage_state" and spec.declared_path:
        declared = Path(spec.declared_path)
        resolved = (
            declared
            if declared.is_absolute()
            else _resolve_benchmark_relative_path(declared, benchmark_root)
        )
        try:
            if resolved.exists() and resolved.stat().st_size > 0:
                # Validate JSON shape before trusting.
                json.loads(resolved.read_text(encoding="utf-8"))
                return "trust_path"
        except (OSError, json.JSONDecodeError):
            pass
    return "skip"


def _trust_declared_path(
    *,
    spec: _SiteSpec,
    benchmark_root: Path,
    output_path: Path,
) -> None:
    """Copy the benchmark-declared storage_state into Phase 0d's output location.

    We materialize a copy (rather than a symlink or reference) so that
    subsequent runs using the canonical ``logs/phase_0d/<site>/storage_state.json``
    path behave identically to the form_login / generator_script paths.
    """
    declared = Path(spec.declared_path)
    resolved = (
        declared
        if declared.is_absolute()
        else _resolve_benchmark_relative_path(declared, benchmark_root)
    )
    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuthBootstrapError(
            f"trust-path failed: cannot read declared storage_state at {resolved}: {exc}"
        ) from exc
    if not raw.strip():
        raise AuthBootstrapError(
            f"trust-path failed: declared storage_state at {resolved} is empty"
        )
    try:
        json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AuthBootstrapError(
            f"trust-path failed: declared storage_state at {resolved} is not valid JSON: {exc}"
        ) from exc
    output_path.write_text(raw, encoding="utf-8")


def _validate_storage_state_artifact(
    *,
    spec: _SiteSpec,
    artifact_path: Path,
    site_url: str,
) -> None:
    """Validate that a produced artifact is usable for the declared live host."""
    payload, error = read_storage_state_payload(artifact_path)
    if error is not None:
        raise AuthBootstrapError(error)
    if site_url:
        host_error = storage_state_preflight_error_for_payload(
            artifact_path,
            payload,
            site_url,
        )
        if host_error is not None:
            raise AuthBootstrapError(host_error)
        return
    if storage_state_recorded_hosts(payload):
        logger.warning(
            "Phase 0d: generated storage_state for site %r without --instances; "
            "host binding could not be validated. Rigor runs should pass the "
            "same generated instances file used by Phase 2c/4.",
            spec.site_name,
        )


async def _bootstrap_via_form_login(
    *,
    spec: _SiteSpec,
    site_url: str,
    output_path: Path,
    timeout_ms: int = 30_000,
    playwright_factory: Any | None = None,
) -> None:
    """Built-in Playwright form-login bootstrap for Phase 0d.

    Launches a headless Chromium, navigates to ``login_url`` (resolved against
    ``site_url`` when relative), fills the discovered credentials, clicks
    submit, waits for ``success_url_substring`` to appear in the final URL,
    and dumps ``storage_state`` to ``output_path``.

    ``playwright_factory`` is injected by tests to mock Playwright without
    requiring the real package to be installed. In production we import
    ``playwright.async_api.async_playwright`` lazily so operators who never
    use form_login do not need Playwright at all.

    Raises :class:`AuthBootstrapError` with a precise diagnostic on any
    failure (missing selectors, timeout, bad credentials, etc.).
    """
    recipe = spec.form_login
    if not isinstance(recipe, dict):
        raise AuthBootstrapError("form_login recipe missing — cannot bootstrap")
    credentials = spec.credentials
    if not isinstance(credentials, dict):
        raise AuthBootstrapError(
            "authentication.credentials must be an object with string username+password"
        )
    username = credentials.get("username")
    password = credentials.get("password")
    if not isinstance(username, str) or not username:
        raise AuthBootstrapError("authentication.credentials.username must be a non-empty string")
    if not isinstance(password, str) or not password:
        raise AuthBootstrapError("authentication.credentials.password must be a non-empty string")

    login_url = recipe["login_url"]
    # Allow relative login_url when site_url is available (e.g. "/login" ->
    # "<site_url>/login"). Absolute URLs pass through unchanged.
    if "://" not in login_url:
        if not site_url:
            raise AuthBootstrapError(
                f"form_login.login_url {login_url!r} is relative but no site_url was "
                "supplied; pass --instances or declare an absolute login_url"
            )
        login_url = site_url.rstrip("/") + "/" + login_url.lstrip("/")

    success_substring = recipe["success_url_substring"]

    # Remove any stale artifact so a partial write cannot satisfy the
    # post-generation non-empty check (mirrors _run_generator).
    if output_path.exists():
        try:
            output_path.unlink()
        except OSError:
            pass
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if playwright_factory is None:
        try:
            from playwright.async_api import async_playwright as _pw_factory
        except ImportError as exc:
            raise AuthBootstrapError(
                "form_login bootstrap requires Playwright: install 'playwright' and run "
                "'playwright install chromium', or switch the site to generator_script / "
                "pre_auth_script. Underlying import error: " + repr(exc)
            ) from exc
        playwright_factory = _pw_factory

    try:
        async with playwright_factory() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                context = await browser.new_context()
                try:
                    page = await context.new_page()
                    await page.goto(login_url, timeout=timeout_ms)
                    await page.fill(recipe["username_selector"], username, timeout=timeout_ms)
                    await page.fill(recipe["password_selector"], password, timeout=timeout_ms)
                    await page.click(recipe["submit_selector"], timeout=timeout_ms)
                    # Wait for success_url_substring to appear in page.url.
                    await page.wait_for_url(
                        lambda url: success_substring in url,
                        timeout=timeout_ms,
                    )
                    await context.storage_state(path=str(output_path))
                finally:
                    await context.close()
            finally:
                await browser.close()
    except AuthBootstrapError:
        raise
    except Exception as exc:
        raise AuthBootstrapError(
            f"form_login bootstrap failed for site {spec.site_name!r}: {exc!r}"
        ) from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise AuthBootstrapError(
            f"form_login bootstrap for site {spec.site_name!r} did not write a non-empty "
            f"storage_state at {output_path}"
        )
    try:
        json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthBootstrapError(
            f"form_login bootstrap for site {spec.site_name!r} wrote invalid JSON to "
            f"{output_path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Consumer helpers (used by BrowserUseAgent._resolve_auth at Phase 3 launch).
# ---------------------------------------------------------------------------


def phase_0d_artifact_path(site_name: str, state_dir: Path | None = None) -> Path:
    """Return the canonical artifact path for a site's Phase 0d output."""
    site, site_error = safe_phase_0d_site_name(site_name)
    if site_error is not None or site is None:
        raise ValueError(site_error or "site_name must be non-empty")
    base = Path(state_dir) if state_dir is not None else get_state_dir()
    return base / "phase_0d" / site / "storage_state.json"


def _phase_0d_instance_key(instance: dict[str, Any]) -> str:
    raw = "|".join(
        str(instance.get(key) or "") for key in ("site_url", "replica_name", "replica_index")
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"instance_{digest}"


def phase_0d_instance_artifact_path(
    site_name: str,
    instance: dict[str, Any],
    state_dir: Path | None = None,
) -> Path:
    """Return the runtime-refresh artifact path for one concrete instance."""
    site_root = phase_0d_artifact_path(site_name, state_dir=state_dir).parent
    return site_root / "instances" / _phase_0d_instance_key(instance) / "storage_state.json"


def phase_0d_completion_path(site_name: str, state_dir: Path | None = None) -> Path:
    """Return the canonical completion marker path for a site's Phase 0d output."""
    site, site_error = safe_phase_0d_site_name(site_name)
    if site_error is not None or site is None:
        raise ValueError(site_error or "site_name must be non-empty")
    base = Path(state_dir) if state_dir is not None else get_state_dir()
    return base / "phase_0d" / site / "completion.json"


def _phase_0d_state_metadata(
    *,
    benchmark: Path,
    instances: str | Path | None,
) -> dict[str, str]:
    payload = {"benchmark_path": str(benchmark)}
    if instances is not None:
        payload["instances_path"] = str(Path(instances))
    return payload


# ---------------------------------------------------------------------------
# Liveness-validated cache (used by run() and _load_or_mint_storage_state).
# ---------------------------------------------------------------------------


def _force_remint() -> bool:
    return os.environ.get(FORCE_REMINT_ENV, "").strip().lower() in {"1", "true", "yes"}


def _load_session_from_storage_state(artifact_path: Path) -> requests.Session:
    """Build a ``requests.Session`` whose cookie jar reflects ``storage_state.json``."""
    session = requests.Session()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    cookies = payload.get("cookies") if isinstance(payload, dict) else None
    if isinstance(cookies, list):
        for cookie in cookies:
            if not isinstance(cookie, dict):
                continue
            name = cookie.get("name")
            value = cookie.get("value")
            if not isinstance(name, str) or not isinstance(value, str):
                continue
            domain = cookie.get("domain") if isinstance(cookie.get("domain"), str) else None
            path = cookie.get("path") if isinstance(cookie.get("path"), str) else "/"
            session.cookies.set(name, value, domain=domain, path=path)
    return session


def _editor_for_probe(site_name: str, instance: dict[str, Any], session: requests.Session) -> Any:
    from worldsim.editors import EDITOR_REGISTRY

    editor_cls = EDITOR_REGISTRY.get(("webarena_verified", site_name.strip().lower()))
    if editor_cls is None:
        return None
    return editor_cls(instance, session)


def _liveness_check_passes(
    *,
    site_name: str,
    artifact_path: Path,
    instance: dict[str, Any],
    now_fn: Any | None = None,
) -> bool:
    """Decide whether a cached storage_state may be reused without re-minting.

    Returns True when the validator version matches AND either the sidecar's
    ``last_validated_at`` is within ``LIVENESS_SOFT_TTL_SECONDS`` or the
    editor's live ``probe_authenticated`` confirms the session is alive.
    On a successful probe, stamps a fresh ``last_validated_at`` so the next
    call inside the TTL window can short-circuit.
    """
    from worldsim.storage_state_preflight import (
        read_storage_state_meta,
        update_storage_state_meta_validation,
    )

    meta = read_storage_state_meta(artifact_path)
    if meta is None:
        return True
    if int(meta.get("validator_version") or 0) != CURRENT_VALIDATOR_VERSION:
        return False
    now = now_fn() if now_fn is not None else datetime.now(UTC)
    last_validated_raw = meta.get("last_validated_at")
    if isinstance(last_validated_raw, str):
        try:
            last_validated = datetime.fromisoformat(last_validated_raw)
        except ValueError:
            last_validated = None
        else:
            if last_validated.tzinfo is None:
                last_validated = last_validated.replace(tzinfo=UTC)
            if (now - last_validated).total_seconds() < LIVENESS_SOFT_TTL_SECONDS:
                return True

    session = _load_session_from_storage_state(artifact_path)
    editor = _editor_for_probe(site_name, instance, session)
    if editor is None:
        return True
    try:
        alive = editor.probe_authenticated()
    except NotImplementedError:
        return True
    if alive:
        update_storage_state_meta_validation(
            artifact_path,
            last_validated_at=now,
            validator_version=CURRENT_VALIDATOR_VERSION,
        )
    return alive


async def _load_or_mint_storage_state(
    instance: dict[str, Any],
    *,
    force_remint: bool = False,
    benchmark_root: Path | None = None,
) -> Path:
    """Return a usable storage_state path, re-minting when the cache is dead.

    Composition rule: reuse the cached artifact iff the sidecar's
    ``validator_version`` matches and either (a) ``last_validated_at`` is
    within ``LIVENESS_SOFT_TTL_SECONDS`` or (b) ``probe_authenticated``
    confirms the session is alive. Otherwise, re-mint via
    :func:`reacquire_storage_state`. ``WORLDSIM_STORAGE_STATE_FORCE_REMINT``
    forces the re-mint branch regardless of cache state.
    """
    site_name = str(instance.get("site_name") or "").strip().lower()
    if not site_name:
        raise AuthBootstrapError("instance is missing site_name")
    artifact_path = phase_0d_artifact_path(site_name)
    if (
        not force_remint
        and not _force_remint()
        and artifact_path.exists()
        and _liveness_check_passes(
            site_name=site_name,
            artifact_path=artifact_path,
            instance=instance,
        )
    ):
        return artifact_path
    refreshed = await reacquire_storage_state(
        site_name=site_name,
        instance=instance,
        benchmark_root=benchmark_root,
    )
    from worldsim.storage_state_preflight import write_storage_state_meta

    write_storage_state_meta(refreshed, mechanism="load_or_mint_remint")
    return refreshed


__all__ = [
    "CURRENT_VALIDATOR_VERSION",
    "FORCE_REMINT_ENV",
    "LIVENESS_SOFT_TTL_SECONDS",
    "AuthBootstrapError",
    "phase_0d_artifact_path",
    "phase_0d_completion_path",
    "reacquire_storage_state",
    "run",
]
