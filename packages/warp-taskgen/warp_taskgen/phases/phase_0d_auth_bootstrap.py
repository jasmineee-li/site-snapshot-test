"""Phase 0d: Storage-state auth artifact bootstrap.

Canonical source: ``docs/warp-taskgen-technical-spec.md`` "Phase 0d"
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
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from warp_taskgen.agent_auth import safe_phase_0d_site_name
from warp_taskgen.atomic_io import write_json_atomic
from warp_taskgen.config import has_configured_agent_auth, load_benchmark_config
from warp_taskgen.phases.phase_0d_form_login import _bootstrap_via_form_login
from warp_taskgen.phases.phase_0d_generator_dispatch import (
    _choose_dispatch,
    _compute_input_hash,
    _is_idempotent_skip,
    _load_module,  # noqa: F401 - bound here so tests can patch it on the runner
    _run_generator,
    _trust_declared_path,
    _validate_storage_state_artifact,
    _verify_generate_signature,  # noqa: F401 - bound here for the same reason
)
from warp_taskgen.phases.phase_0d_site_auth_specs import (
    AuthBootstrapError,
    _collect_storage_state_specs,
    _extract_form_login_recipe,
    _load_site_urls,
    _SiteSpec,
    _spec_from_context,  # noqa: F401 - bound here for the same reason
)
from warp_taskgen.state import get_state_dir, save_state

logger = logging.getLogger(__name__)


# Bumped whenever ``probe_authenticated`` semantics or the meta-sidecar
# liveness fields change. Older sidecars with a smaller value force re-mint
# even when the input hash and TTL would otherwise allow reuse.
CURRENT_VALIDATOR_VERSION = 2

# Skip the live ``probe_authenticated`` call when the sidecar's
# ``last_validated_at`` is fresher than this. 30 minutes is conservative
# enough that a server-side session GC inside the window costs at most one
# re-mint per task batch, while avoiding a probe HTTP call on every load.
LIVENESS_SOFT_TTL_SECONDS = 30 * 60

# Operator break-glass: when set to ``true``/``1``, every Phase 0d
# load-or-mint call re-mints regardless of cache state.
FORCE_REMINT_ENV = "WORLDSIM_STORAGE_STATE_FORCE_REMINT"


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    failpoint_base: str | None = None,
) -> None:
    write_json_atomic(path, payload, failpoint_base=failpoint_base)


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
    site_instances: dict[str, list[dict[str, Any]]] = {}
    instances_path = getattr(args, "instances", None)
    if instances_path is not None:
        try:
            config = load_benchmark_config(instances_path)
            for inst in config.instances:
                if has_configured_agent_auth(inst.agent_auth):
                    instance_agent_auths.setdefault(inst.site_name, inst.agent_auth)
                site_instances.setdefault(inst.site_name, []).append(inst.model_dump())
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
        from warp_taskgen.storage_state_preflight import write_storage_state_meta

        write_storage_state_meta(artifact_path, mechanism=dispatch)
        generated.append(spec.site_name)
        logger.info(
            "Phase 0d: generated %s for site %r via %s",
            artifact_path,
            spec.site_name,
            dispatch,
        )

        # Per-instance mint: each replica of a site has its own SECRET_KEY_BASE
        # and per-replica DB, so a cookie minted against one replica is rejected
        # by all others. When more than one instance is configured for this
        # site, mint a per-replica artifact under
        # ``logs/phase_0d/<site>/instances/<instance_id>/storage_state.json``
        # using each replica's own ``site_url`` for liveness probing.
        # Single-instance configs skip the loop and continue using the shared
        # top-level artifact (backward compat).
        per_site_instances = site_instances.get(spec.site_name, [])
        if len(per_site_instances) > 1:
            per_instance_failures = await _mint_per_instance_artifacts(
                spec=spec,
                instances=per_site_instances,
                state_dir=state_dir,
                benchmark_root=benchmark,
            )
            for instance_failure in per_instance_failures:
                failures.append((spec.site_name, instance_failure))

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


async def _mint_per_instance_artifacts(
    *,
    spec: _SiteSpec,
    instances: list[dict[str, Any]],
    state_dir: Path,
    benchmark_root: Path,
) -> list[str]:
    """Mint a per-instance storage_state for every replica of ``spec.site_name``.

    Each per-replica artifact lives at
    ``<state_dir>/phase_0d/<site>/instances/<instance_id>/storage_state.json``.
    The minting goes through :func:`reacquire_storage_state`, which probes the
    replica's own ``site_url`` and reuses the cached artifact when its sidecar
    is fresh. ``WORLDSIM_STORAGE_STATE_FORCE_REMINT`` forces re-mint of every
    instance regardless of cache state.

    Returns a list of human-readable failure strings (one per failed instance).
    Per-instance failures do not abort the loop; the caller decides whether
    they roll up to a Phase 0d failure.
    """
    force = _force_remint()
    failures: list[str] = []
    for instance in instances:
        instance_id = phase_0d_instance_id(instance)
        per_instance_path = phase_0d_instance_artifact_path(
            spec.site_name, instance, state_dir=state_dir
        )
        if (
            not force
            and per_instance_path.exists()
            and _liveness_check_passes(
                site_name=spec.site_name,
                artifact_path=per_instance_path,
                instance=instance,
            )
        ):
            logger.info(
                "Phase 0d: skipping per-instance mint for site %r instance %s "
                "(artifact present, liveness ok)",
                spec.site_name,
                instance_id,
            )
            continue
        try:
            refreshed_path = await reacquire_storage_state(
                site_name=spec.site_name,
                instance=instance,
                benchmark_root=benchmark_root,
            )
        except AuthBootstrapError as exc:
            failures.append(f"per-instance {instance_id}: {exc}")
            logger.error(
                "Phase 0d: per-instance mint failed for site %r instance %s: %s",
                spec.site_name,
                instance_id,
                exc,
            )
            continue
        except Exception as exc:
            failures.append(f"per-instance {instance_id}: {exc!r}")
            logger.exception(
                "Phase 0d: per-instance mint unexpected failure for site %r instance %s",
                spec.site_name,
                instance_id,
            )
            continue
        from warp_taskgen.storage_state_preflight import write_storage_state_meta

        write_storage_state_meta(refreshed_path, mechanism="per_instance_mint")
        logger.info(
            "Phase 0d: minted per-instance storage_state for site %r instance %s at %s",
            spec.site_name,
            instance_id,
            refreshed_path,
        )
    return failures


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
# Consumer helpers (used by BrowserUseAgent._resolve_auth at Phase 3 launch).
# ---------------------------------------------------------------------------


def phase_0d_artifact_path(site_name: str, state_dir: Path | None = None) -> Path:
    """Return the canonical artifact path for a site's Phase 0d output."""
    site, site_error = safe_phase_0d_site_name(site_name)
    if site_error is not None or site is None:
        raise ValueError(site_error or "site_name must be non-empty")
    base = Path(state_dir) if state_dir is not None else get_state_dir()
    return base / "phase_0d" / site / "storage_state.json"


def phase_0d_instance_id(instance: dict[str, Any]) -> str:
    """Return a stable per-instance directory key.

    Folds ``site_url``, ``replica_name``, and ``replica_index`` into a
    16-hex-char digest so two replicas of the same site that share none of
    those fields are still distinguishable, and so the same instance produces
    the same key whether it is consumed as a Pydantic dump or a raw dict.
    """
    raw = "|".join(
        str(instance.get(key) or "") for key in ("site_url", "replica_name", "replica_index")
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"instance_{digest}"


# Alias kept for legacy callers.
_phase_0d_instance_key = phase_0d_instance_id


def phase_0d_instance_artifact_path(
    site_name: str,
    instance: dict[str, Any],
    state_dir: Path | None = None,
) -> Path:
    """Return the runtime-refresh artifact path for one concrete instance."""
    site_root = phase_0d_artifact_path(site_name, state_dir=state_dir).parent
    return site_root / "instances" / phase_0d_instance_id(instance) / "storage_state.json"


def phase_0d_instance_artifact_path_by_id(
    site_name: str,
    instance_id: str,
    state_dir: Path | None = None,
) -> Path:
    """Return the per-instance artifact path keyed by an already-computed instance id."""
    site_root = phase_0d_artifact_path(site_name, state_dir=state_dir).parent
    return site_root / "instances" / instance_id / "storage_state.json"


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
    from warp_taskgen.seeding.site_contracts import default_seed_registry

    registration = default_seed_registry().get("webarena_verified", site_name.strip().lower())
    editor_cls = registration.editor_factory if registration is not None else None
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
    from warp_taskgen.storage_state_preflight import (
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
    browser_alive = _storage_state_browser_liveness_check(site_name, instance, session)
    if browser_alive is None:
        editor = _editor_for_probe(site_name, instance, session)
        if editor is None:
            return True
        try:
            alive = editor.probe_authenticated()
        except NotImplementedError:
            return True
    else:
        alive = browser_alive
    if alive:
        update_storage_state_meta_validation(
            artifact_path,
            last_validated_at=now,
            validator_version=CURRENT_VALIDATOR_VERSION,
        )
    return alive


def _storage_state_browser_liveness_check(
    site_name: str,
    instance: dict[str, Any],
    session: requests.Session,
) -> bool | None:
    """Return browser-cookie liveness for storage_state-backed sites.

    Editor probes often use API tokens or request headers. That proves the
    editor can mutate data, not that a Browser Use page loaded with
    ``storage_state`` is authenticated. GitLab needs a UI probe because its
    PAT and Rails session cookie fail independently.
    """

    site = str(site_name or "").strip().lower()
    site_url = str(instance.get("site_url") or "").rstrip("/")
    if site != "gitlab" or not site_url:
        return None
    response = session.get(
        f"{site_url}/-/profile",
        timeout=10,
        allow_redirects=False,
    )
    if response.status_code == 200:
        return True
    if response.status_code in {401, 403}:
        return False
    if 300 <= response.status_code < 400:
        location = response.headers.get("Location") or ""
        if "/users/sign_in" in location or "/login" in location:
            return False
    response.raise_for_status()
    return False


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
    from warp_taskgen.storage_state_preflight import write_storage_state_meta

    write_storage_state_meta(refreshed, mechanism="load_or_mint_remint")
    return refreshed


__all__ = [
    "CURRENT_VALIDATOR_VERSION",
    "FORCE_REMINT_ENV",
    "LIVENESS_SOFT_TTL_SECONDS",
    "AuthBootstrapError",
    "phase_0d_artifact_path",
    "phase_0d_completion_path",
    "phase_0d_instance_artifact_path",
    "phase_0d_instance_artifact_path_by_id",
    "phase_0d_instance_id",
    "reacquire_storage_state",
    "run",
]
