"""Phase 0d generator dispatch: input hashing, path resolution, and dispatch.

Owns the content-addressed input hash and idempotency check, generator_script
resolution/loading/invocation, the dispatch precedence decision, the trust-path
copy of a benchmark-declared ``storage_state``, and the produced-artifact
validation. See the ``warp_taskgen.phases.phase_0d_auth_bootstrap`` runner for
the phase contract.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any

from warp_taskgen.agent_auth import (
    read_storage_state_payload,
    storage_state_preflight_error_for_payload,
    storage_state_recorded_hosts,
)
from warp_taskgen.phases.phase_0d_site_auth_specs import AuthBootstrapError, _SiteSpec

logger = logging.getLogger(__name__)


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
    module_name = f"warp_taskgen._phase_0d_generator_{site_name}_{abs(hash(str(script_path)))}_{content_digest}"

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
# Dispatch selection
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
