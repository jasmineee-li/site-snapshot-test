"""Constrained fix-patch validator for Phase 3's diagnosis-fix loop.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` (Phase 3 Benign
Validation, fix-loop section) and ``/tmp/plan_constrained_fix_patches.md``.

Phase 3's diagnosis sandbox can propose patches to a task's ``reward_function``
or ``data_seed``. Without a guard, the pipeline has merged hallucinated patches
(e.g. ``POST /api/v4/session`` on GitLab 9.0) straight into ``apply_data_seed``.
This module gates every patch before it touches ``_apply_fix``: host-side, at
patch time, using only the site profile and the live ``BenchmarkInstance``.

Benchmark-agnostic: the allowlist is derived at runtime from the site profile's
``verification_capabilities`` (and optionally a top-level ``seeding_endpoints``
field, when Phase 0c emits one). No hard-coded URLs.

Contract:

    validate_fix_patch(patch, site_profile, site_url) -> list[str]

Returns a list of error strings; empty list means safe to merge. SQL validation
is delegated to ``worldsim._sandbox_validator.validate_seed_sql`` (single source
of truth for SQL guardrails).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

from worldsim._sandbox_validator import validate_seed_sql

logger = logging.getLogger(__name__)


# HTTP methods the seeding layer is allowed to emit. DELETE (destructive),
# HEAD (probing / info-leak), and OPTIONS (CORS probe) are rejected so
# diagnosis sandboxes cannot turn fix-patches into reconnaissance tools.
ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "PATCH"})

# Common path placeholders we normalize to a single-segment wildcard when
# turning profile example URLs into an allowlist regex. Mirrors the set
# Phase 0c already emits ("{id}", "{iid}", "{namespace}", ...).
_PLACEHOLDER_RE = re.compile(r"\{[^/{}]+\}")

# Profile URLs carry site-name placeholders like ``__GITLAB__`` that resolve
# to ``instance.site_url`` at runtime. We strip these before path extraction
# so origin matching is done separately against ``site_url``.
_SITE_PLACEHOLDER_RE = re.compile(r"__[A-Z][A-Z0-9_]*__")


# Reward function patches are currently validated shape-only (known top-level
# keys). Deep semantic checks are deferred per plan section 3.5.
_REWARD_FUNCTION_KEYS = frozenset({"eval", "adversarial_reward", "intent", "intent_template"})


def validate_fix_patch(
    patch: dict[str, Any],
    site_profile: dict[str, Any],
    site_url: str,
) -> list[str]:
    """Return a list of error strings for a suggested-fix patch. Empty = safe.

    Args:
        patch: The ``suggested_fix`` object from a diagnosis (``{"target": ...,
            "patch": {...}}``). ``target`` must be one of
            ``reward_function``, ``data_seed``, ``task_removal``, ``none``.
        site_profile: Parsed ``BENCHMARK_PROFILE_{site}.json``. Provides the
            verification-capability examples that anchor the path allowlist.
            Pass ``{}`` if the profile is unavailable; api patches fail closed.
        site_url: Live benchmark instance base URL (scheme+host[+port]). Every
            composed URL in the patch must match this origin.
    """
    errors: list[str] = []

    if not isinstance(patch, dict):
        return ["suggested_fix must be an object"]

    target = patch.get("target")
    body = patch.get("patch")

    if target in {"task_removal", "none"}:
        if body is not None:
            errors.append(f"suggested_fix patch must be null for target {target!r}")
        return errors

    if target not in {"reward_function", "data_seed"}:
        return [f"unsupported suggested_fix target {target!r}"]

    if body is None:
        # No-op patch is legal; _apply_fix treats it as "no change".
        return errors

    if not isinstance(body, dict):
        return ["suggested_fix patch must be an object or null"]

    if target == "reward_function":
        errors.extend(_validate_reward_patch(body))
        return errors

    # target == "data_seed"
    errors.extend(_validate_data_seed_patch(body, site_profile, site_url))
    return errors


# ---------------------------------------------------------------------------
# Reward function patches
# ---------------------------------------------------------------------------


def _validate_reward_patch(patch: dict[str, Any]) -> list[str]:
    """Shape-only check on reward_function merge patches."""
    errors: list[str] = []
    unknown = sorted(k for k in patch.keys() if k not in _REWARD_FUNCTION_KEYS)
    if unknown:
        errors.append(
            "reward_function patch contains unknown top-level keys: "
            f"{unknown}; expected a subset of {sorted(_REWARD_FUNCTION_KEYS)}"
        )
    return errors


# ---------------------------------------------------------------------------
# Data seed patches
# ---------------------------------------------------------------------------


def _validate_data_seed_patch(
    patch: dict[str, Any],
    site_profile: dict[str, Any],
    site_url: str,
) -> list[str]:
    """Validate a data_seed merge patch against the profile + live origin."""
    errors: list[str] = []

    mechanism = patch.get("mechanism")
    # A merge patch may omit `mechanism` when only tweaking existing fields;
    # in that case infer from present keys.
    if mechanism is None:
        if "statements" in patch:
            mechanism = "sql"
        elif "api_calls" in patch:
            mechanism = "api"
        elif "state" in patch:
            mechanism = "state_push"
        else:
            # Patch touches neither seed shape nor mechanism: nothing
            # dangerous to enforce here.
            return errors

    if mechanism == "sql":
        statements = patch.get("statements")
        if statements is None:
            errors.append("sql data seed patch must include a 'statements' list")
            return errors
        if not isinstance(statements, list) or not statements:
            errors.append("sql data seed patch 'statements' must be a non-empty list")
            return errors
        for idx, stmt in enumerate(statements):
            if not isinstance(stmt, str):
                errors.append(f"sql data seed statements[{idx}] must be a string")
                continue
            err = validate_seed_sql(stmt)
            if err is not None:
                errors.append(f"sql data seed statements[{idx}]: {err}")
        return errors

    if mechanism == "api":
        api_calls = patch.get("api_calls")
        if api_calls is None:
            errors.append("api data seed patch must include an 'api_calls' list")
            return errors
        if not isinstance(api_calls, list) or not api_calls:
            errors.append("api data seed patch 'api_calls' must be a non-empty list")
            return errors
        allowlist = _build_path_allowlist(site_profile)
        if not allowlist:
            errors.append(
                "api data seed rejected: site profile has no observed API "
                "endpoints to anchor an allowlist (fail-closed per plan)"
            )
            return errors
        for idx, call in enumerate(api_calls):
            errors.extend(_validate_api_call(call, idx, allowlist, site_url))
        return errors

    if mechanism == "state_push":
        state = patch.get("state")
        if state is None:
            errors.append("state_push data seed patch must include a 'state' value")
            return errors
        if not isinstance(state, dict):
            errors.append("state_push data seed 'state' must be a JSON object")
            return errors
        try:
            json.dumps(state)
        except (TypeError, ValueError) as exc:
            errors.append(f"state_push data seed 'state' must be JSON-serializable: {exc}")
        return errors

    errors.append(f"unknown data seed mechanism: {mechanism!r}")
    return errors


# ---------------------------------------------------------------------------
# API-call validation helpers
# ---------------------------------------------------------------------------


def _validate_api_call(
    call: Any,
    idx: int,
    allowlist: list[re.Pattern[str]],
    site_url: str,
) -> list[str]:
    """Validate one entry of an `api_calls` list."""
    errors: list[str] = []
    prefix = f"api_calls[{idx}]"

    if not isinstance(call, dict):
        return [f"{prefix} must be an object"]

    method = call.get("method")
    if not isinstance(method, str) or not method.strip():
        errors.append(f"{prefix} must include a method string")
    else:
        up = method.strip().upper()
        if up not in ALLOWED_METHODS:
            errors.append(
                f"{prefix} method {method!r} not allowed (allowed: {sorted(ALLOWED_METHODS)})"
            )

    path = call.get("path")
    if not isinstance(path, str) or not path:
        errors.append(f"{prefix} must include a 'path' starting with '/'")
    elif urlparse(path).scheme:
        # Full URL: origin + path checked together in _validate_api_path.
        errors.extend(_validate_api_path(path, prefix, allowlist, site_url))
    elif not path.startswith("/"):
        errors.append(f"{prefix} must include a 'path' starting with '/'")
    else:
        errors.extend(_validate_api_path(path, prefix, allowlist, site_url))

    body = call.get("body")
    if body is not None:
        try:
            json.dumps(body)
        except (TypeError, ValueError) as exc:
            errors.append(f"{prefix} body must be JSON-serializable: {exc}")

    return errors


def _validate_api_path(
    path: str,
    prefix: str,
    allowlist: list[re.Pattern[str]],
    site_url: str,
) -> list[str]:
    """Check origin match and path allowlist for one API call."""
    errors: list[str] = []

    # If `path` is actually a full URL (unusual but legal in merge patches),
    # enforce origin match explicitly. Otherwise the path is relative to
    # site_url, so origin is trivially satisfied.
    parsed_path = urlparse(path)
    if parsed_path.scheme:
        parsed_site = urlparse(site_url)
        if (
            parsed_path.scheme != parsed_site.scheme
            or (parsed_path.hostname or "") != (parsed_site.hostname or "")
            or _port(parsed_path) != _port(parsed_site)
        ):
            errors.append(
                f"{prefix} origin {parsed_path.scheme}://{parsed_path.netloc} "
                f"does not match live instance {site_url}"
            )
            return errors
        check_path = parsed_path.path or "/"
    else:
        check_path = path.split("?", 1)[0].split("#", 1)[0]

    if not any(p.match(check_path) for p in allowlist):
        errors.append(
            f"{prefix} path {check_path!r} is not on the site's API allowlist "
            "(not seen in BENCHMARK_PROFILE verification_capabilities or "
            "seeding_endpoints)"
        )

    return errors


def _port(parsed: Any) -> int | None:
    """Return the explicit or default port for a parsed URL."""
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "http":
        return 80
    if parsed.scheme == "https":
        return 443
    return None


# ---------------------------------------------------------------------------
# Allowlist construction
# ---------------------------------------------------------------------------


def _build_path_allowlist(site_profile: dict[str, Any]) -> list[re.Pattern[str]]:
    """Return compiled path patterns harvested from a site profile.

    Sources (in order):
    1. Top-level ``seeding_endpoints: [{method, url, purpose}]`` if Phase 0c
       emits one (plan section 1 option).
    2. ``verification_capabilities[*].examples[*].eval_config.expected.url``
       — URLs the recon sandbox observed a real request against.

    URL tokens like ``__GITLAB__`` and path placeholders like ``{id}`` are
    normalized so runtime paths match. Returns an empty list if nothing
    is harvested: callers fail closed on api patches in that case.
    """
    raw_urls: list[str] = []

    seeding_endpoints = site_profile.get("seeding_endpoints")
    if isinstance(seeding_endpoints, list):
        for entry in seeding_endpoints:
            if isinstance(entry, dict):
                url = entry.get("url") or entry.get("path")
                if isinstance(url, str):
                    raw_urls.append(url)

    for cap in site_profile.get("verification_capabilities", []) or []:
        if not isinstance(cap, dict):
            continue
        for example in cap.get("examples", []) or []:
            if not isinstance(example, dict):
                continue
            cfg = example.get("eval_config")
            if not isinstance(cfg, dict):
                continue
            expected = cfg.get("expected")
            if not isinstance(expected, dict):
                continue
            url = expected.get("url")
            if isinstance(url, str):
                raw_urls.append(url)
            elif isinstance(url, list):
                raw_urls.extend(u for u in url if isinstance(u, str))

    patterns: list[re.Pattern[str]] = []
    seen: set[str] = set()
    for url in raw_urls:
        path = _extract_path(url)
        if not path or path in seen:
            continue
        seen.add(path)
        patterns.append(_path_to_regex(path))
    return patterns


def _extract_path(url: str) -> str | None:
    """Strip scheme/host/site-placeholder and return just the path component."""
    stripped = _SITE_PLACEHOLDER_RE.sub("", url).strip()
    if not stripped:
        return None
    # After stripping __GITLAB__, shape looks like "/api/v4/projects/79/fork"
    # or still a full URL if the profile had a literal scheme.
    parsed = urlparse(stripped)
    if parsed.scheme:
        path = parsed.path
    else:
        path = stripped.split("?", 1)[0].split("#", 1)[0]
    if not path.startswith("/"):
        # Anchored URLs without a leading slash are suspicious; skip.
        return None
    return path


def _path_to_regex(path: str) -> re.Pattern[str]:
    """Convert a profile path (with `{id}`/`{iid}` placeholders) to a regex.

    The regex matches exact paths that share the same prefix segment-structure.
    Placeholders collapse to `[^/]+` so e.g. `/api/v4/projects/{id}/fork`
    matches `/api/v4/projects/79/fork` but not `/api/v4/session`.
    Also allows numeric/identifier substitution where the profile has a
    literal id (e.g. `/api/v4/projects/79/fork` also matches `/api/v4/projects/42/fork`).
    """
    # Split into segments, escape literal text, replace placeholders and
    # pure-numeric segments with a wildcard.
    parts: list[str] = []
    for segment in path.strip("/").split("/"):
        if not segment:
            continue
        if _PLACEHOLDER_RE.fullmatch(segment) or segment.isdigit():
            parts.append(r"[^/]+")
        elif _PLACEHOLDER_RE.search(segment):
            # Segment contains a placeholder plus literal text, e.g. "{id}.json".
            escaped = _PLACEHOLDER_RE.sub(r"[^/]+", re.escape(segment))
            # re.escape escapes the `{` and `}` — walk back through for the
            # placeholder substitution.
            escaped = re.sub(r"\\\{[^/{}]+\\\}", r"[^/]+", re.escape(segment))
            parts.append(escaped)
        else:
            parts.append(re.escape(segment))
    body = "/" + "/".join(parts) if parts else "/"
    # Allow an optional trailing slash.
    return re.compile(rf"^{body}/?$")
