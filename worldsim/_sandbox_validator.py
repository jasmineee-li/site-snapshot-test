#!/usr/bin/env python3
"""In-sandbox output validator for WorldSim v5.

This script runs INSIDE a Modal sandbox at /workspace/_validate.py.
It validates Claude Code's output before the sandbox exits, catching
schema errors within the same session at zero extra cost.

Requirements:
- Zero external dependencies (stdlib only)
- No worldsim imports (the package is not installed in the sandbox)

Usage:
    python /workspace/_validate.py <schema> [options]

Exit code 0 = valid, 1 = invalid.
Prints JSON to stdout: {"valid": true/false, "errors": [...]}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Data-seed validation (mirrors worldsim/seeding.py)
# ---------------------------------------------------------------------------

_MULTI_STATEMENT_PATTERN = re.compile(r";(?=(?:[^']|'[^']*')*$)")
_DISALLOWED_SQL_KEYWORDS = re.compile(
    r"\b("
    r"DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|DELETE|REPLACE|MERGE|CALL|DO|EXEC|EXECUTE|"
    r"BEGIN|COMMIT|ROLLBACK|SAVEPOINT|PREPARE|DEALLOCATE|COPY|LOAD|ATTACH|DETACH|VACUUM"
    r")\b",
    re.IGNORECASE,
)
# Matches single-quoted SQL string literals with SQL's `''` escape. We strip
# these before running the disallowed-keyword check so that keyword-like words
# ("DO NOT", "MERGE request") appearing inside string values are not flagged.
_SQL_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")


def validate_seed_sql(statement: str) -> str | None:
    """Return an error string if the SQL statement is invalid, else None."""
    normalized = statement.strip()
    if not normalized:
        return "SQL seed statement is empty"
    if _MULTI_STATEMENT_PATTERN.search(normalized.rstrip(";")):
        return "SQL seed must be a single statement"
    if _DISALLOWED_SQL_KEYWORDS.search(_SQL_STRING_LITERAL.sub("''", normalized)):
        return f"SQL seed contains a disallowed keyword: {normalized[:100]}..."

    first_token = normalized.split(None, 1)[0].upper()
    if first_token not in {"INSERT", "UPDATE"}:
        return f"SQL seed must start with INSERT or UPDATE, got {first_token!r}"
    if first_token == "UPDATE" and " WHERE " not in f" {normalized.upper()} ":
        return "UPDATE seed statements must include a WHERE clause"
    return None


def validate_data_seed(seed: object, *, allow_none: bool = False) -> list[str]:
    """Return a list of errors for a data seed payload."""
    errors: list[str] = []
    if not isinstance(seed, dict):
        errors.append("data seed must be an object")
        return errors

    mechanism = seed.get("mechanism")
    if mechanism in (None, "none"):
        if allow_none:
            return errors
        errors.append("data seed must declare a non-empty mechanism")
        return errors

    if mechanism == "sql":
        statements = seed.get("statements")
        if not isinstance(statements, list) or not statements:
            errors.append("sql data seed must include a non-empty statements list")
            return errors
        for stmt in statements:
            if not isinstance(stmt, str):
                errors.append("sql data seed statements must be strings")
                continue
            sql_err = validate_seed_sql(stmt)
            if sql_err is not None:
                errors.append(sql_err)
        return errors

    if mechanism == "api":
        api_calls = seed.get("api_calls")
        if not isinstance(api_calls, list) or not api_calls:
            errors.append("api data seed must include a non-empty api_calls list")
            return errors
        for call in api_calls:
            if not isinstance(call, dict):
                errors.append("api data seed calls must be objects")
                continue
            method = call.get("method")
            path = call.get("path")
            if not isinstance(method, str) or not method.strip():
                errors.append("api data seed calls must include a method")
            if not isinstance(path, str) or not path.startswith("/"):
                errors.append("api data seed calls must include a path starting with '/'")
        return errors

    if mechanism == "state_push":
        if "state" not in seed:
            errors.append("state_push data seed must include a state payload")
        return errors

    errors.append(f"unknown data seed mechanism: {mechanism!r}")
    return errors


# ---------------------------------------------------------------------------
# Per-schema validators
# ---------------------------------------------------------------------------


def validate_manifest(data: object) -> list[str]:
    """Validate BENCHMARK_MANIFEST.json structure."""
    errors: list[str] = []
    if not isinstance(data, dict):
        errors.append("manifest must be a JSON object")
        return errors

    if "sites" not in data:
        errors.append("manifest missing 'sites' field")
    elif not isinstance(data["sites"], list) or not data["sites"]:
        errors.append("manifest 'sites' must be a non-empty array")
    else:
        for i, site in enumerate(data["sites"]):
            if not isinstance(site, dict):
                errors.append(f"sites[{i}] must be an object")
                continue
            if "name" not in site:
                errors.append(f"sites[{i}] missing 'name'")

    if "evaluation" not in data:
        errors.append("manifest missing 'evaluation' field")
    elif not isinstance(data["evaluation"], dict):
        errors.append("manifest 'evaluation' must be an object")

    return errors


def validate_profile(data: object, *, site_name: str) -> list[str]:
    """Validate BENCHMARK_PROFILE.json structure and cross-references."""
    errors: list[str] = []
    if not isinstance(data, dict):
        errors.append("profile must be a JSON object")
        return errors

    # site_name mismatch
    profile_site = data.get("site_name")
    if profile_site and profile_site != site_name:
        errors.append(f"profile site_name mismatch: expected {site_name!r}, got {profile_site!r}")

    entity_fields = _entity_field_index(data.get("data_model"))

    # Cross-ref injection surfaces against data_model
    for surface in data.get("injection_surface", []):
        source = surface.get("source_field", "")
        if source and "." in source:
            entity_name, _, field_name = source.partition(".")
            if entity_name not in entity_fields and entity_fields:
                errors.append(
                    f"injection surface {surface.get('id', '?')!r} references "
                    f"unknown entity {entity_name!r} in {source!r}"
                )
            elif entity_fields and field_name not in entity_fields.get(entity_name, set()):
                errors.append(
                    f"injection surface {surface.get('id', '?')!r} references "
                    f"unknown field {source!r}"
                )

    return errors


def _entity_field_index(data_model: object) -> dict[str, set[str]]:
    """Return {entity_name: {field_name}} for a data model payload."""
    index: dict[str, set[str]] = {}
    if not isinstance(data_model, list):
        return index

    for entity in data_model:
        if not isinstance(entity, dict):
            continue
        entity_name = entity.get("entity")
        if not isinstance(entity_name, str) or not entity_name:
            continue
        fields = index.setdefault(entity_name, set())
        for field in entity.get("fields", []):
            if not isinstance(field, dict):
                continue
            field_name = field.get("name")
            if isinstance(field_name, str) and field_name:
                fields.add(field_name)
    return index


def validate_benign_tasks(data: object, *, site_name: str) -> list[str]:
    """Validate benign_tasks.json structure."""
    errors: list[str] = []
    if not isinstance(data, list):
        errors.append("benign tasks must be a JSON array")
        return errors

    if not data:
        errors.append("benign tasks array is empty")
        return errors

    _REQUIRED_FIELDS = ("id", "site", "instruction", "start_urls", "reward_function")
    _ALLOWED_EVALUATORS = {"NetworkEventEvaluator", "AgentResponseEvaluator"}
    id_pattern = re.compile(rf"^novel_{re.escape(site_name)}_\d+$")

    for i, task in enumerate(data):
        prefix = f"task {i}"
        if not isinstance(task, dict):
            errors.append(f"{prefix} is not an object")
            continue

        task_id = str(task.get("id", "?"))
        prefix = f"task {i} ({task_id})"

        missing = [f for f in _REQUIRED_FIELDS if f not in task]
        if missing:
            errors.append(f"{prefix} missing required fields: {', '.join(missing)}")
            continue

        if not id_pattern.match(task_id):
            errors.append(f"{prefix} id must match novel_{site_name}_<n>")

        if task.get("site") != site_name:
            errors.append(f"{prefix} site must be {site_name!r}")

        start_urls = task.get("start_urls")
        if not isinstance(start_urls, list) or not start_urls:
            errors.append(f"{prefix} start_urls must be a non-empty list")
        else:
            for url in start_urls:
                if isinstance(url, str) and "__" in url:
                    # Has placeholder tokens (e.g. __SITE__)
                    pass

        reward = task.get("reward_function")
        if not isinstance(reward, dict):
            errors.append(f"{prefix} reward_function must be an object")
            continue

        eval_configs = reward.get("eval")
        if not isinstance(eval_configs, list) or not eval_configs:
            errors.append(f"{prefix} reward_function.eval must be a non-empty list")
            continue

        for ei, config in enumerate(eval_configs):
            if not isinstance(config, dict):
                errors.append(f"{prefix} eval[{ei}] must be an object")
                continue
            evaluator = config.get("evaluator")
            if evaluator not in _ALLOWED_EVALUATORS:
                errors.append(f"{prefix} eval[{ei}] uses unsupported evaluator {evaluator!r}")

    return errors


def validate_adversarial_tasks(data: object) -> list[str]:
    """Validate adversarial_tasks.json structure and cross-reference benign tasks.

    Supports the **minimal output schema** where Claude only produces
    ``id``, ``benign_task_id``, ``adversarial_data_seed``, and
    ``adversarial_reward``.  The validator simulates the merge that the
    orchestrator will perform (copying immutable fields from the benign
    task) and validates the *merged* result.
    """
    errors: list[str] = []
    if not isinstance(data, list):
        errors.append("adversarial tasks must be a JSON array")
        return errors

    if not data:
        errors.append("adversarial tasks array is empty")
        return errors

    # Load benign tasks for cross-reference and merge simulation
    benign_path = Path("/workspace/tasks/benign_tasks.json")
    benign_by_id: dict[str, dict] = {}
    if benign_path.exists():
        try:
            benign_tasks = json.loads(benign_path.read_text())
            if isinstance(benign_tasks, list):
                benign_by_id = {
                    str(t.get("id", "")): t for t in benign_tasks if isinstance(t, dict)
                }
        except json.JSONDecodeError:
            errors.append("could not parse benign_tasks.json for cross-reference")

    # Minimal schema only requires these fields from Claude's output.
    _REQUIRED_FIELDS = (
        "id",
        "benign_task_id",
        "adversarial_data_seed",
    )

    for i, task in enumerate(data):
        prefix = f"task {i}"
        if not isinstance(task, dict):
            errors.append(f"{prefix} is not an object")
            continue

        task_id = str(task.get("id", "?"))
        prefix = f"task {i} ({task_id})"

        missing = [f for f in _REQUIRED_FIELDS if f not in task]
        if missing:
            errors.append(f"{prefix} missing required fields: {', '.join(missing)}")
            continue

        # Cross-reference benign task
        benign_task_id = str(task.get("benign_task_id", ""))
        if benign_by_id and benign_task_id not in benign_by_id:
            errors.append(f"{prefix} references unknown benign_task_id {benign_task_id!r}")
            # Cannot simulate merge without a benign task — skip remaining checks.
            continue

        # Simulate the orchestrator merge: copy immutable fields from the
        # benign task so we can validate the merged result.
        benign_task = benign_by_id.get(benign_task_id) if benign_by_id else None
        merged = dict(task)
        if benign_task is not None:
            for field in ("instruction", "site", "sites", "start_urls", "data_seed"):
                if field in benign_task:
                    merged[field] = benign_task[field]

        # Validate adversarial_reward — accept either top-level or nested.
        adv_reward = task.get("adversarial_reward")
        reward_fn = task.get("reward_function")
        if adv_reward is None and isinstance(reward_fn, dict):
            adv_reward = reward_fn.get("adversarial_reward")

        if adv_reward is None:
            errors.append(
                f"{prefix} missing adversarial_reward (neither top-level nor in reward_function)"
            )
        elif not isinstance(adv_reward, dict) or not adv_reward:
            errors.append(f"{prefix} adversarial_reward must be a non-empty object")
        elif "type" not in adv_reward:
            errors.append(f"{prefix} adversarial_reward missing type field")

        # Validate adversarial_data_seed
        seed_errors = validate_data_seed(task.get("adversarial_data_seed"), allow_none=False)
        for se in seed_errors:
            errors.append(f"{prefix} adversarial_data_seed: {se}")

    return errors


def validate_diagnosis(data: object) -> list[str]:
    """Validate diagnosis.json structure."""
    errors: list[str] = []
    if not isinstance(data, dict):
        errors.append("diagnosis must be a JSON object")
        return errors

    valid_root_causes = {
        "reward_bug",
        "seed_bug",
        "impossible",
        "too_hard",
        "agent_limitation",
    }
    root_cause = data.get("root_cause")
    if root_cause not in valid_root_causes:
        errors.append(f"root_cause must be one of {sorted(valid_root_causes)}, got {root_cause!r}")

    suggested_fix = data.get("suggested_fix")
    if suggested_fix is not None:
        if not isinstance(suggested_fix, dict):
            errors.append("suggested_fix must be an object or null")
        elif "target" not in suggested_fix:
            errors.append("suggested_fix missing 'target' field")

    return errors


def validate_ecological_validity(data: object) -> list[str]:
    """Validate ecological_validity.json structure."""
    errors: list[str] = []
    if not isinstance(data, dict):
        errors.append("ecological validity must be a JSON object")
        return errors

    probes = data.get("probes")
    if not isinstance(probes, list):
        errors.append("ecological validity missing 'probes' array")
    else:
        for i, probe in enumerate(probes):
            if not isinstance(probe, dict):
                errors.append(f"probes[{i}] must be an object")
                continue
            score = probe.get("score")
            if not isinstance(score, (int, float)):
                errors.append(f"probes[{i}] missing numeric 'score'")
            elif score < 0.0 or score > 1.0:
                errors.append(f"probes[{i}] score must be 0.0-1.0, got {score}")

    average = data.get("average_score")
    if not isinstance(average, (int, float)):
        errors.append("ecological validity missing numeric 'average_score'")
    elif isinstance(average, (int, float)) and (average < 0.0 or average > 1.0):
        errors.append(f"average_score must be 0.0-1.0, got {average}")

    return errors


def validate_judge_recommendation(data: object) -> list[str]:
    """Validate judge_recommendation.json structure."""
    errors: list[str] = []
    if not isinstance(data, dict):
        errors.append("judge recommendation must be a JSON object")
        return errors

    if "diagnosis" not in data:
        errors.append("judge recommendation missing 'diagnosis'")
    if "refusal_trigger" not in data:
        errors.append("judge recommendation missing 'refusal_trigger'")

    valid_strategies = {
        "authority_framing",
        "semantic_coherence",
        "specificity",
        "visual_concealment",
        "verbosity_adjustment",
    }

    strategies = data.get("recommended_strategies")
    if not isinstance(strategies, list):
        errors.append("judge recommendation missing 'recommended_strategies' array")
    else:
        for i, strategy in enumerate(strategies):
            if not isinstance(strategy, dict):
                errors.append(f"recommended_strategies[{i}] must be an object")
                continue
            s = strategy.get("strategy")
            if s not in valid_strategies:
                errors.append(
                    f"recommended_strategies[{i}] strategy must be one of "
                    f"{sorted(valid_strategies)}, got {s!r}"
                )

    return errors


def validate_revised_task(data: object) -> list[str]:
    """Validate revised_task.json structure."""
    errors: list[str] = []
    if not isinstance(data, dict):
        errors.append("revised task must be a JSON object")
        return errors

    seed = data.get("adversarial_data_seed")
    if seed is not None:
        seed_errors = validate_data_seed(seed, allow_none=False)
        for se in seed_errors:
            errors.append(f"adversarial_data_seed: {se}")

    return errors


def validate_variant_task(data: object) -> list[str]:
    """Validate variant_task.json structure."""
    errors: list[str] = []
    if not isinstance(data, dict):
        errors.append("variant task must be a JSON object")
        return errors

    seed = data.get("adversarial_data_seed")
    if seed is not None:
        seed_errors = validate_data_seed(seed, allow_none=False)
        for se in seed_errors:
            errors.append(f"adversarial_data_seed: {se}")

    return errors


def validate_verification_capabilities(data: object, *, site_name: str) -> list[str]:
    """Validate VERIFICATION_CAPABILITIES.json structure."""
    errors: list[str] = []
    if not isinstance(data, list):
        errors.append("verification capabilities must be a JSON array")
        return errors
    if not data:
        errors.append("verification capabilities array is empty")
        return errors
    for i, cap in enumerate(data):
        prefix = f"capabilities[{i}]"
        if not isinstance(cap, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if "eval_type" not in cap or not isinstance(cap.get("eval_type"), str):
            errors.append(f"{prefix} missing 'eval_type' string")
        if "description" not in cap or not isinstance(cap.get("description"), str):
            errors.append(f"{prefix} missing 'description' string")
    return errors


def validate_data_model_profile(data: object, *, site_name: str) -> list[str]:
    """Validate DATA_MODEL.json structure."""
    errors: list[str] = []
    if not isinstance(data, list):
        errors.append("data model must be a JSON array")
        return errors
    if not data:
        errors.append("data model array is empty")
        return errors
    for i, entity in enumerate(data):
        prefix = f"entity[{i}]"
        if not isinstance(entity, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if "entity" not in entity or not isinstance(entity.get("entity"), str):
            errors.append(f"{prefix} missing 'entity' string")
        fields = entity.get("fields")
        if not isinstance(fields, list) or not fields:
            errors.append(f"{prefix} must have a non-empty 'fields' array")
    return errors


def validate_injection_surface(
    data: object,
    *,
    site_name: str,
    data_model: object | None = None,
) -> list[str]:
    """Validate INJECTION_SURFACE.json structure."""
    errors: list[str] = []
    if not isinstance(data, dict):
        errors.append("injection surface must be a JSON object")
        return errors

    surfaces = data.get("injection_surface")
    if not isinstance(surfaces, list):
        errors.append("missing 'injection_surface' array")
        return errors

    surface_ids: set[str] = set()
    for i, surface in enumerate(surfaces):
        prefix = f"injection_surface[{i}]"
        if not isinstance(surface, dict):
            errors.append(f"{prefix} must be an object")
            continue
        sid = surface.get("id")
        if not isinstance(sid, str) or not sid:
            errors.append(f"{prefix} missing 'id' string")
        else:
            surface_ids.add(sid)
        if "source_field" not in surface:
            errors.append(f"{prefix} missing 'source_field'")
        elif isinstance(surface.get("source_field"), str) and data_model is not None:
            source_field = surface["source_field"]
            if "." not in source_field:
                errors.append(f"{prefix} source_field must use entity.field format")
            else:
                entity_name, _, field_name = source_field.partition(".")
                entity_fields = _entity_field_index(data_model)
                if entity_name not in entity_fields and entity_fields:
                    errors.append(
                        f"{prefix} references unknown entity {entity_name!r} in {source_field!r}"
                    )
                elif entity_fields and field_name not in entity_fields.get(entity_name, set()):
                    errors.append(f"{prefix} references unknown field {source_field!r}")
        realism = surface.get("attacker_realism")
        if realism not in ("low", "medium", "high"):
            errors.append(f"{prefix} attacker_realism must be low/medium/high, got {realism!r}")

    coverage = data.get("existing_task_coverage")
    if not isinstance(coverage, dict):
        errors.append("missing 'existing_task_coverage' object")
    else:
        for key in (
            "injection_surfaces_with_task_coverage",
            "injection_surfaces_without_task_coverage",
        ):
            items = coverage.get(key)
            if not isinstance(items, list):
                errors.append(f"existing_task_coverage missing '{key}' array")
            elif surface_ids:
                for item in items:
                    if item not in surface_ids:
                        errors.append(
                            f"existing_task_coverage.{key} references unknown surface {item!r}"
                        )

    return errors


# Machine-readable authentication mechanism types. See plan §1.
_AUTH_MECHANISM_TYPES = frozenset(
    {
        "storage_state",
        "form_login",
        "http_basic",
        "http_headers",
        "client_cert",
        "pre_auth_script",
        "none",
        "unknown",
    }
)

# Map type -> expected sub-object key (for the "exactly one populated" check).
_AUTH_MECHANISM_SUB_KEYS = {
    "storage_state": "storage_state",
    "form_login": "form_login",
    "http_basic": "http_basic",
    "http_headers": "http_headers",
    "client_cert": "client_cert",
    "pre_auth_script": "pre_auth_script",
}


_FORM_LOGIN_REQUIRED_SELECTORS = (
    "login_url",
    "username_selector",
    "password_selector",
    "submit_selector",
)


def _validate_form_login_recipe(sub: object, *, location: str) -> list[str]:
    """Validate a form_login recipe dict (shared for top-level + nested use).

    ``location`` is the dotted path used in error messages (e.g.
    ``"auth_mechanism.form_login"`` or
    ``"auth_mechanism.storage_state.form_login"``) so we can reuse the
    validator for both the top-level ``form_login`` type and the nested
    ``storage_state.form_login`` bootstrap recipe supported by Phase 0d.

    Accepts both ``success_url_substring`` (canonical, Phase 0d's lookup key)
    and the legacy ``success_substring`` name for backward compatibility with
    previously-discovered AGENT_CONTEXT files. At least one must be a
    non-empty string.
    """
    errors: list[str] = []
    if not isinstance(sub, dict):
        errors.append(f"{location} must be an object")
        return errors
    for key in _FORM_LOGIN_REQUIRED_SELECTORS:
        val = sub.get(key)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"{location}.{key} must be a non-empty string")
    # success_url_substring (canonical) or success_substring (legacy alias).
    primary = sub.get("success_url_substring")
    legacy = sub.get("success_substring")
    has_primary = isinstance(primary, str) and primary.strip()
    has_legacy = isinstance(legacy, str) and legacy.strip()
    if primary is not None and not isinstance(primary, str):
        errors.append(f"{location}.success_url_substring must be a string or null")
    if legacy is not None and not isinstance(legacy, str):
        errors.append(f"{location}.success_substring must be a string or null")
    if not has_primary and not has_legacy:
        errors.append(
            f"{location} must declare success_url_substring (or legacy success_substring) "
            "so Phase 0d can detect login success"
        )
    return errors


def _validate_auth_mechanism(auth_mech: object) -> list[str]:
    """Validate the ``auth_mechanism`` object. Returns a list of error strings.

    Contract (additive to the prose ``authentication`` block):
    - ``type`` present, string, in enum.
    - Each ``type`` requires its corresponding sub-object populated with the
      right keys; ``none`` / ``unknown`` require non-empty ``notes``.
    - Exactly one sub-object populated relative to the declared ``type``.
    - ``storage_state`` may additionally carry a nested ``form_login`` recipe
      (login_url + selectors + success_url_substring); Phase 0d uses it to
      auto-bootstrap the artifact without a hand-authored generator_script.
    - When a ``form_login`` recipe is declared (top-level or nested under
      ``storage_state``), ``authentication.credentials`` must be an object
      with string ``username`` and ``password`` — the bootstrap needs them to
      fill the form. This cross-block check runs in
      :func:`validate_agent_context`.
    - No disk-existence checks here; runtime (Phase 3 launch) enforces paths.
    """
    errors: list[str] = []
    if not isinstance(auth_mech, dict):
        errors.append("auth_mechanism must be a JSON object")
        return errors

    mech_type = auth_mech.get("type")
    if mech_type is None:
        errors.append("auth_mechanism missing 'type'")
        return errors
    if not isinstance(mech_type, str):
        errors.append("auth_mechanism.type must be a string")
        return errors
    if mech_type not in _AUTH_MECHANISM_TYPES:
        errors.append(
            f"auth_mechanism.type {mech_type!r} not in allowed set "
            f"({sorted(_AUTH_MECHANISM_TYPES)})"
        )
        return errors

    # Per-type required fields.
    if mech_type == "storage_state":
        sub = auth_mech.get("storage_state")
        if not isinstance(sub, dict):
            errors.append(
                "auth_mechanism.storage_state must be an object when type='storage_state'"
            )
        else:
            path = sub.get("path")
            if not isinstance(path, str) or not path.strip():
                errors.append("auth_mechanism.storage_state.path must be a non-empty string")
            gen = sub.get("generator_script")
            if gen is not None and not isinstance(gen, str):
                errors.append(
                    "auth_mechanism.storage_state.generator_script must be a string or null"
                )
            refresh = sub.get("per_task_refresh")
            if refresh is not None and not isinstance(refresh, bool):
                errors.append("auth_mechanism.storage_state.per_task_refresh must be a boolean")
            # Nested form_login recipe (optional). When declared, Phase 0d uses
            # the built-in form-login bootstrapper to produce the artifact
            # without needing a hand-authored generator_script. We validate the
            # recipe shape here; the "exactly one top-level sub-object" rule
            # below is unaffected because ``form_login`` is nested under
            # ``storage_state``, not a peer.
            nested_form = sub.get("form_login")
            if nested_form is not None:
                errors.extend(
                    _validate_form_login_recipe(
                        nested_form, location="auth_mechanism.storage_state.form_login"
                    )
                )

    elif mech_type == "form_login":
        sub = auth_mech.get("form_login")
        errors.extend(_validate_form_login_recipe(sub, location="auth_mechanism.form_login"))

    elif mech_type == "http_basic":
        sub = auth_mech.get("http_basic")
        if not isinstance(sub, dict):
            errors.append("auth_mechanism.http_basic must be an object when type='http_basic'")
        else:
            for key in ("username", "password"):
                val = sub.get(key)
                if not isinstance(val, str) or not val:
                    errors.append(f"auth_mechanism.http_basic.{key} must be a non-empty string")

    elif mech_type == "http_headers":
        sub = auth_mech.get("http_headers")
        if not isinstance(sub, dict):
            errors.append("auth_mechanism.http_headers must be an object when type='http_headers'")
        else:
            headers = sub.get("headers")
            if not isinstance(headers, dict) or not headers:
                errors.append("auth_mechanism.http_headers.headers must be a non-empty object")
            elif not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
                errors.append("auth_mechanism.http_headers.headers must be a string->string map")
            scope = sub.get("scope_url_pattern")
            if scope is not None and not isinstance(scope, str):
                errors.append(
                    "auth_mechanism.http_headers.scope_url_pattern must be a string or null"
                )

    elif mech_type == "client_cert":
        sub = auth_mech.get("client_cert")
        if not isinstance(sub, dict):
            errors.append("auth_mechanism.client_cert must be an object when type='client_cert'")
        else:
            for key in ("cert_path", "key_path", "origin"):
                val = sub.get(key)
                if not isinstance(val, str) or not val.strip():
                    errors.append(f"auth_mechanism.client_cert.{key} must be a non-empty string")

    elif mech_type == "pre_auth_script":
        sub = auth_mech.get("pre_auth_script")
        if not isinstance(sub, dict):
            errors.append(
                "auth_mechanism.pre_auth_script must be an object when type='pre_auth_script'"
            )
        else:
            script_path = sub.get("script_path")
            if not isinstance(script_path, str) or not script_path.strip():
                errors.append(
                    "auth_mechanism.pre_auth_script.script_path must be a non-empty string"
                )
            args = sub.get("args")
            if args is not None and not isinstance(args, list):
                errors.append("auth_mechanism.pre_auth_script.args must be a list or null")

    elif mech_type in ("none", "unknown"):
        notes = auth_mech.get("notes")
        if not isinstance(notes, str) or not notes.strip():
            errors.append(
                f"auth_mechanism.notes must be a non-empty string when type='{mech_type}'"
            )

    # Exactly one sub-object populated: the one matching ``type``.
    expected_key = _AUTH_MECHANISM_SUB_KEYS.get(mech_type)
    populated_sub_keys = [
        key for key in _AUTH_MECHANISM_SUB_KEYS.values() if auth_mech.get(key) is not None
    ]
    if expected_key is not None:
        extras = [k for k in populated_sub_keys if k != expected_key]
        if extras:
            errors.append(
                f"auth_mechanism has extra sub-objects {sorted(extras)} for type={mech_type!r}; "
                "exactly one sub-object must be populated"
            )
    else:
        # none / unknown: no sub-object should be populated.
        if populated_sub_keys:
            errors.append(
                f"auth_mechanism type={mech_type!r} must not populate any sub-object; "
                f"found {sorted(populated_sub_keys)}"
            )

    return errors


def validate_agent_context(data: object, *, site_name: str) -> list[str]:
    """Validate AGENT_CONTEXT.json structure."""
    errors: list[str] = []
    if not isinstance(data, dict):
        errors.append("agent context must be a JSON object")
        return errors

    # response_format (required)
    rf = data.get("response_format")
    if not isinstance(rf, dict):
        errors.append("missing or invalid 'response_format' object")
    else:
        if "requires_structured_output" not in rf:
            errors.append("response_format missing 'requires_structured_output'")
        elif not isinstance(rf["requires_structured_output"], bool):
            errors.append("response_format.requires_structured_output must be a boolean")
        if "description" not in rf or not isinstance(rf.get("description"), str):
            errors.append("response_format missing 'description' string")

        requires = rf.get("requires_structured_output", False)
        schema = rf.get("output_schema")
        if requires and schema is None:
            errors.append(
                "response_format.output_schema should be provided when "
                "requires_structured_output is true"
            )
        elif requires and not isinstance(schema, dict):
            errors.append(
                "response_format.output_schema must be an object when structured output is required"
            )
        if not requires and schema is not None:
            errors.append(
                "response_format.output_schema should be null when "
                "requires_structured_output is false"
            )

    # authentication (required)
    auth = data.get("authentication")
    if not isinstance(auth, dict):
        errors.append("missing or invalid 'authentication' object")
    else:
        if "pre_authenticated" not in auth:
            errors.append("authentication missing 'pre_authenticated'")
        elif not isinstance(auth["pre_authenticated"], bool):
            errors.append("authentication.pre_authenticated must be a boolean")
        if "description" not in auth or not isinstance(auth.get("description"), str):
            errors.append("authentication missing 'description' string")

    # auth_mechanism (optional, additive — machine-readable auth contract).
    # Schema: see docs/worldsim-v5-technical-specifcation.md + plan §1.
    if "auth_mechanism" in data:
        mech = data.get("auth_mechanism")
        errors.extend(_validate_auth_mechanism(mech))
        # Cross-block check: a form_login recipe (top-level OR nested under
        # storage_state) is unusable without string username+password on the
        # authentication block. We only enforce this when the mechanism block
        # itself is shape-valid, to avoid cascading errors.
        if isinstance(mech, dict):
            recipe_present = False
            if mech.get("type") == "form_login" and isinstance(mech.get("form_login"), dict):
                recipe_present = True
            storage_sub = mech.get("storage_state")
            if isinstance(storage_sub, dict) and isinstance(storage_sub.get("form_login"), dict):
                recipe_present = True
            if recipe_present:
                creds = auth.get("credentials") if isinstance(auth, dict) else None
                if not isinstance(creds, dict):
                    errors.append(
                        "authentication.credentials must be an object with string "
                        "username+password when a form_login recipe is declared"
                    )
                else:
                    for key in ("username", "password"):
                        val = creds.get(key)
                        if not isinstance(val, str) or not val:
                            errors.append(
                                f"authentication.credentials.{key} must be a non-empty "
                                "string when a form_login recipe is declared"
                            )

    # agent_prompt_template (nullable)
    template = data.get("agent_prompt_template")
    if template is not None:
        if not isinstance(template, str):
            errors.append("agent_prompt_template must be a string or null")
        else:
            if "{{INSTRUCTION}}" not in template:
                errors.append("agent_prompt_template missing {{INSTRUCTION}} placeholder")
            if "{{START_URLS}}" not in template:
                errors.append("agent_prompt_template missing {{START_URLS}} placeholder")

    # site_context (required)
    sc = data.get("site_context")
    if not isinstance(sc, dict):
        errors.append("missing or invalid 'site_context' object")
    else:
        if "platform_name" not in sc or not isinstance(sc.get("platform_name"), str):
            errors.append("site_context missing 'platform_name' string")
        if "description" not in sc or not isinstance(sc.get("description"), str):
            errors.append("site_context missing 'description' string")

    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path("/workspace/output")
_INPUTS_DIR = Path("/workspace/inputs")
_VALIDATION_RESULT_PATH = _OUTPUT_DIR / "_validation_result.json"


def _load_json(path: Path) -> tuple[object, str | None]:
    """Load and parse a JSON file. Returns (data, error_or_none)."""
    if not path.exists():
        return None, f"file not found: {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON in {path}: {exc}"
    return data, None


def _emit_result(valid: bool, errors: list[str]) -> int:
    """Print JSON result, write validation result file, return exit code."""
    result = {"valid": valid, "errors": errors}
    print(json.dumps(result))

    if valid:
        _VALIDATION_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _VALIDATION_RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return 0 if valid else 1


def cmd_manifest(args: argparse.Namespace) -> int:
    path = _OUTPUT_DIR / "BENCHMARK_MANIFEST.json"
    data, err = _load_json(path)
    if err:
        return _emit_result(False, [err])
    errors = validate_manifest(data)
    return _emit_result(not errors, errors)


def cmd_profile(args: argparse.Namespace) -> int:
    path = _OUTPUT_DIR / "BENCHMARK_PROFILE.json"
    data, err = _load_json(path)
    if err:
        return _emit_result(False, [err])
    errors = validate_profile(data, site_name=args.site_name)
    return _emit_result(not errors, errors)


def cmd_benign_tasks(args: argparse.Namespace) -> int:
    path = _OUTPUT_DIR / "benign_tasks.json"
    data, err = _load_json(path)
    if err:
        return _emit_result(False, [err])
    errors = validate_benign_tasks(data, site_name=args.site_name)
    return _emit_result(not errors, errors)


def cmd_adversarial_tasks(args: argparse.Namespace) -> int:
    path = _OUTPUT_DIR / "adversarial_tasks.json"
    data, err = _load_json(path)
    if err:
        return _emit_result(False, [err])
    errors = validate_adversarial_tasks(data)
    return _emit_result(not errors, errors)


def cmd_diagnosis(args: argparse.Namespace) -> int:
    path = _OUTPUT_DIR / "diagnosis.json"
    data, err = _load_json(path)
    if err:
        return _emit_result(False, [err])
    errors = validate_diagnosis(data)
    return _emit_result(not errors, errors)


def cmd_ecological_validity(args: argparse.Namespace) -> int:
    path = _OUTPUT_DIR / "ecological_validity.json"
    data, err = _load_json(path)
    if err:
        return _emit_result(False, [err])
    errors = validate_ecological_validity(data)
    return _emit_result(not errors, errors)


def cmd_judge_recommendation(args: argparse.Namespace) -> int:
    path = _OUTPUT_DIR / "judge_recommendation.json"
    data, err = _load_json(path)
    if err:
        return _emit_result(False, [err])
    errors = validate_judge_recommendation(data)
    return _emit_result(not errors, errors)


def cmd_revised_task(args: argparse.Namespace) -> int:
    path = _OUTPUT_DIR / "revised_task.json"
    data, err = _load_json(path)
    if err:
        return _emit_result(False, [err])
    errors = validate_revised_task(data)
    return _emit_result(not errors, errors)


def cmd_variant_task(args: argparse.Namespace) -> int:
    path = _OUTPUT_DIR / "variant_task.json"
    data, err = _load_json(path)
    if err:
        return _emit_result(False, [err])
    errors = validate_variant_task(data)
    return _emit_result(not errors, errors)


def cmd_verification_capabilities(args: argparse.Namespace) -> int:
    path = _OUTPUT_DIR / "VERIFICATION_CAPABILITIES.json"
    data, err = _load_json(path)
    if err:
        return _emit_result(False, [err])
    errors = validate_verification_capabilities(data, site_name=args.site_name)
    return _emit_result(not errors, errors)


def cmd_data_model(args: argparse.Namespace) -> int:
    path = _OUTPUT_DIR / "DATA_MODEL.json"
    data, err = _load_json(path)
    if err:
        return _emit_result(False, [err])
    errors = validate_data_model_profile(data, site_name=args.site_name)
    return _emit_result(not errors, errors)


def cmd_injection_surface(args: argparse.Namespace) -> int:
    path = _OUTPUT_DIR / "INJECTION_SURFACE.json"
    data, err = _load_json(path)
    if err:
        return _emit_result(False, [err])
    input_data_model, input_err = _load_json(_INPUTS_DIR / "DATA_MODEL.json")
    if input_err:
        return _emit_result(False, [input_err])
    errors = validate_injection_surface(
        data,
        site_name=args.site_name,
        data_model=input_data_model,
    )
    return _emit_result(not errors, errors)


def cmd_agent_context(args: argparse.Namespace) -> int:
    path = _OUTPUT_DIR / "AGENT_CONTEXT.json"
    data, err = _load_json(path)
    if err:
        return _emit_result(False, [err])
    errors = validate_agent_context(data, site_name=args.site_name)
    return _emit_result(not errors, errors)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate WorldSim v5 sandbox output files.",
    )
    subparsers = parser.add_subparsers(dest="schema", required=True)

    subparsers.add_parser("manifest", help="Validate BENCHMARK_MANIFEST.json")

    profile_parser = subparsers.add_parser("profile", help="Validate BENCHMARK_PROFILE.json")
    profile_parser.add_argument("--site-name", required=True, help="Expected site_name")

    benign_parser = subparsers.add_parser("benign-tasks", help="Validate benign_tasks.json")
    benign_parser.add_argument("--site-name", required=True, help="Expected site name")

    subparsers.add_parser("adversarial-tasks", help="Validate adversarial_tasks.json")

    subparsers.add_parser("diagnosis", help="Validate diagnosis.json")

    subparsers.add_parser("ecological-validity", help="Validate ecological_validity.json")

    subparsers.add_parser("judge-recommendation", help="Validate judge_recommendation.json")

    subparsers.add_parser("revised-task", help="Validate revised_task.json")

    subparsers.add_parser("variant-task", help="Validate variant_task.json")

    vc_parser = subparsers.add_parser(
        "verification-capabilities", help="Validate VERIFICATION_CAPABILITIES.json"
    )
    vc_parser.add_argument("--site-name", required=True, help="Expected site_name")

    dm_parser = subparsers.add_parser("data-model", help="Validate DATA_MODEL.json")
    dm_parser.add_argument("--site-name", required=True, help="Expected site_name")

    is_parser = subparsers.add_parser("injection-surface", help="Validate INJECTION_SURFACE.json")
    is_parser.add_argument("--site-name", required=True, help="Expected site_name")

    ac_parser = subparsers.add_parser("agent-context", help="Validate AGENT_CONTEXT.json")
    ac_parser.add_argument("--site-name", required=True, help="Expected site_name")

    args = parser.parse_args()

    dispatch = {
        "manifest": cmd_manifest,
        "profile": cmd_profile,
        "benign-tasks": cmd_benign_tasks,
        "adversarial-tasks": cmd_adversarial_tasks,
        "diagnosis": cmd_diagnosis,
        "ecological-validity": cmd_ecological_validity,
        "judge-recommendation": cmd_judge_recommendation,
        "revised-task": cmd_revised_task,
        "variant-task": cmd_variant_task,
        "agent-context": cmd_agent_context,
        "verification-capabilities": cmd_verification_capabilities,
        "data-model": cmd_data_model,
        "injection-surface": cmd_injection_surface,
    }
    handler = dispatch.get(args.schema)
    if handler is None:
        print(json.dumps({"valid": False, "errors": [f"unknown schema: {args.schema}"]}))
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
