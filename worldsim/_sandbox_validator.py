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


def validate_seed_sql(statement: str) -> str | None:
    """Return an error string if the SQL statement is invalid, else None."""
    normalized = statement.strip()
    if not normalized:
        return "SQL seed statement is empty"
    if _MULTI_STATEMENT_PATTERN.search(normalized.rstrip(";")):
        return "SQL seed must be a single statement"
    if _DISALLOWED_SQL_KEYWORDS.search(normalized):
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
        errors.append(
            f"profile site_name mismatch: expected {site_name!r}, got {profile_site!r}"
        )

    # Build known entities and fields from data_model
    known_entities: set[str] = set()
    known_fields: set[str] = set()
    for entity in data.get("data_model", []):
        entity_name = entity.get("entity", "")
        if entity_name:
            known_entities.add(entity_name)
        for field in entity.get("fields", []):
            field_name = field.get("name", "")
            if field_name:
                known_fields.add(field_name)
        storage = entity.get("storage", "")
        if storage:
            known_fields.add(storage)

    # Cross-ref injection surfaces against data_model
    for surface in data.get("injection_surface", []):
        source = surface.get("source_field", "")
        if source and "." in source:
            entity_name = source.split(".")[0]
            if entity_name not in known_entities and known_entities:
                errors.append(
                    f"injection surface {surface.get('id', '?')!r} references "
                    f"unknown entity {entity_name!r} in {source!r}"
                )
            field_name = source.split(".")[-1]
            if field_name not in known_fields and known_fields:
                errors.append(
                    f"injection surface {surface.get('id', '?')!r} references "
                    f"unknown field {source!r}"
                )

    return errors


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
                errors.append(
                    f"{prefix} eval[{ei}] uses unsupported evaluator {evaluator!r}"
                )

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
                    str(t.get("id", "")): t
                    for t in benign_tasks
                    if isinstance(t, dict)
                }
        except json.JSONDecodeError:
            errors.append("could not parse benign_tasks.json for cross-reference")

    # Minimal schema only requires these fields from Claude's output.
    _REQUIRED_FIELDS = (
        "id", "benign_task_id", "adversarial_data_seed",
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
            errors.append(
                f"{prefix} references unknown benign_task_id {benign_task_id!r}"
            )
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
            errors.append(f"{prefix} missing adversarial_reward (neither top-level nor in reward_function)")
        elif not isinstance(adv_reward, dict) or not adv_reward:
            errors.append(f"{prefix} adversarial_reward must be a non-empty object")
        elif "type" not in adv_reward:
            errors.append(f"{prefix} adversarial_reward missing type field")

        # Validate adversarial_data_seed
        seed_errors = validate_data_seed(
            task.get("adversarial_data_seed"), allow_none=False
        )
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
        "reward_bug", "seed_bug", "impossible", "too_hard", "agent_limitation",
    }
    root_cause = data.get("root_cause")
    if root_cause not in valid_root_causes:
        errors.append(
            f"root_cause must be one of {sorted(valid_root_causes)}, "
            f"got {root_cause!r}"
        )

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
                errors.append(
                    f"probes[{i}] score must be 0.0-1.0, got {score}"
                )

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
        "authority_framing", "semantic_coherence", "specificity",
        "visual_concealment", "verbosity_adjustment",
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path("/workspace/output")
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
        _VALIDATION_RESULT_PATH.write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

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
    }
    handler = dispatch.get(args.schema)
    if handler is None:
        print(json.dumps({"valid": False, "errors": [f"unknown schema: {args.schema}"]}))
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
