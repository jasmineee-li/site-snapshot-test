"""Helpers for building benchmark-specific agent prompts from discovered context."""

from __future__ import annotations

import json
from typing import Any


def build_agent_prompt(
    agent_context: dict[str, Any] | None,
    instruction: str,
    start_urls: list[str],
    *,
    task: dict[str, Any] | None = None,
) -> str | None:
    """Return a benchmark-specific agent prompt, or None for the default fallback."""
    if not agent_context:
        return None

    sections: list[str] = []
    site_header = _build_site_header(agent_context.get("site_context"))
    auth_section = _build_auth_section(agent_context.get("authentication"))
    rendered_template = _render_prompt_template(agent_context, instruction, start_urls)
    if rendered_template:
        sections.append(rendered_template)
        if site_header:
            sections.append(site_header)
        if auth_section:
            sections.append(auth_section)
    else:
        if site_header:
            sections.append(site_header)

        if auth_section:
            sections.append(auth_section)

        sections.append(_build_task_section(instruction, start_urls))

    response_format_section = _build_response_format_section(
        agent_context.get("response_format"),
        task=task,
    )
    if response_format_section:
        sections.append(response_format_section)

    prompt = "\n\n".join(section for section in sections if section)
    return prompt or None


def _render_prompt_template(
    agent_context: dict[str, Any],
    instruction: str,
    start_urls: list[str],
) -> str | None:
    template = agent_context.get("agent_prompt_template")
    if not isinstance(template, str) or not template.strip():
        return None

    return template.replace("{{INSTRUCTION}}", instruction).replace(
        "{{START_URLS}}",
        _format_start_urls(start_urls),
    )


def _build_site_header(site_context: object) -> str | None:
    if not isinstance(site_context, dict):
        return None

    platform = str(site_context.get("platform_name", "")).strip()
    description = str(site_context.get("description", "")).strip()
    if not platform and not description:
        return None

    if platform and description:
        return f"You are an autonomous web agent operating on {platform}. {description}"
    if platform:
        return f"You are an autonomous web agent operating on {platform}."
    return description


def _build_auth_section(authentication: object) -> str | None:
    if not isinstance(authentication, dict):
        return None

    blocks: list[str] = []
    description = str(authentication.get("description", "")).strip()
    if description:
        blocks.append(description)

    credentials = authentication.get("credentials")
    rendered_credentials = _render_credentials(credentials)
    if rendered_credentials:
        blocks.append(rendered_credentials)

    if not blocks:
        return None

    return "## Authentication\n" + "\n\n".join(blocks)


def _render_credentials(credentials: object) -> str | None:
    if not isinstance(credentials, dict):
        return None

    lines = [
        f"- {key}: {value}"
        for key, value in credentials.items()
        if isinstance(key, str) and str(value).strip()
    ]
    if not lines:
        return None
    return "Credentials:\n" + "\n".join(lines)


def _build_task_section(instruction: str, start_urls: list[str]) -> str:
    return (
        "## Task\n"
        f"- **Objective:** {instruction}\n"
        f"- **Start URLs:**\n{_format_start_urls(start_urls)}"
    )


def _build_response_format_section(
    response_format: object,
    *,
    task: dict[str, Any] | None = None,
) -> str | None:
    if not isinstance(response_format, dict):
        return None

    if not response_format.get("requires_structured_output"):
        return None

    schema = response_format.get("output_schema")
    if not isinstance(schema, dict):
        return None

    blocks: list[str] = []
    description = str(response_format.get("description", "")).strip()
    if description:
        blocks.append(description)

    per_task_requirement = _render_per_task_format_requirement(response_format, task)
    if per_task_requirement:
        blocks.append(per_task_requirement)

    reward_requirement = _render_reward_schema_format_requirement(task)
    if reward_requirement:
        blocks.append(reward_requirement)

    schema_str = json.dumps(schema, indent=2)
    blocks.append(
        f"Return your final answer as JSON matching this schema:\n```json\n{schema_str}\n```"
    )
    return "## Response Format\n" + "\n\n".join(blocks)


def _render_per_task_format_requirement(
    response_format: dict[str, Any],
    task: dict[str, Any] | None,
) -> str | None:
    if not isinstance(task, dict):
        return None

    field_name = response_format.get("per_task_format_field")
    if not isinstance(field_name, str) or not field_name.strip():
        return None

    instantiation_dict = task.get("instantiation_dict")
    if not isinstance(instantiation_dict, dict):
        return None

    value = instantiation_dict.get(field_name)
    if value is None:
        return None
    if isinstance(value, str):
        rendered = value.strip()
    else:
        rendered = json.dumps(value, indent=2, sort_keys=True)
    if not rendered:
        return None
    return f"Per-task format requirement:\n{rendered}"


def _render_reward_schema_format_requirement(task: dict[str, Any] | None) -> str | None:
    """Render a task-local retrieved_data schema without leaking expected answers."""
    config = _first_agent_response_config(task)
    if not config:
        return None

    expected = config.get("expected")
    if isinstance(expected, dict):
        task_type = str(expected.get("task_type") or "").casefold()
        status = str(expected.get("status") or "").upper()
        if task_type and task_type != "retrieve":
            return None
        if status == "NOT_FOUND_ERROR":
            return None

    results_schema = config.get("results_schema")
    if not isinstance(results_schema, dict) or not results_schema:
        return None

    blocks = [
        "Task-specific retrieved_data schema:\n"
        f"```json\n{json.dumps(results_schema, indent=2, sort_keys=True)}\n```"
    ]
    if _is_array_of_strings_schema(results_schema):
        blocks.append(
            "Set `retrieved_data` to an array of strings. For a single-value answer, "
            'use a one-element array such as `["value"]`, not a bare string or object.'
        )
        instruction = str(task.get("instruction") or "") if isinstance(task, dict) else ""
        if "blank" in instruction.casefold() and "populated" in instruction.casefold():
            blocks.append(
                "For blank/populated status tasks, the one array item must be either "
                "`blank` or `populated`."
            )
    return "\n\n".join(blocks)


def _first_agent_response_config(task: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(task, dict):
        return None
    reward = task.get("reward_function")
    if not isinstance(reward, dict):
        return None
    benign_reward = reward.get("benign_reward")
    if isinstance(benign_reward, dict):
        reward = benign_reward
    evals = reward.get("eval")
    if isinstance(evals, list):
        for config in evals:
            if isinstance(config, dict) and config.get("evaluator") == "AgentResponseEvaluator":
                return config
    if reward.get("type") == "AgentResponseEvaluator":
        return reward
    return None


def _is_array_of_strings_schema(schema: dict[str, Any]) -> bool:
    if schema.get("type") != "array":
        return False
    items = schema.get("items")
    return isinstance(items, dict) and items.get("type", "string") == "string"


def _format_start_urls(start_urls: list[str]) -> str:
    if not start_urls:
        return "  - (none provided)"
    return "\n".join(f"  - {url}" for url in start_urls)
