from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldsim_agentlab_runner.worldsim_task import latest_assistant_message


def write_worldsim_artifacts(
    output_dir: Path,
    *,
    episode_info: list[Any],
    final_result: str | None,
    status: str,
    errors: list[str],
    task_instruction: str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = [_history_entry(step) for step in episode_info if getattr(step, "step", None) is not None]
    (output_dir / "history.json").write_text(
        json.dumps(
            {
                "history": history,
                "runner": "agentlab",
                "trajectory_format": "worldsim-agentlab-history-v1",
                "partial": status != "success",
                "errors": list(errors),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "final_response.json").write_text(
        json.dumps(
            {
                "status": status,
                "final_result": final_result,
                "errors": list(errors),
                "steps": max(0, len(history) - 1),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_needham_trace(output_dir, task_instruction=task_instruction, history=history)


def final_result_from_env(env: Any, fallback: str | None = None) -> str | None:
    chat = getattr(getattr(env, "unwrapped", env), "chat", None)
    message = latest_assistant_message(getattr(chat, "messages", None))
    return message or fallback


def _history_entry(step: Any) -> dict[str, Any]:
    obs = getattr(step, "obs", None)
    if not isinstance(obs, dict):
        obs = {}
    agent_info = getattr(step, "agent_info", None)
    if not isinstance(agent_info, dict):
        agent_info = {}
    action = getattr(step, "action", None)
    state = {
        "url": _string_or_none(obs.get("url")),
        "title": _active_title(obs),
        "screenshot_path": f"screenshots/step_{getattr(step, 'step', 0)}.png",
    }
    result_items: list[dict[str, Any]] = []
    last_error = obs.get("last_action_error")
    if isinstance(last_error, str) and last_error:
        result_items.append({"extracted_content": last_error})
    task_info = getattr(step, "task_info", None)
    if isinstance(task_info, dict) and isinstance(task_info.get("worldsim_final_result"), str):
        result_items.append({"extracted_content": task_info["worldsim_final_result"]})
    return {
        "model_output": {
            "thinking": _string_or_none(agent_info.get("think")),
            "evaluation_previous_goal": None,
            "memory": None,
            "next_goal": None,
            "action": _action_projection(action),
        },
        "result": result_items,
        "state": state,
        "state_message": _state_message(obs),
        "agentlab": {
            "step": getattr(step, "step", None),
            "reward": getattr(step, "reward", None),
            "raw_reward": getattr(step, "raw_reward", None),
            "terminated": getattr(step, "terminated", None),
            "truncated": getattr(step, "truncated", None),
        },
    }


def _action_projection(action: Any) -> list[dict[str, Any]]:
    if action is None:
        return []
    return [{"agentlab_action": {"raw": str(action)}}]


def _active_title(obs: dict[str, Any]) -> str | None:
    titles = obs.get("open_pages_titles")
    active = obs.get("active_page_index")
    try:
        index = int(active[0]) if hasattr(active, "__getitem__") else int(active)
    except Exception:
        index = 0
    if isinstance(titles, (list, tuple)) and 0 <= index < len(titles):
        return _string_or_none(titles[index])
    return None


def _state_message(obs: dict[str, Any]) -> str:
    parts = []
    url = obs.get("url")
    if isinstance(url, str) and url:
        parts.append(f"URL: {url}")
    last_action = obs.get("last_action")
    if isinstance(last_action, str) and last_action:
        parts.append(f"Last action: {last_action}")
    last_error = obs.get("last_action_error")
    if isinstance(last_error, str) and last_error:
        parts.append(f"Last action error: {last_error}")
    return "\n".join(parts)


def _write_needham_trace(output_dir: Path, *, task_instruction: str, history: list[dict[str, Any]]) -> None:
    messages: list[dict[str, Any]] = [{"role": "user", "text": task_instruction}]
    for index, step in enumerate(history):
        model_output = step.get("model_output") if isinstance(step, dict) else {}
        action_items = model_output.get("action") if isinstance(model_output, dict) else []
        tool_calls = []
        if isinstance(action_items, list):
            for action in action_items:
                if isinstance(action, dict) and "agentlab_action" in action:
                    tool_calls.append(
                        {
                            "id": str(len(tool_calls)),
                            "function": "agentlab_action",
                            "arguments": action["agentlab_action"],
                        }
                    )
        text = model_output.get("thinking") if isinstance(model_output, dict) else ""
        messages.append(
            {
                "role": "assistant",
                "text": text or "",
                "tool_calls": tool_calls or None,
                "provenance": {"source": "agentlab_history", "source_step": index},
            }
        )
        result_text = "\n".join(
            str(item.get("extracted_content", ""))
            for item in step.get("result", [])
            if isinstance(item, dict)
        )
        if result_text or tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "text": result_text,
                    "function": "agentlab_action",
                    "provenance": {"source": "agentlab_history", "source_step": index},
                }
            )
    (output_dir / "needham_trace.json").write_text(
        json.dumps(
            {
                "format": "needham-agentlab-v1",
                "transcript_format": "needham-xml-v1",
                "messages": messages,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "needham_trace.xml").write_text(_simple_xml(messages), encoding="utf-8")


def _simple_xml(messages: list[dict[str, Any]]) -> str:
    lines = ["<transcript>"]
    for message in messages:
        role = str(message.get("role") or "user")
        text = _xml_escape(str(message.get("text") or ""))
        lines.append(f'  <message role="{_xml_escape(role)}">{text}</message>')
    lines.append("</transcript>")
    return "\n".join(lines)


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None
