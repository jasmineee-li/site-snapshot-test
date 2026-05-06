from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from worldsim_agentlab_runner.worldsim_task import latest_assistant_message

TRACE_FORMAT = "needham-agentlab-v1"
TRANSCRIPT_FORMAT = "needham-xml-v1"
_TOOL_RESULT_CAP = 3000


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
            "observation_text": _observation_text(obs),
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
    next_tool_id = 0
    for index, step in enumerate(history):
        model_output = step.get("model_output") if isinstance(step, dict) else {}
        action_items = model_output.get("action") if isinstance(model_output, dict) else []
        tool_calls = []
        if isinstance(action_items, list):
            for action in action_items:
                if isinstance(action, dict) and "agentlab_action" in action:
                    tool_calls.append(
                        {
                            "id": str(next_tool_id),
                            "function": "agentlab_action",
                            "arguments": action["agentlab_action"],
                        }
                    )
                    next_tool_id += 1
        text = model_output.get("thinking") if isinstance(model_output, dict) else ""
        messages.append(
            {
                "role": "assistant",
                "text": text or "",
                "tool_calls": tool_calls or None,
                "provenance": {"source": "agentlab_history", "source_step": index},
            }
        )
        result_text = _needham_tool_result_text(step)
        if result_text or tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "text": result_text,
                    "function": "agentlab_action",
                    "provenance": {"source": "agentlab_history", "source_step": index},
                }
            )
    xml = _format_needham_xml(messages)
    (output_dir / "needham_trace.json").write_text(
        json.dumps(
            {
                "format": TRACE_FORMAT,
                "transcript_format": TRANSCRIPT_FORMAT,
                "source": "agentlab_history",
                "messages": messages,
                "xml": xml,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "needham_trace.xml").write_text(xml, encoding="utf-8")


def _needham_tool_result_text(step: dict[str, Any]) -> str:
    agentlab = step.get("agentlab") if isinstance(step, dict) else {}
    if isinstance(agentlab, dict):
        observation = agentlab.get("observation_text")
        if isinstance(observation, str) and observation:
            return observation[:_TOOL_RESULT_CAP]
    result_text = "\n".join(
        str(item.get("extracted_content", ""))
        for item in step.get("result", [])
        if isinstance(item, dict)
    )
    if result_text:
        return result_text[:_TOOL_RESULT_CAP]
    state_message = step.get("state_message") if isinstance(step, dict) else None
    if isinstance(state_message, str):
        return state_message[:_TOOL_RESULT_CAP]
    return ""


def _observation_text(obs: dict[str, Any]) -> str:
    axtree = obs.get("axtree_txt")
    if isinstance(axtree, str) and axtree:
        return axtree
    return _state_message(obs)


def _format_needham_xml(messages: list[dict[str, Any]]) -> str:
    return "\n".join(_format_needham_message(message) for message in messages) + "\n\n"


def _format_needham_message(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "user")
    text = _xml_escape(str(message.get("text") or ""))
    function = message.get("function")
    if role == "tool" and isinstance(function, str):
        result = f'<message role="{_xml_escape(role)}", function="{_xml_escape(function)}">\n{text}'
    else:
        result = f'<message role="{_xml_escape(role)}">\n{text}'
    tool_calls = message.get("tool_calls")
    if role == "assistant" and isinstance(tool_calls, list):
        result += "\n<tool_calls>"
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            call_id = str(tool_call.get("id") or "")
            call_function = str(tool_call.get("function") or "")
            result += f'<tool_call id="{_xml_escape(call_id)}" function="{_xml_escape(call_function)}">'
            arguments = tool_call.get("arguments")
            if isinstance(arguments, dict):
                for arg_name, arg_value in arguments.items():
                    result += (
                        f'\n<argument name="{_xml_escape(str(arg_name))}">'
                        f"{_xml_escape(str(arg_value))}</argument>"
                    )
            result += "\n</tool_call>"
        result += "\n</tool_calls>"
    result += "\n</message>"
    return result


def _xml_escape(value: str) -> str:
    return escape(value, entities={'"': "&quot;"})


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None
