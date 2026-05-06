from __future__ import annotations

import ast
import json
import re
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
    history = [
        _history_entry(step) for step in episode_info if getattr(step, "step", None) is not None
    ]
    (output_dir / "history.json").write_text(
        json.dumps(
            {
                "history": history,
                "runner": "agentlab",
                "trajectory_format": "worldsim-agentlab-history-v1",
                "partial": _episode_partial(history=history, final_result=final_result),
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
    _write_needham_trace(
        output_dir,
        task_instruction=task_instruction,
        history=history,
        final_result=final_result,
    )


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
            "thinking": _extract_agentlab_reasoning(agent_info),
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
    parsed = _parse_agentlab_action(action)
    return [{parsed["name"]: parsed["arguments"]}]


def _parse_agentlab_action(action: Any) -> dict[str, Any]:
    if isinstance(action, dict) and action:
        name = str(next(iter(action)))
        raw_args = action.get(name)
        arguments = raw_args if isinstance(raw_args, dict) else {"value": raw_args}
        return {"name": name, "arguments": {**arguments, "raw": str(action)}}
    raw = str(action)
    try:
        node = ast.parse(raw, mode="eval").body
    except SyntaxError:
        return {"name": "agentlab_action", "arguments": {"raw": raw}}
    if not isinstance(node, ast.Call):
        return {"name": "agentlab_action", "arguments": {"raw": raw}}
    if isinstance(node.func, ast.Name):
        name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        name = node.func.attr
    else:
        name = "agentlab_action"
    arguments: dict[str, Any] = {"raw": raw}
    positional: list[Any] = []
    for item in node.args:
        try:
            positional.append(ast.literal_eval(item))
        except Exception:
            positional.append(ast.unparse(item))
    if positional:
        arguments["args"] = positional
    for keyword in node.keywords:
        if keyword.arg is None:
            continue
        try:
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        except Exception:
            arguments[keyword.arg] = ast.unparse(keyword.value)
    return {"name": name or "agentlab_action", "arguments": arguments}


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


def _episode_partial(*, history: list[dict[str, Any]], final_result: str | None) -> bool:
    if not history:
        return True
    if isinstance(final_result, str) and final_result.strip():
        return False
    last = history[-1].get("agentlab") if isinstance(history[-1], dict) else {}
    if isinstance(last, dict) and (last.get("terminated") or last.get("truncated")):
        return False
    return True


def _write_needham_trace(
    output_dir: Path,
    *,
    task_instruction: str,
    history: list[dict[str, Any]],
    final_result: str | None,
) -> None:
    messages: list[dict[str, Any]] = [{"role": "user", "text": task_instruction}]
    next_tool_id = 0
    for index, step in enumerate(history):
        model_output = step.get("model_output") if isinstance(step, dict) else {}
        action_items = model_output.get("action") if isinstance(model_output, dict) else []
        tool_calls = []
        if isinstance(action_items, list):
            for action in action_items:
                if isinstance(action, dict) and action:
                    function = str(next(iter(action)))
                    arguments = action.get(function)
                    if not isinstance(arguments, dict):
                        arguments = {"value": arguments}
                    tool_calls.append(
                        {
                            "id": str(next_tool_id),
                            "function": function,
                            "arguments": arguments,
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
                    "function": tool_calls[0]["function"] if tool_calls else "agentlab_action",
                    "provenance": {"source": "agentlab_history", "source_step": index},
                }
            )
    if (
        isinstance(final_result, str)
        and final_result.strip()
        and not _final_result_already_present(final_result, messages)
    ):
        messages.append(
            {
                "role": "assistant",
                "text": final_result.strip(),
                "tool_calls": None,
                "provenance": {"source": "agentlab_final_response"},
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


def _extract_agentlab_reasoning(agent_info: dict[str, Any]) -> str | None:
    """Preserve AgentLab model intent from all known reasoning surfaces.

    AgentLab's parsed ``think`` field is not enough for parity: OpenRouter
    reasoning models can store hidden reasoning under ``chat_messages`` and
    scratchpad-style experiments can put free-form reasoning in tagged content.
    """

    parts: list[str] = []
    raw_content = ""
    chat_messages = agent_info.get("chat_messages")
    messages = _safe_get(chat_messages, "messages", []) or []
    if isinstance(messages, (list, tuple)):
        for message in reversed(messages):
            if _safe_get(message, "role") != "assistant":
                continue
            reasoning = _safe_get(message, "reasoning", "")
            if isinstance(reasoning, str) and reasoning.strip():
                parts.append(reasoning.strip())
            content = _safe_get(message, "content", "")
            if isinstance(content, str):
                raw_content = content
            break

    think = agent_info.get("think")
    if isinstance(think, str) and think.strip():
        _append_if_distinct(parts, think.strip())

    for tag in ("unfiltered_observations", "scratchpad", "reflection"):
        for match in re.finditer(rf"<{tag}>(.*?)</{tag}>", raw_content, re.DOTALL | re.IGNORECASE):
            block = match.group(1).strip()
            if block:
                _append_if_distinct(parts, block)

    return "\n\n".join(parts) if parts else None


def _append_if_distinct(parts: list[str], value: str) -> None:
    if any(value in existing or existing in value for existing in parts):
        return
    parts.append(value)


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _final_result_already_present(final_result: str, messages: list[dict[str, Any]]) -> bool:
    final = final_result.strip()
    for message in reversed(messages):
        text = message.get("text")
        if isinstance(text, str) and (final == text.strip() or final in text):
            return True
    return False


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
            result += (
                f'<tool_call id="{_xml_escape(call_id)}" function="{_xml_escape(call_function)}">'
            )
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
