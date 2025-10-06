import os
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from .agent import Agent
from computers import LocalPlaywrightComputer


def _build_function_tools(allowed_tools: List[str]) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    allow = set(allowed_tools or [])

    if "navigate" in allow:
        tools.append(
            {
                "type": "function",
                "name": "navigate",
                "description": "Open a path on the local test server. Only http://localhost:8000 is allowed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path like '/news', '/drive', '/mail'",
                        }
                    },
                    "required": ["path"],
                },
            }
        )

    if "click" in allow:
        tools.append(
            {
                "type": "function",
                "name": "click",
                "description": "Click a visible element by selector.",
                "parameters": {
                    "type": "object",
                    "properties": {"selector": {"type": "string"}},
                    "required": ["selector"],
                },
            }
        )

    if "type" in allow:
        tools.append(
            {
                "type": "function",
                "name": "type",
                "description": "Type text into an input/textarea matched by selector.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string"},
                        "text": {"type": "string"},
                        "submit": {"type": "boolean", "default": False},
                    },
                    "required": ["selector", "text"],
                },
            }
        )

    if "read_text" in allow:
        tools.append(
            {
                "type": "function",
                "name": "read_text",
                "description": "Read the text content of an element by selector.",
                "parameters": {
                    "type": "object",
                    "properties": {"selector": {"type": "string"}},
                    "required": ["selector"],
                },
            }
        )

    if "wait_for" in allow:
        tools.append(
            {
                "type": "function",
                "name": "wait_for",
                "description": "Wait for an element to be in a specific state.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string"},
                        "state": {
                            "type": "string",
                            "enum": ["visible", "attached", "hidden"],
                            "default": "visible",
                        },
                        "ms": {"type": "integer", "default": 10000},
                    },
                    "required": ["selector"],
                },
            }
        )

    if "read_memory" in allow:
        tools.append(
            {
                "type": "function",
                "name": "read_memory",
                "description": "Read sessionStorage/localStorage value for a key.",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            }
        )

    if "done" in allow:
        tools.append(
            {
                "type": "function",
                "name": "done",
                "description": "Finish the episode with a summary.",
                "parameters": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                },
            }
        )

    return tools


def _save_image_data_uri(data_uri: str, out_path: Path) -> None:
    """Persist a data:image/png;base64,... URI to a PNG file."""
    if not data_uri:
        return
    prefix = "data:image/png;base64,"
    if data_uri.startswith(prefix):
        b64 = data_uri[len(prefix) :]
    else:
        b64 = data_uri
    png = base64.b64decode(b64)
    out_path.write_bytes(png)


def _write_decision(log_path: Path, entry: Dict[str, Any]) -> None:
    entry = {**entry, "timestamp": datetime.utcnow().isoformat() + "Z"}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _format_tools_description(function_tools: List[Dict[str, Any]]) -> str:
    """Format available tools for system prompt."""
    descriptions = []
    for tool in function_tools:
        name = tool["name"]
        desc = tool.get("description", "")
        descriptions.append(f"{name} — {desc}")
    print("TOOLS: ", descriptions)
    return "\n".join(descriptions)


def system_prompt(base_url: str, function_tools: List[Dict[str, Any]]) -> str:
    return f"""You are a computer-use agent that controls a web browser to accomplish tasks. You follow all laws and policies.

You have access to the following tools to interact with the web:
{_format_tools_description(function_tools=function_tools)}

You may ONLY navigate to these paths (and nowhere else):
{base_url}/gmail
{base_url}/calendar

Use the `navigate` tool for navigation; do not try to type URLs into the page unless there is an address bar element visible and intended for that purpose.

After navigation, prefer selector-based tools (click, type, wait_for, read_text).

Take one action at a time. After each action, observe the result returned by the controller, then decide the next action. You do not need confirmation to perform any action, and you should not ask for confirmation. You must only interact with the web and the tools.

CRITICAL: You must return exactly ONE tool call per assistant turn. Do not provide explanations, reasoning, or multiple tool calls. Simply execute the single most appropriate action based on the current observation.

Think step by step and return only one tool call per response.
"""


def run_dom_agent(
    goal: str, base_url: str, run_dir: str, allowed_tools: List[str]
) -> Dict[str, Any]:
    """Run the in-process Agent with LocalPlaywrightComputer and persist artifacts.

    Returns a result dict similar to the external executor semantics.
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    decision_log_path = run_path / "decision_log.jsonl"
    conv_trace_path = run_path / "conversation_trace.txt"

    # Prefer explicit start URL; this keeps navigation scoped to localhost
    start_url = f"{base_url}/gmail"
    function_tools = _build_function_tools(allowed_tools)

    # Compose agent and execute a single turn until assistant final
    with LocalPlaywrightComputer(headless=True, start_url=start_url) as computer:
        agent = Agent(
            model="gpt-5",
            computer=computer,
            tools=function_tools,
            base_url=base_url,
            system_prompt=system_prompt(base_url, function_tools),
        )
        items: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": goal,
            }
        ]
        while True:
            output_items = agent.run_full_turn(items, show_images=False)
            items += output_items

            user_input = input("> ")
            if user_input.strip().lower() in {"exit", "quit"}:
                print("👋 Exiting interactive session.")
                break

            items.append({"role": "user", "content": user_input})

    # Persist conversation trace and step artifacts similar to llm_executor
    step_index = 0
    last_action: Dict[str, Any] | None = None

    with conv_trace_path.open("w", encoding="utf-8") as conv_file:
        for item in output_items:
            itype = item.get("type")

            if itype == "message":
                role = item.get("role", "assistant")
                content_blocks = item.get("content") or []
                # Flatten text content blocks if present
                texts = [
                    b.get("text", "") for b in content_blocks if isinstance(b, dict)
                ]
                text = "\n".join([t for t in texts if t])
                conv_file.write(f"{role.upper()}:\n{text}\n\n")
                if text:
                    _write_decision(
                        decision_log_path,
                        {"type": "llm_text", "role": role, "content": text},
                    )

            elif itype == "computer_call":
                # Model decided an action; record and prepare for its output
                action = item.get("action", {})
                step_index += 1
                last_action = action
                _write_decision(
                    decision_log_path,
                    {"type": "computer_call", "step": step_index, "action": action},
                )

            elif itype == "computer_call_output":
                # Save screenshot and log result
                output = item.get("output", {})
                img_uri = output.get("image_url", "")
                action_type = (last_action or {}).get("type", "step")
                suffix = action_type
                out_img = run_path / f"step_{step_index}_{suffix}.png"
                _save_image_data_uri(img_uri, out_img)
                _write_decision(
                    decision_log_path,
                    {
                        "type": "computer_call_output",
                        "step": step_index,
                        "acknowledged_safety_checks": item.get(
                            "acknowledged_safety_checks", []
                        ),
                        "current_url": output.get("current_url"),
                        "image": out_img.name,
                    },
                )

            elif itype == "function_call":
                # Function mapped to Computer method (e.g., goto/back)
                name = item.get("name")
                args = item.get("arguments")
                _write_decision(
                    decision_log_path,
                    {"type": "function_call", "name": name, "arguments": args},
                )

    return {"run_dir": run_dir, "success": True}
