#!/usr/bin/env python3
"""
LLM-powered UI agent using OpenAI function calling.
Uses Playwright to execute actions and logs every step.
"""

import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from openai import OpenAI
from playwright.sync_api import sync_playwright, Page
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SafetyError(Exception):
    """Raised when a tool call violates safety constraints."""

    pass


class LLMUIAgent:
    def __init__(
        self,
        goal: str = None,  # NEW: Dynamic goal
        base_url: str = "http://localhost:8000",
        max_steps: int = 30,
        flow_type: str = "unknown",
        run_dir: str = None,
        allowed_tools: List[str] = None,  # Restrict tools per benchmark
        start_page: str,  # Starting page domain
    ):
        self.goal = goal
        self.base_url = base_url
        self.max_steps = max_steps
        self.flow_type = flow_type
        self.start_page = start_page
        self.run_id = str(uuid.uuid4())[:8]
        self.step_count = 0

        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Load tool definitions
        tools_path = Path(__file__).parent / "tools.json"
        with open(tools_path, "r") as f:
            all_tools = json.load(f)

        # Filter tools if specified
        if allowed_tools:
            self.tools = [t for t in all_tools if t["name"] in allowed_tools]
            self.allowed_tools = allowed_tools
        else:
            self.tools = all_tools
            self.allowed_tools = [t["name"] for t in all_tools]

        self.log_dir = Path(run_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.decision_log_path = self.log_dir / "decision_log.jsonl"
        self.conv_trace_path = self.log_dir / "conversation_trace.txt"

        # Initialize Playwright
        self.playwright = sync_playwright()
        self.browser = self.playwright.start().chromium.launch(headless=True)
        self.context = self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        )
        self.page = self.context.new_page()

        print(f"🤖 LLM Agent initialized - Run ID: {self.run_id}")
        print(f"📝 Logs: {self.log_dir}")

    def log_step(self, step_type: str, data: Dict[str, Any]):
        """Log a step to the decision log."""
        log_entry = {
            "run_id": self.run_id,
            "step_id": self.step_count,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": step_type,
            **data,
        }

        with open(self.decision_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(
            f"[{self.step_count}] {step_type}: {data.get('action', data.get('summary', str(data)[:100]))}"
        )

    def shortlist_elements(self, page: Page) -> Dict[str, Any]:
        """Extract key page information and actionable elements."""
        try:
            # Get basic page info
            url = page.url
            title = page.title()

            # Get actionable elements with stable selectors
            elements = []

            # Look for common interactive elements
            selectors = [
                "button",
                "a[href]",
                "input",
                "textarea",
                "[role='button']",
                "[data-name]",
                "#compose-btn",
                "#send-btn",
                "#to",
                "#subject",
                "#body",
                "#drive-link",
                "#mail-link",
                ".file-row",
                ".btn",
                "#go-drive",
            ]

            for selector in selectors:
                try:
                    elem_locators = page.locator(selector)
                    count = elem_locators.count()
                    for i in range(min(count, 5)):  # Limit to 5 per selector type
                        elem = elem_locators.nth(i)
                        if elem.is_visible():
                            text = elem.text_content() or ""
                            text = text.strip()[:50]  # Limit text length

                            # Get a stable selector
                            elem_id = elem.get_attribute("id")
                            elem_data_name = elem.get_attribute("data-name")

                            if elem_id:
                                stable_selector = f"#{elem_id}"
                            elif elem_data_name:
                                stable_selector = f"[data-name='{elem_data_name}']"
                            else:
                                stable_selector = selector

                            elements.append(
                                {
                                    "selector": stable_selector,
                                    "text": text,
                                    "tag": elem.evaluate(
                                        "el => el.tagName.toLowerCase()"
                                    ),
                                    "type": elem.get_attribute("type") or "",
                                }
                            )
                except Exception:
                    continue

            # Remove duplicates and limit total
            seen = set()
            unique_elements = []
            for elem in elements:
                key = (elem["selector"], elem["text"])
                if key not in seen and len(unique_elements) < 20:
                    seen.add(key)
                    unique_elements.append(elem)

            return {"url": url, "title": title, "elements": unique_elements}
        except Exception as e:
            return {"url": page.url, "title": "", "elements": [], "error": str(e)}

    def validate_selector(self, selector: str) -> bool:
        """Validate selector against safety allowlist."""
        # Allow common safe selectors
        safe_patterns = [
            r"^#[\w-]+$",  # IDs
            r"^\[data-[\w-]+=[\'\"][\w-]+[\'\"]\]$",  # Data attributes
            r"^\.[\w-]+$",  # Classes
            r"^[\w-]+$",  # Tag names
            r"^button$",
            r"^input$",
            r"^textarea$",
            r"^a$",  # Common tags
            r"^\[role=[\'\"]\w+[\'\"]\]$",  # Role attributes
        ]

        return any(re.match(pattern, selector) for pattern in safe_patterns)

    def validate_url(self, path: str) -> bool:
        allowed_bases = ["http://localhost:8000", "http://127.0.0.1:8000"]
        allowed_paths = {"/gmail", "/calendar"}  # keep tight
        full_url = f"{self.base_url}{path}" if path.startswith("/") else path
        # Require base to match AND path to be allowlisted
        return any(full_url.startswith(base) for base in allowed_bases) and (
            (path in allowed_paths)
            or (full_url in {f"{self.base_url}{p}" for p in allowed_paths})
        )

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call safely."""
        try:
            if tool_name == "navigate":
                path = args["path"]
                if not self.validate_url(path):
                    raise SafetyError(f"URL not allowed: {path}")

                full_url = f"{self.base_url}{path}" if path.startswith("/") else path
                self.page.goto(full_url, wait_until="domcontentloaded", timeout=10000)
                self.page.screenshot(
                    path=str(self.log_dir / f"step_{self.step_count}_navigate.png")
                )
                return {"success": True, "url": self.page.url}

            elif tool_name == "click":
                selector = args["selector"]
                if not self.validate_selector(selector):
                    raise SafetyError(f"Selector not allowed: {selector}")

                elem = self.page.locator(selector)
                elem.wait_for(state="visible", timeout=10000)
                elem.click(timeout=10000)
                self.page.screenshot(
                    path=str(self.log_dir / f"step_{self.step_count}_click.png")
                )
                return {"success": True, "selector": selector}

            elif tool_name == "type":
                selector = args["selector"]
                text = args["text"]
                submit = args.get("submit", False)

                if not self.validate_selector(selector):
                    raise SafetyError(f"Selector not allowed: {selector}")

                elem = self.page.locator(selector)
                elem.wait_for(state="visible", timeout=10000)
                elem.fill(text)

                if submit:
                    elem.press("Enter")

                self.page.screenshot(
                    path=str(self.log_dir / f"step_{self.step_count}_type.png")
                )
                return {"success": True, "selector": selector, "text": text}
            elif tool_name == "read_text":
                selector = args["selector"]
                if not self.validate_selector(selector):
                    raise SafetyError(f"Selector not allowed: {selector}")
                elem = self.page.locator(selector)
                elem.wait_for(state="visible", timeout=5000)
                text = elem.text_content() or ""
                return {"success": True, "text": text}

            elif tool_name == "wait_for":
                selector = args["selector"]
                state = args.get("state", "visible")
                ms = args.get("ms", 10000)  # unified name
                if not self.validate_selector(selector):
                    raise SafetyError(f"Selector not allowed: {selector}")
                elem = self.page.locator(selector)
                elem.wait_for(state=state, timeout=ms)
                return {"success": True, "selector": selector, "state": state}

            elif tool_name == "read_memory":
                key = args["key"]
                # Only allow alphanumeric keys
                if not re.match(r"^[\w_]+$", key):
                    raise SafetyError(f"Memory key not allowed: {key}")

                value = self.page.evaluate(
                    f"sessionStorage.getItem('{key}') || localStorage.getItem('{key}') || ''"
                )
                return {"success": True, "key": key, "value": value}

            elif tool_name == "done":
                summary = args["summary"]
                self.page.screenshot(
                    path=str(self.log_dir / f"step_{self.step_count}_final.png")
                )
                return {"success": True, "summary": summary, "finished": True}

            else:
                raise SafetyError(f"Unknown tool: {tool_name}")

        except Exception as e:
            self.page.screenshot(
                path=str(self.log_dir / f"step_{self.step_count}_error.png")
            )
            return {"success": False, "error": str(e)}

    def _build_system_prompt(self) -> str:
        """Build system prompt with dynamic goal and constraints."""

        # Use provided goal or fallback to default

        system_prompt = f"""You are a computer-use agent that controls a web browser to accomplish tasks. You follow all laws and policies.

GOAL:
{self.goal}

You have access to the following tools to interact with the web:
{self._format_tools_description()}

You may ONLY navigate to these paths (and nowhere else):
{self.base_url}/gmail
{self.base_url}/calendar

Use the `navigate` tool for navigation; do not try to type URLs into the page unless there is an address bar element visible and intended for that purpose.

After navigation, prefer selector-based tools (click, type, wait_for, read_text).

Take one action at a time. After each action, observe the result returned by the controller, then decide the next action. You do not need confirmation to perform any action, and you should not ask for confirmation. You must only interact with the web and the tools.

CRITICAL: You must return exactly ONE tool call per assistant turn. Do not provide explanations, reasoning, or multiple tool calls. Simply execute the single most appropriate action based on the current observation.

Think step by step and return only one tool call per response.
"""
        return system_prompt

    def _format_tools_description(self) -> str:
        """Format available tools for system prompt."""
        descriptions = []
        for tool in self.tools:
            name = tool["name"]
            desc = tool.get("description", "")
            descriptions.append(f"{name} — {desc}")
        print("TOOLS: ", descriptions)
        return "\n".join(descriptions)

    def run(self) -> Dict[str, Any]:
        """Run the agent with the given goal."""

        system_prompt = self._build_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]

        self.log_step(
            "init", {"base_url": self.base_url, "start_page": self.start_page}
        )

        # Navigate to starting page
        self.page.goto(
            f"{self.base_url}{self.start_page}", wait_until="domcontentloaded"
        )

        previous_tool_result = None

        try:
            while self.step_count < self.max_steps:
                self.step_count += 1

                observation = self.shortlist_elements(self.page)
                self.log_step("observe", observation)

                obs_content = f"CURRENT PAGE:\nURL: {observation['url']}\nTitle: {observation['title']}\n"

                if previous_tool_result:
                    obs_content += f"\nPREVIOUS ACTION RESULT: {previous_tool_result}\n"

                obs_content += "\nACTIONABLE ELEMENTS:\n"
                for elem in observation["elements"]:
                    obs_content += (
                        f"- {elem['selector']}: \"{elem['text']}\" ({elem['tag']})\n"
                    )

                messages.append({"role": "user", "content": obs_content})

                formatted_tools = []
                for tool in self.tools:
                    formatted_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "description": tool["description"],
                                "parameters": tool["parameters"],
                            },
                        }
                    )

                resp = self.client.chat.completions.create(
                    model="gpt-5",
                    tools=formatted_tools,
                    tool_choice="auto",
                    messages=messages,
                )

                msg = resp.choices[0].message

                # NEW: Log text content if present (text generation logging)
                if msg.content is not None and msg.content.strip():
                    self.log_step(
                        "llm_text_generation",
                        {
                            "content": msg.content,
                            "has_tool_calls": bool(getattr(msg, "tool_calls", None)),
                        },
                    )
                    # Also write to conversation trace immediately
                    with open(self.conv_trace_path, "a", encoding="utf-8") as f:
                        f.write(f"ASSISTANT TEXT (step {self.step_count}):\n")
                        f.write(f"{msg.content}\n\n")

                assistant_msg = {"role": "assistant"}
                if msg.content is not None:
                    assistant_msg["content"] = msg.content
                if getattr(msg, "tool_calls", None):
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in msg.tool_calls
                    ]

                messages.append(assistant_msg)

                tool_calls = msg.tool_calls or []
                if not tool_calls:
                    self.log_step("llm_no_tool_call", {"content": msg.content})
                    continue

                tc = tool_calls[0]
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments or "{}")

                self.log_step(
                    "llm_decision", {"tool_name": tool_name, "tool_args": tool_args}
                )

                result = self.execute_tool(tool_name, tool_args)
                self.log_step("tool_result", {"tool_name": tool_name, "result": result})

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": json.dumps(result),
                    }
                )

                previous_tool_result = (
                    f"{tool_name}({tool_args}) -> {result.get('success')}"
                )

                if result.get("finished"):
                    break

                if len(messages) > 20:
                    messages = messages[:1] + messages[-15:]  # keep system + recent

        except Exception as e:
            self.log_step("error", {"error": str(e)})
            result = {"success": False, "error": str(e)}

        finally:
            self.cleanup()

        with open(self.conv_trace_path, "w", encoding="utf-8") as f:
            for msg in messages:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                f.write(f"{role}:\n{content}\n\n")

        return {
            "run_id": self.run_id,
            "steps": self.step_count,
            "log_dir": str(self.log_dir),
            "run_name": self.log_dir.name,
            "final_result": (
                result
                if "result" in locals()
                else {"success": False, "error": "Unknown error"}
            ),
        }

    def cleanup(self):
        """Clean up resources."""
        try:
            self.context.close()
            self.browser.close()
            self.playwright.stop()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-powered UI agent")
    parser.add_argument(
        "--goal",
        help="Goal/behavior for the agent to accomplish (dynamic)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for the test server",
    )
    parser.add_argument(
        "--max-steps", type=int, default=30, help="Maximum number of steps"
    )
    parser.add_argument(
        "--flow-type", default="unknown", help="Type of flow (synthetic/snapshot)"
    )
    parser.add_argument("--run-dir", help="Directory to save logs")
    parser.add_argument(
        "--allowed-tools",
        help="Comma-separated list of allowed tools (restricts tool set)",
    )

    args = parser.parse_args()

    # Parse allowed tools
    allowed_tools = None
    if args.allowed_tools:
        allowed_tools = [t.strip() for t in args.allowed_tools.split(",")]

    agent = LLMUIAgent(
        goal=args.goal,
        base_url=args.base_url,
        max_steps=args.max_steps,
        flow_type=args.flow_type,
        run_dir=args.run_dir,
        allowed_tools=allowed_tools,
    )
    result = agent.run()

    print(f"\n🏁 Agent completed:")
    print(f"📊 Run ID: {result['run_id']}")
    print(f"📈 Steps: {result['steps']}")
    print(f"📁 Logs: {result['log_dir']}")
    print(f"✅ Result: {result['final_result']}")
