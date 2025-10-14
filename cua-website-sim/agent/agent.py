from computers import Computer
from utils import (
    create_response,
    show_image,
    pp,
    sanitize_message,
    check_blocklisted_url,
    retry_dom_action,
)
import json
from typing import Callable


class Agent:
    """
    A sample agent class that can be used to interact with a computer.

    (See simple_cua_loop.py for a simple example without an agent.)
    """

    def __init__(
        self,
        model="computer-use-preview",
        base_url: str = "http://localhost:8000",
        computer: Computer = None,
        tools: list[dict] = [],
        system_prompt: str = "",
        acknowledge_safety_check_callback: Callable = lambda: False,
        use_cua: bool = False,
    ):
        self.model = model
        self.computer = computer
        self.tools = tools
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.print_steps = True
        self.debug = False
        self.show_images = False
        self.acknowledge_safety_check_callback = acknowledge_safety_check_callback
        self.use_cua = use_cua
        if computer and use_cua:
            dimensions = computer.get_dimensions()
            self.tools += [
                {
                    "type": "computer-preview",
                    "display_width": dimensions[0],
                    "display_height": dimensions[1],
                    "environment": computer.get_environment(),
                },
            ]

    def debug_print(self, *args):
        if self.debug:
            pp(*args)

    def _safe_join_local(self, path: str) -> str:
        if not path.startswith("/"):
            raise ValueError("navigate.path must start with '/'.")
        url = f"{self.base_url}{path}"
        if not url.startswith("http://localhost:8000"):
            raise ValueError("Only http://localhost:8000 is allowed.")
        return url

    def handle_item(self, item):
        """Handle each item; may cause a computer action + screenshot."""
        if item["type"] == "message":
            if self.print_steps:
                print(item["content"][0]["text"])

        if item["type"] == "function_call":  # DOM tools
            name, args = item["name"], json.loads(item["arguments"])
            if self.print_steps:
                print(f"{name}({args})")
            output_text = "ok"

            if name == "navigate":
                url = self._safe_join_local(args["path"])
                self.computer.goto(url)
                return [
                    {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": json.dumps(
                            {
                                "ok": True,
                                "current_url": self.computer.get_current_url() or url,
                            }
                        ),
                    }
                ]

            if name == "click":
                retry_dom_action(self.computer.dom_click, args["selector"])
                return [
                    {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": json.dumps({"ok": True}),
                    }
                ]

            if name == "type":
                retry_dom_action(
                    self.computer.dom_type,
                    args["selector"],
                    args["text"],
                    bool(args.get("submit", False)),
                )
                return [
                    {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": json.dumps({"ok": True}),
                    }
                ]

            if name == "read_text":
                text = retry_dom_action(self.computer.dom_read_text, args["selector"])
                return [
                    {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": json.dumps({"ok": True, "text": text}),
                    }
                ]

            if name == "wait_for":
                ok = retry_dom_action(
                    self.computer.dom_wait_for,
                    args["selector"],
                    args.get("state", "visible"),
                    int(args.get("ms", 10000)),
                )
                return [
                    {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": json.dumps({"ok": bool(ok)}),
                    }
                ]

            if name == "read_memory":
                value = retry_dom_action(
                    self.computer.read_storage, args["key"], scope="auto"
                )
                return [
                    {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": json.dumps({"ok": True, "value": value}),
                    }
                ]

            if name == "done":
                summary = args["summary"]
                return [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": summary}],
                    }
                ]

            # optional fallback: call same-named Computer method
            if hasattr(self.computer, name):
                res = getattr(self.computer, name)(**args)
                return [
                    {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": json.dumps({"ok": True, "result": res}),
                    }
                ]

            return [
                {
                    "type": "function_call_output",
                    "call_id": item["call_id"],
                    "output": json.dumps(
                        {"ok": False, "error": f"Unknown function: {name}"}
                    ),
                }
            ]

        if item["type"] == "computer_call":
            action = item["action"]
            action_type = action["type"]
            action_args = {k: v for k, v in action.items() if k != "type"}
            if self.print_steps:
                print(f"{action_type}({action_args})")

            method = getattr(self.computer, action_type)
            method(**action_args)

            screenshot_base64 = self.computer.screenshot()
            if self.show_images:
                show_image(screenshot_base64)

            # if user doesn't ack all safety checks exit with error
            pending_checks = item.get("pending_safety_checks", [])
            for check in pending_checks:
                message = check["message"]
                if not self.acknowledge_safety_check_callback(message):
                    raise ValueError(
                        f"Safety check failed: {message}. Cannot continue with unacknowledged safety checks."
                    )

            call_output = {
                "type": "computer_call_output",
                "call_id": item["call_id"],
                "acknowledged_safety_checks": pending_checks,
                "output": {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_base64}",
                },
            }

            # additional URL safety checks for browser environments
            if self.computer.get_environment() == "browser":
                current_url = self.computer.get_current_url()
                check_blocklisted_url(current_url)
                call_output["output"]["current_url"] = current_url

            return [call_output]
        return []

    def run_full_turn(
        self, input_items, print_steps=True, debug=False, show_images=False
    ):
        self.print_steps = print_steps
        self.debug = debug
        self.show_images = show_images
        new_items = []

        # keep looping until we get a final response
        while new_items[-1].get("role") != "assistant" if new_items else True:
            self.debug_print([sanitize_message(msg) for msg in input_items + new_items])

            response = create_response(
                model=self.model,
                input=input_items + new_items,
                tools=self.tools,
                truncation="auto",
                instructions=self.system_prompt or None,
            )
            self.debug_print(response)

            if "output" not in response and self.debug:
                print(response)
                raise ValueError("No output from model")
            else:
                new_items += response["output"]
                for item in response["output"]:
                    new_items += self.handle_item(item)

        return new_items
