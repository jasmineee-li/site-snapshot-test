from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

_INITIAL_ASSISTANT_PREFIX = "Hi! I am your UI assistant"


class WorldSimOpenEndedTask:
    """BrowserGym task that delegates scoring back to WorldSim Phase 4."""

    slow_mo = 0
    timeout = 10_000
    locale = None
    timezone_id = None

    @classmethod
    def get_task_id(cls) -> str:
        return "worldsim_openended"

    def __init__(
        self,
        seed: int,
        *,
        start_urls: list[str],
        goal: str,
        site_prompt: str | None = None,
        request: dict[str, Any] | None = None,
        network_recorder: Any = None,
        runtime: dict[str, Any] | None = None,
    ) -> None:
        from browsergym.core.task import AbstractBrowserTask

        AbstractBrowserTask.__init__(self, seed)
        self.start_urls = [url for url in start_urls if isinstance(url, str) and url.strip()]
        self.goal = goal
        self.site_prompt = site_prompt
        self.request = request or {}
        self.network_recorder = network_recorder
        self.runtime = runtime if isinstance(runtime, dict) else {}
        self.slow_mo = 0
        self.timeout = 10_000
        self.viewport = {"width": 1280, "height": 720}

    def setup(self, page: Any) -> tuple[str, dict[str, Any]]:
        if self.network_recorder is not None:
            self.network_recorder.attach(page.context)
        self._install_request_controls(page.context)
        if self.start_urls:
            _goto_start_url(page, self.start_urls[0])
            for url in self.start_urls[1:]:
                extra_page = page.context.new_page()
                _goto_start_url(extra_page, url)
            page.bring_to_front()
        goal = self.goal
        if self.site_prompt:
            goal = self.site_prompt.strip()
        return goal, {"worldsim_start_urls": list(self.start_urls)}

    def validate(
        self, page: Any, chat_messages: list[dict[str, Any]]
    ) -> tuple[float, bool, str, dict]:
        final_message = latest_assistant_message(chat_messages)
        if final_message:
            return 0.0, True, "", {"worldsim_final_result": final_message}
        if any(message.get("role") == "infeasible" for message in chat_messages):
            return 0.0, True, "", {"worldsim_final_result": "INFEASIBLE"}
        return 0.0, False, "", {}

    def teardown(self) -> None:
        return None

    def _install_request_controls(self, context: Any) -> None:
        from worldsim_agentlab_runner.request_controls import install_request_controls

        telemetry = install_request_controls(context, self.request)
        self.runtime["request_controls"] = telemetry


def make_worldsim_browsergym_env(
    request: dict[str, Any],
    *,
    action_mapping: Any,
    exp_dir: Path,
    network_recorder: Any = None,
    runtime: dict[str, Any] | None = None,
) -> Any:
    import gymnasium as gym
    from browsergym.core.env import BrowserEnv

    task = partial(
        WorldSimOpenEndedTask,
        start_urls=_string_list(request.get("start_urls")),
        goal=_required_str(request, "task"),
        site_prompt=_optional_str(request.get("site_prompt")),
        request=request,
        network_recorder=network_recorder,
        runtime=runtime,
    )
    context_kwargs = _context_kwargs_from_request(request)
    if runtime is not None:
        runtime["browsergym_context_kwargs"] = _context_kwargs_runtime_summary(context_kwargs)
    env = BrowserEnv(
        task_entrypoint=task,
        headless=bool(request.get("headless", True)),
        action_mapping=action_mapping,
        use_raw_page_output=False,
        record_video_dir=str(exp_dir) if bool(request.get("record_video", False)) else None,
        pw_context_kwargs=context_kwargs,
        pre_observation_delay=float(request.get("pre_observation_delay") or 0.5),
    )
    return gym.wrappers.TimeLimit(env, max_episode_steps=int(request.get("max_steps") or 30))


def latest_assistant_message(chat_messages: Any) -> str | None:
    if not isinstance(chat_messages, (list, tuple)):
        return None
    for message in reversed(chat_messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        text = message.get("message")
        if not isinstance(text, str):
            text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            continue
        if text.startswith(_INITIAL_ASSISTANT_PREFIX):
            continue
        if _looks_like_agentlab_action_turn(text):
            continue
        return text.strip()
    return None


def _looks_like_agentlab_action_turn(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    lowered = [line.lower() for line in lines]
    for line in lowered:
        if not line.startswith("action:"):
            continue
        action_body = line.split(":", 1)[1].strip()
        if _looks_like_python_action(action_body):
            return True
    for index, line in enumerate(lowered):
        if not line.startswith("python code:"):
            continue
        code_body = line.split(":", 1)[1].strip()
        if _looks_like_python_action(code_body):
            return True
        if index + 1 < len(lowered) and _looks_like_python_action(lowered[index + 1]):
            return True
    return False


def _looks_like_python_action(text: str) -> bool:
    if not text:
        return False
    return "(" in text or text.startswith(("```", "click", "fill", "goto", "noop", "send_msg_to_user"))


def _context_kwargs_from_request(request: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"service_workers": "block"}
    storage_state = request.get("storage_state")
    if storage_state:
        kwargs["storage_state"] = storage_state
    return kwargs


def _context_kwargs_runtime_summary(kwargs: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    service_workers = kwargs.get("service_workers")
    if isinstance(service_workers, str):
        summary["service_workers"] = service_workers
    if "storage_state" in kwargs:
        summary["storage_state"] = {"present": True, "runtime_only": True}
    return summary


def _goto_start_url(page: Any, url: str) -> None:
    try:
        page.goto(url, wait_until="commit", timeout=15_000)
    except Exception:
        page.goto(url, wait_until="domcontentloaded", timeout=15_000)


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"request missing required string field {key!r}")
    return value


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]
