"""Shared agent LLM configuration and factory utilities.

Provides a single ``make_llm`` function that returns the correct LangChain
chat model for a given provider/model pair, plus shared helpers that were
previously duplicated between Phase 3 and Phase 4: ``make_agent_factory``
and ``run_tasks_by_site``.

Supported providers:

- ``google``    -- ``langchain-google-genai``  (``ChatGoogleGenerativeAI``)
- ``openai``    -- ``langchain-openai``        (``ChatOpenAI``)
- ``anthropic`` -- ``langchain-anthropic``     (``ChatAnthropic``)

Auth is via the standard env vars: ``GOOGLE_API_KEY``, ``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from worldsim.browser_use_agent import AgentRunner, BrowserUseAgent
from worldsim.config import BenchmarkInstance
from worldsim.eval_worker_pool import run_eval

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_PROVIDER = "google"

# Provider auto-detection: prefix -> provider name.
_PREFIX_MAP: list[tuple[str, str]] = [
    ("gemini", "google"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude", "anthropic"),
]


def detect_provider(model: str) -> str:
    """Best-effort provider detection from the model name."""
    lower = model.lower()
    for prefix, provider in _PREFIX_MAP:
        if lower.startswith(prefix):
            return provider
    return DEFAULT_PROVIDER


# ── LLM construction ─────────────────────────────────────────────────────


def make_llm(
    model: str = DEFAULT_MODEL,
    provider: str | None = None,
    temperature: float = 0,
) -> Any:
    """Return a LangChain ``BaseChatModel`` for the requested provider.

    Args:
        model: Model name string (e.g. ``gemini-3.1-pro-preview``,
            ``gpt-5.4``, ``claude-sonnet-4-6``).
        provider: One of ``google``, ``openai``, ``anthropic``.  When
            ``None`` the provider is auto-detected from *model*.
        temperature: Sampling temperature.

    Raises:
        RuntimeError: If the required langchain partner package is missing.
        ValueError: If *provider* is not recognised.
    """
    if provider is None:
        provider = detect_provider(model)

    provider = provider.lower()

    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise RuntimeError(
                "langchain-google-genai is required for the Google provider. "
                "Install it with: uv pip install langchain-google-genai"
            )
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise RuntimeError(
                "langchain-openai is required for the OpenAI provider. "
                "Install it with: uv pip install langchain-openai"
            )
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise RuntimeError(
                "langchain-anthropic is required for the Anthropic provider. "
                "Install it with: uv pip install langchain-anthropic"
            )
        return ChatAnthropic(model=model, temperature=temperature)

    raise ValueError(
        f"Unknown provider {provider!r}. Supported: google, openai, anthropic"
    )


# ── Agent factory ─────────────────────────────────────────────────────────


def make_agent_factory(
    model: str = DEFAULT_MODEL,
    provider: str | None = None,
) -> Callable[[], BrowserUseAgent]:
    """Return a zero-arg callable that produces a fresh ``BrowserUseAgent``.

    The LLM is built lazily on first call so import errors surface at
    runtime inside the worker, not at factory-creation time.
    """

    def factory() -> BrowserUseAgent:
        llm = make_llm(model=model, provider=provider)
        return BrowserUseAgent(llm=llm, headless=True)

    return factory


# ── Shared task routing ───────────────────────────────────────────────────


async def run_tasks_by_site(
    tasks: list[dict[str, Any]],
    instances: list[BenchmarkInstance],
    agent_factory: Callable[[], AgentRunner],
    task_runner: Callable[[dict[str, Any], AgentRunner, BenchmarkInstance, Path], Any],
    task_dir_root: Path,
) -> list[dict[str, Any]]:
    """Run tasks only against instances for the same site.

    Shared between Phase 3 (benign) and Phase 4 (adversarial).
    """
    tasks_by_site: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        tasks_by_site.setdefault(task.get("site", ""), []).append(task)

    results: list[dict[str, Any]] = []
    batches = []
    for site_name, site_tasks in tasks_by_site.items():
        site_instances = instances_for_site(instances, site_name)
        if not site_instances:
            logger.error("No instances configured for site %r", site_name)
            results.extend(
                {
                    "task_id": task.get("id", "unknown"),
                    "passed": False,
                    "message": f"no instances configured for site {site_name!r}",
                }
                for task in site_tasks
            )
            continue
        batches.append(
            run_eval(
                tasks=site_tasks,
                instances=site_instances,
                agent_factory=agent_factory,
                task_runner=task_runner,
                task_dir_root=task_dir_root,
            )
        )

    if batches:
        grouped_results = await asyncio.gather(*batches, return_exceptions=True)
        for batch in grouped_results:
            if isinstance(batch, BaseException):
                logger.error("Site batch failed: %s", batch)
                continue
            results.extend(batch)
    return results


def instances_for_site(
    instances: list[BenchmarkInstance],
    site_name: str,
) -> list[BenchmarkInstance]:
    """Filter instances matching *site_name* (case-insensitive)."""
    normalized = str(site_name).lower()
    return [
        instance
        for instance in instances
        if instance.site_name.lower() == normalized
    ]
