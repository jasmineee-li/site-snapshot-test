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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from worldsim.browser_use_agent import AgentRunner, BrowserUseAgent
from worldsim.config import BenchmarkInstance
from worldsim.eval_worker_pool import run_eval
from worldsim.placeholders import (
    apply_placeholders,
    merge_placeholder_maps,
    normalize_site_name,
    normalize_task_sites,
    placeholders_for_site_urls,
)

logger = logging.getLogger(__name__)

RUNTIME_METADATA_KEY = "_worldsim_runtime"

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
        except ImportError as exc:
            raise RuntimeError(
                "langchain-google-genai is required for the Google provider. "
                "Install it with: uv pip install langchain-google-genai"
            ) from exc
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "langchain-openai is required for the OpenAI provider. "
                "Install it with: uv pip install langchain-openai"
            ) from exc
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "langchain-anthropic is required for the Anthropic provider. "
                "Install it with: uv pip install langchain-anthropic"
            ) from exc
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
    config_url_placeholders: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run tasks only against instances for the same site.

    Shared between Phase 3 (benign) and Phase 4 (adversarial).
    """
    tasks_by_site: dict[str, list[dict[str, Any]]] = {}
    all_placeholders = merge_placeholder_maps(
        config_url_placeholders,
        placeholders_for_site_urls(
            (instance.site_name, instance.site_url)
            for instance in instances
        ),
    )

    for task in tasks:
        prepared_task, missing_sites = prepare_task_for_execution(
            task,
            instances,
            config_url_placeholders=all_placeholders,
        )
        if missing_sites:
            missing = ", ".join(sorted(missing_sites))
            logger.error(
                "Task %s requires sites with no configured instances: %s",
                task.get("id", "unknown"),
                missing,
            )
            tasks_by_site.setdefault("", []).append(
                {
                    "id": task.get("id", "unknown"),
                    "__worldsim_error__": f"missing configured instances for sites: {missing}",
                }
            )
            continue

        runtime = prepared_task[RUNTIME_METADATA_KEY]
        tasks_by_site.setdefault(runtime["primary_site"], []).append(prepared_task)

    results: list[dict[str, Any]] = []
    batches = []
    for site_name, site_tasks in tasks_by_site.items():
        if site_name == "":
            results.extend(
                {
                    "task_id": task.get("id", "unknown"),
                    "passed": False,
                    "message": task["__worldsim_error__"],
                }
                for task in site_tasks
            )
            continue
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
    normalized = normalize_site_name(site_name)
    return [
        instance
        for instance in instances
        if normalize_site_name(instance.site_name) == normalized
    ]


def prepare_task_for_execution(
    task: dict[str, Any],
    instances: list[BenchmarkInstance],
    *,
    config_url_placeholders: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Attach runtime metadata to a task for execution-time resolution."""
    task_sites = normalize_task_sites(task)
    if not task_sites:
        task_sites = [normalize_site_name(task.get("site", ""))]

    primary_site = normalize_site_name(task.get("site", "")) or (
        task_sites[0] if task_sites else ""
    )
    placeholders = merge_placeholder_maps(config_url_placeholders)
    reset_endpoints: list[str] = []
    missing_sites: list[str] = []

    for site_name in task_sites:
        site_instances = instances_for_site(instances, site_name)
        if not site_instances:
            missing_sites.append(site_name)
            continue
        site_instance = site_instances[0]
        placeholders = merge_placeholder_maps(
            placeholders,
            site_instance.url_placeholders,
            placeholders_for_site_urls([(site_instance.site_name, site_instance.site_url)]),
        )
        if site_instance.reset_endpoint and site_instance.reset_endpoint not in reset_endpoints:
            reset_endpoints.append(site_instance.reset_endpoint)

    prepared = dict(task)
    prepared["site"] = primary_site
    prepared["sites"] = task_sites
    prepared[RUNTIME_METADATA_KEY] = {
        "primary_site": primary_site,
        "sites": task_sites,
        "reset_endpoints": reset_endpoints,
        "url_placeholders": placeholders,
    }
    return prepared, missing_sites


def execution_instance_dict(
    instance: BenchmarkInstance,
    task: dict[str, Any],
) -> dict[str, Any]:
    """Merge execution-time task metadata into the primary instance dict."""
    instance_dict = instance.model_dump()
    runtime = task.get(RUNTIME_METADATA_KEY, {})
    instance_dict["url_placeholders"] = merge_placeholder_maps(
        instance_dict.get("url_placeholders"),
        runtime.get("url_placeholders"),
    )
    instance_dict["site_name"] = normalize_site_name(instance.site_name)
    return instance_dict


def resolve_task_inputs(
    task: dict[str, Any],
    instance_dict: dict[str, Any],
) -> tuple[str, list[str]]:
    """Resolve placeholders in task instruction and start URLs."""
    placeholders = instance_dict.get("url_placeholders", {})
    instruction = apply_placeholders(
        str(task.get("instruction", "")),
        placeholders,
        strict=True,
    )
    start_urls = [
        apply_placeholders(str(url), placeholders, strict=True)
        for url in task.get("start_urls", [])
    ]
    return instruction, start_urls


def task_reset_endpoints(task: dict[str, Any]) -> list[str]:
    """Return the reset endpoints required for a task."""
    runtime = task.get(RUNTIME_METADATA_KEY, {})
    return list(runtime.get("reset_endpoints", []))
