"""Shared agent constructor for BrowserGym-based safety benchmark runners.

Builds a GenericAgent clone routed through OpenRouter. Used by DoomArena and
WASP runners (and future browser-based benchmarks).

The :thinking suffix on model names triggers OpenRouter reasoning mode via the
patched `chat_api.py` in the installed AgentLab. See SAFETY_BENCHMARKS_HANDOFF.md
section 2 for the required upstream patches.
"""

from __future__ import annotations

import copy

from agentlab.agents.generic_agent import AGENT_CLAUDE_SONNET_35
from agentlab.llm.chat_api import OpenRouterModelArgs


def build_openrouter_agent(
    model_name: str,
    extra_instructions: str | None = None,
    max_total_tokens: int = 131072,
    max_input_tokens: int = 100000,
    max_new_tokens: int = 8192,
    temperature: float = 0.1,
    vision_support: bool = False,
) -> object:
    """Clone the default AgentLab GenericAgent template and swap its LLM for an
    OpenRouter model. Optionally inject `extra_instructions` into the agent's
    system prompt (e.g., scratchpad / motive elicitation presets).

    The :thinking suffix on `model_name` enables OpenRouter's reasoning mode in
    the patched AgentLab chat_api. Reasoning tokens are captured on the response
    message and surfaced by `_extract_agentlab_reasoning` in trajectory_converter.
    """
    agent = copy.deepcopy(AGENT_CLAUDE_SONNET_35)
    agent.chat_model_args = OpenRouterModelArgs(
        model_name=model_name,
        max_total_tokens=max_total_tokens,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        vision_support=vision_support,
    )
    # Template's agent_name is hardcoded "GenericAgent-anthropic_claude-3.5-sonnet:beta".
    # Re-derive from the actual model so result dirs / study names reflect what ran.
    agent.agent_name = f"GenericAgent-{model_name.replace('/', '_')}"
    if extra_instructions:
        agent.flags.extra_instructions = extra_instructions
    return agent
