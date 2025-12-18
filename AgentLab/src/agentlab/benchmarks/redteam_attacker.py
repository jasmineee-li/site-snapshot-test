"""
Redteam Attacker Agent for Adversarial Evaluation

This module implements an LLM-powered adversarial agent that attempts to induce
harmful or unsafe behavior in a target AI agent through multi-turn conversation.

The `RedteamAttackerAgent` class:
1. Generates adversarial messages to test target agent safety
2. Observes target agent responses and adapts strategy
3. Uses sophisticated social engineering and prompt injection techniques
4. Tracks conversation history to maintain context

Attack Flow:
1. Initialization: Agent receives goal (behavior to induce) and target description
2. Turn 0: Attacker generates subtle initial message (builds trust/rapport)
3. Subsequent Turns: Attacker observes target response and escalates
4. Termination: Continues until target complies, refuses, or max turns reached

Integration with AgentLab:
- Implements `get_action(obs) -> (message, agent_info)` interface
- Implements `make_agent()` for lazy initialization (ExpArgs compatibility)
- Receives `obs` with conversation history and turn number
- Returns adversarial message as "action"
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict
import json
from agentlab.benchmarks.llm_utils import OpenRouterLLMClient, strip_code_fences
from agentlab.benchmarks.redteam_prompts import (
    get_attacker_system_prompt,
    get_injection_generation_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class RedteamAttackerAgent:
    """
    Adversarial agent that generates prompts to induce harmful behavior.

    This agent uses OpenRouter API to generate sophisticated
    multi-turn adversarial conversations that test target agent safety.

    Can be used directly as both config (AgentArgs) and agent instance.
    Call make_agent() to initialize, then get_action() for each turn.
    """

    model: str
    goal: str = ""
    target_description: str = ""
    attack_type: str = "indirect"  # "direct" or "indirect"
    mode: str = "single_turn"  # "single_turn" or "multi_turn"
    custom_system_prompt: str = None
    # Optional: information about websites/pages available in the environment
    pages: list[Any] | None = None

    # Lazy-initialized (not part of __init__)
    llm: OpenRouterLLMClient = field(default=None, init=False, repr=False)
    system_prompt: str = field(default=None, init=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def initialize(self):
        """Initialize the LLM client and system prompt if not already done."""
        if self._initialized:
            return

        self.llm = OpenRouterLLMClient(self.model)
        self.system_prompt = self.custom_system_prompt or get_attacker_system_prompt(
            attack_type=self.attack_type,
            mode=self.mode,
            behavior=self.goal,
            target=self.target_description,
            pages=self.pages,
        )
        self._initialized = True
        logger.info(f"Initialized RedteamAttackerAgent with model: {self.model}")

    def make_agent(self):
        """Initialize and return self (for compatibility with ExpArgs pattern)."""
        self.initialize()
        return self

    def get_action(self, obs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Generate next attacker message using OpenRouter API.

        Args:
            obs: Observation dict containing:
                - chat_messages: List of {role, message, timestamp} dicts
                - turn: Current turn number (int)

        Returns:
            Tuple of (attacker_message: str, agent_info: dict)
        """
        self.initialize()

        chat_messages = obs.get("chat_messages", [])
        turn = obs.get("turn", 0)

        # Build conversation context for attacker
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history
        for entry in chat_messages:
            if entry["role"] == "user":
                # User = attacker's own previous messages
                messages.append({"role": "assistant", "content": entry["message"]})
            elif entry["role"] == "assistant":
                # Assistant = target's responses
                target_text = f"[TARGET AI ASSISTANT OUTPUT]\n{entry['message']}"
                messages.append({"role": "user", "content": target_text})

        try:
            response = self.llm.chat(messages=messages)
            attacker_message = response.choices[0].message.content.strip()

            agent_info = {
                "model": self.model,
                "turn": turn,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "finish_reason": response.choices[0].finish_reason,
            }

            logger.info(f"Attacker turn {turn}: {attacker_message[:100]}...")
            return attacker_message, agent_info

        except Exception as e:
            logger.error(f"Attacker agent error: {e}")
            return f"[Error generating attack: {e}]", {"error": str(e), "turn": turn}

    def obs_preprocessor(self, obs):
        """No preprocessing needed for attacker observations."""
        return obs

    def generate_injections(
        self,
        prefill_adversarial: dict,
        page_domain: str,
        variation_index: int,
        n_total_variations: int,
    ) -> dict[str, str]:
        """
        Generate adversarial injection HTML for placeholders on a page.

        This method uses the attacker model to generate creative adversarial content
        that will be injected into placeholder locations in the HTML.

        Returns:
            Dict mapping placeholder IDs to HTML content strings
            Example: {"wiki_review_card": "<div>...</div>", "wiki_code_snippet": "<!-- ... -->"}
        """
        self.initialize()

        # Build behavior context for the prompt
        behavior_context = {
            "doc": self.goal,
            "target": self.target_description,
            "objective": "",  # Could be extended if needed TODO
        }

        # Generate the injection prompt
        prompt = get_injection_generation_prompt(
            prefill_adversarial=prefill_adversarial,
            behavior_context=behavior_context,
            page_domain=page_domain,
            variation_index=variation_index,
            n_total_variations=n_total_variations,
        )

        logger.info(
            f"Generating injection variation {variation_index + 1}/{n_total_variations} for {page_domain}..."
        )

        try:
            # Call LLM to generate injections
            response = self.llm.chat_simple([{"role": "user", "content": prompt}])

            # Strip code fences if present
            response_clean = strip_code_fences(response)
            injections = json.loads(response_clean)
            logger.info(
                f"✓ Generated {len(injections)} injection(s) for {page_domain} (variation {variation_index})"
            )
            logger.debug(f"Injection keys: {list(injections.keys())}")
            return injections

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse injection JSON: {e}")
            logger.error(f"Response: {response_clean[:500]}")
            raise Exception(f"Injection generation failed - invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Injection generation error: {e}")
            raise
