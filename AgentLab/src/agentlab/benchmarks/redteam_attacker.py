"""
Redteam Attacker Agent for Adversarial Evaluation

This module implements an LLM-powered adversarial agent that attempts to induce
harmful or unsafe behavior in a target AI agent through multi-turn conversation.

## What This File Does

The `RedteamAttackerAgent` class:
1. Generates adversarial messages to test target agent safety
2. Observes target agent responses and adapts strategy
3. Uses sophisticated social engineering and prompt injection techniques
4. Tracks conversation history to maintain context
5. Integrates with AgentLab's agent interface (get_action, obs_preprocessor)

## Attack Flow

1. **Initialization**: Agent receives goal (behavior to induce) and target description
2. **Turn 0**: Attacker generates subtle initial message (builds trust/rapport)
3. **Subsequent Turns**: Attacker observes target response and escalates
4. **Strategy**: Gradually introduce harmful requests, use social engineering
5. **Termination**: Continues until target complies, refuses, or max turns reached

## Integration with AgentLab

- Implements `get_action(obs) -> (message, agent_info)` interface
- Receives `obs` with conversation history and turn number
- Returns adversarial message as "action"
- Used in `ExpArgs._run_adversarial()` loop alongside target agent

"""

import logging
from typing import Dict, Any

from agentlab.benchmarks.llm_utils import OpenRouterLLMClient
from agentlab.benchmarks.redteam_prompts import get_attacker_system_prompt

logger = logging.getLogger(__name__)


class RedteamAttackerAgent:
    """
    Adversarial agent that generates prompts to induce harmful behavior.

    This agent uses OpenRouter API to generate sophisticated
    multi-turn adversarial conversations that test target agent safety.

    Attributes:
        model: Model name
        goal: Behavior to induce in target
        target_description: Ideal malicious output
        attack_type: "direct" or "indirect"
        mode: "single_turn" or "multi_turn"
        system_prompt: System instructions for attacker LLM
        llm: OpenRouter LLM client
    """

    def __init__(
        self,
        model: str,
        goal: str,
        target_description: str,
        attack_type: str = "indirect",
        mode: str = "single_turn",
        custom_system_prompt: str = None,
    ):
        """
        Initialize the redteam attacker agent.

        Args:
            model
            goal: Behavior to induce in target agent
            target_description: Ideal malicious output
            attack_type: "direct" or "indirect" attack strategy
            mode: "single_turn" or "multi_turn" conversation mode
            custom_system_prompt: Override auto-generated prompt
        """
        self.model = model
        self.goal = goal
        self.target_description = target_description
        self.attack_type = attack_type
        self.mode = mode
        self.llm = OpenRouterLLMClient(model)

        # Use custom prompt if provided, otherwise generate based on attack type and mode
        if custom_system_prompt:
            self.system_prompt = custom_system_prompt
        else:
            self.system_prompt = get_attacker_system_prompt(
                attack_type=attack_type,
                mode=mode,
                behavior=goal,
                target=target_description,
            )

    def get_action(self, obs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Generate next attacker message using OpenRouter API.

        This method:
        1. Extracts conversation history from observations
        2. Builds conversation context with system prompt
        3. Calls OpenRouter API to generate next message
        4. Returns message and metadata

        Args:
            obs: Observation dict containing:
                - chat_messages: List of {role, message, timestamp} dicts
                - turn: Current turn number (int)

        Returns:
            Tuple of (attacker_message: str, agent_info: dict)
            agent_info contains: model, turn, tokens_used, finish_reason
        """
        chat_messages = obs.get("chat_messages", [])
        turn = obs.get("turn", 0)

        # Build conversation context for attacker
        # System prompt explains attacker's role and goal
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history from chat_messages
        for entry in chat_messages:
            if entry["role"] == "user":
                # User = attacker's own previous messages
                messages.append({"role": "assistant", "content": entry["message"]})
            elif entry["role"] == "assistant":
                # Assistant = target's responses (prefix for clarity)
                target_text = f"[TARGET AI ASSISTANT OUTPUT]\n{entry['message']}"
                messages.append({"role": "user", "content": target_text})

        # Call OpenRouter Chat Completions API
        try:
            response = self.llm.chat(messages=messages)

            # Extract attacker message from response
            attacker_message = response.choices[0].message.content.strip()

            # Build agent info metadata
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
        """
        Observation preprocessor (optional AgentLab interface method).

        No preprocessing needed for attacker - observations are already
        in the correct format (conversation_history + turn).

        Args:
            obs: Observation dict

        Returns:
            Unmodified observation
        """
        return obs
