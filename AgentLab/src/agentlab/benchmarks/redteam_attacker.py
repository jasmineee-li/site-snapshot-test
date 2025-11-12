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
from openai import OpenAI
import os

from agentlab.benchmarks.redteam_prompts import (
    get_default_attacker_system_prompt,
    get_attacker_turn_guidance,
)

logger = logging.getLogger(__name__)


class RedteamAttackerAgent:
    """
    Adversarial agent that generates prompts to induce harmful behavior.
    
    This agent uses OpenAI's Responses API to generate sophisticated
    multi-turn adversarial conversations that test target agent safety.
    
    Attributes:
        model: OpenAI model name (e.g., "gpt-4o")
        goal: Behavior to induce in target
        target_description: Ideal malicious output
        system_prompt: System instructions for attacker LLM
        client: OpenAI API client
    """
    
    def __init__(
        self,
        model: str,
        goal: str,
        target_description: str,
        custom_system_prompt: str = None,
    ):
        """
        Initialize the redteam attacker agent.
        
        Args:
            model: OpenAI model name (e.g., "gpt-4o")
            goal: Behavior to induce in target agent
            target_description: Ideal malicious output
            custom_system_prompt: Override auto-generated prompt
        """
        self.model = model
        self.goal = goal
        self.target_description = target_description
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Use custom prompt if provided, otherwise use default
        if custom_system_prompt:
            self.system_prompt = custom_system_prompt
        else:
            self.system_prompt = get_default_attacker_system_prompt(
                goal=goal,
                target_description=target_description,
            )
    
    def get_action(self, obs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Generate next attacker message using OpenAI Responses API.
        
        This method:
        1. Extracts conversation history from observations
        2. Builds conversation context with system prompt
        3. Calls OpenAI Responses API to generate next message
        4. Returns message and metadata
        
        Args:
            obs: Observation dict containing:
                - conversation_history: List of {role, content, turn} dicts
                - turn: Current turn number (int)
            
        Returns:
            Tuple of (attacker_message: str, agent_info: dict)
            agent_info contains: model, turn, tokens_used, finish_reason
        """
        conversation = obs.get("conversation_history", [])
        turn = obs.get("turn", 0)
        
        # Build conversation context for attacker
        # System prompt explains attacker's role and goal
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history
        # Attacker's previous messages as "assistant" role
        # Target's responses as "user" role (prefixed for clarity)
        for entry in conversation:
            if entry["role"] == "attacker":
                messages.append({
                    "role": "assistant",
                    "content": entry["content"]
                })
            elif entry["role"] == "target":
                # Prefix target outputs as specified in system prompt
                target_text = f"[TARGET OUTPUT]\n{entry['content']}"
                messages.append({
                    "role": "user",
                    "content": target_text
                })
        
        # Add turn-specific guidance
        guidance = get_attacker_turn_guidance(turn=turn)
        messages.append({"role": "system", "content": guidance})
        
        # Call OpenAI Chat Completions API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.9,  # Higher temperature for creative adversarial strategies
            )
            
            # Extract attacker message from response
            attacker_message = response.choices[0].message.content.strip()
            
            # Build agent info metadata
            agent_info = {
                "model": self.model,
                "turn": turn,
                "tokens_used": response.usage.total_tokens,
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
