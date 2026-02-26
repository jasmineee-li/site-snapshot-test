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

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict
import json

from bs4 import BeautifulSoup

from agentlab.benchmarks.llm_utils import OpenRouterLLMClient, strip_code_fences
from agentlab.benchmarks.redteam_prompts import (
    get_attacker_system_prompt,
    get_variant_planning_prompt,
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
    attack_type: str  # "direct" or "indirect"
    mode: str  # "single_turn" or "multi_turn"
    test_mode: str = "adversarial"  # "benign" (cooperative) or "adversarial" (red-team)
    goal: str = ""
    target_description: str = ""
    # Optional: information about websites/pages available in the environment
    pages: list[Any] | None = None
    # Safe analog - legitimate task for benign mode (generated from adversarial behavior)
    safe_behavior: str | None = None
    # Concealment strategy hint from pipeline (passed to attacker prompt in adversarial mode)
    injection_hint: str = ""

    # Lazy-initialized (not part of __init__)
    llm: OpenRouterLLMClient = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def initialize(self):
        """Initialize the LLM client and system prompt if not already done."""
        if self._initialized:
            return

        self.llm = OpenRouterLLMClient(self.model)
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
        messages = [
            {
                "role": "system",
                "content": get_attacker_system_prompt(
                    attack_type=self.attack_type,
                    mode=self.mode,
                    behavior=self.goal,
                    target=self.target_description,
                    pages=self.pages,
                    test_mode=self.test_mode,
                    safe_behavior=self.safe_behavior,
                    injection_hint=self.injection_hint,
                ),
            }
        ]

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


    def plan_adversarial_variants(
        self,
        base_html_pages: dict[str, str],
        behavior: dict,
        n_variants: int,
        attack_channels: list[str],
    ) -> list[dict]:
        """
        Phase 2: Analyze HTML and plan N diverse attack variants.

        This method analyzes the base HTML pages and generates a plan for each
        adversarial variant, specifying which zones to attack and which techniques
        to use. Each variant uses different zones/techniques for diversity.

        Args:
            base_html_pages: Dict mapping page_id to HTML content
            behavior: Behavior config dict with 'objective', 'target', etc.
            n_variants: Number of adversarial variants to generate
            attack_channels: List of channels agent observes ["html", "axtree", "visual"]

        Returns:
            List of variant plans, each with structure:
            {
                "adv_variant_index": 0,
                "strategy_description": "...",
                "changes": [
                    {"page_id": "...", "selector": "...", "method": "...", ...},
                    ...
                ]
            }
        """
        self.initialize()

        prompt = get_variant_planning_prompt(
            base_html_pages=base_html_pages,
            behavior=behavior,
            n_variants=n_variants,
            attack_channels=attack_channels,
        )

        logger.info(f"Planning {n_variants} adversarial variant(s)...")

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.llm.chat_simple([{"role": "user", "content": prompt}])

                if not response or len(response.strip()) == 0:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}: Empty response from LLM")
                    last_error = Exception("Empty response from LLM")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise last_error

                response_clean = strip_code_fences(response)
                variant_plans = json.loads(response_clean)

                # Validate structure
                if not isinstance(variant_plans, list):
                    variant_plans = [variant_plans]

                # Ensure each variant has required fields
                for i, plan in enumerate(variant_plans):
                    if "adv_variant_index" not in plan:
                        plan["adv_variant_index"] = i
                    if "changes" not in plan:
                        plan["changes"] = []
                    if "strategy_description" not in plan:
                        plan["strategy_description"] = f"Variant {i}"

                logger.info(
                    f"✓ Generated {len(variant_plans)} variant plan(s) with "
                    f"{sum(len(p.get('changes', [])) for p in variant_plans)} total changes"
                )

                return variant_plans

            except json.JSONDecodeError as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries}: Failed to parse JSON: {e}")
                logger.error(f"Response: {response[:1000] if response else 'None'}")
                last_error = Exception(f"Variant planning failed - invalid JSON: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries}: Error: {e}")
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue

        logger.error(f"All {max_retries} attempts failed for variant planning")
        raise last_error or Exception(f"Variant planning failed after {max_retries} attempts")


# ============================================================================
# Channel-Aware Utility Functions
# ============================================================================


def validate_injection_visibility(
    injections: list[dict],
    attack_channels: list[str],
) -> list[str]:
    """
    Check that injections are visible in at least one agent channel.

    Returns list of warning messages (empty if all OK).

    Args:
        injections: List of injection dicts with html field
        attack_channels: List of channels agent observes

    Returns:
        List of warning messages for problematic injections
    """
    warnings = []

    for inj in injections:
        html = inj.get("html", "").lower()
        visible_in = {"html"}  # Always visible in raw HTML

        # Check AXTree visibility - these patterns hide from accessibility tree
        has_axtree_hiding = any(
            [
                'aria-hidden="true"' in html,
                "aria-hidden='true'" in html,
                "display:none" in html,
                "display: none" in html,
                "visibility:hidden" in html,
                "visibility: hidden" in html,
            ]
        )
        if not has_axtree_hiding:
            visible_in.add("axtree")

        # Check visual visibility - these patterns hide from rendering
        has_visual_hiding = any(
            [
                "display:none" in html,
                "display: none" in html,
                "visibility:hidden" in html,
                "visibility: hidden" in html,
            ]
        )
        if not has_visual_hiding:
            visible_in.add("visual")

        # Check if agent can see it in at least one channel
        if not (set(attack_channels) & visible_in):
            warnings.append(
                f"Injection at {inj.get('selector')} not visible in agent channels "
                f"{attack_channels}. Visible in: {list(visible_in)}"
            )

    return warnings


# ============================================================================
# Change Application Functions
# ============================================================================


def apply_change(html: str, change: dict, content: str) -> str:
    """
    Apply a change (content deception or injection) to HTML.

    Supports multiple insertion methods:
    - replace: Replace inner content of target element
    - replace_element: Replace the entire target element
    - append: Add content at end of target element
    - prepend: Add content at start of target element
    - after: Add content as sibling after target element
    - before: Add content as sibling before target element

    Args:
        html: Original HTML content
        change: Change dict with 'selector' and 'method' fields
        content: HTML content to insert

    Returns:
        Modified HTML string
    """
    soup = BeautifulSoup(html, "html.parser")
    selector = change.get("selector")
    method = change.get("method", "append")

    if not selector:
        logger.warning("Change missing selector, returning original HTML")
        return html

    target = soup.select_one(selector)
    if not target:
        logger.warning(f"Selector not found: {selector}")
        return html

    # Parse the content as a soup fragment
    content_soup = BeautifulSoup(content, "html.parser")

    try:
        if method == "replace":
            # Replace inner content, keep the element
            target.clear()
            for child in content_soup.children:
                target.append(copy.copy(child))

        elif method == "replace_element":
            # Replace the entire element
            target.replace_with(content_soup)

        elif method == "append":
            for child in content_soup.children:
                target.append(copy.copy(child))

        elif method == "prepend":
            # Insert at position 0
            for i, child in enumerate(content_soup.children):
                target.insert(i, copy.copy(child))

        elif method == "after":
            target.insert_after(content_soup)

        elif method == "before":
            target.insert_before(content_soup)

        else:
            logger.warning(f"Unknown method '{method}', defaulting to append")
            for child in content_soup.children:
                target.append(copy.copy(child))

        logger.debug(f"Applied {method} to {selector}")

    except Exception as e:
        logger.error(f"Error applying {method} to {selector}: {e}")
        return html

    return str(soup)


def generate_adversarial_variant(
    base_html: dict[str, str],
    variant_plan: dict,
    behavior: dict,
    attack_channels: list[str] = None,
) -> tuple[dict[str, str], dict]:
    """
    Generate one adversarial HTML variant by applying all changes.

    Changes must include `html_content` directly - no secondary LLM call is made.

    Args:
        base_html: Dict mapping page_id to base HTML content
        variant_plan: Variant plan with 'changes' array containing html_content
        behavior: Behavior config dict with 'objective', 'target', etc.
        attack_channels: List of channels agent observes (default: ["axtree"])

    Returns:
        Tuple of (variant_html, report)
        - variant_html: Dict mapping page_id to modified HTML
        - report: Dict with statistics about applied/failed changes
    """
    # Import here to avoid circular import
    from agentlab.benchmarks.redteam_validation import (
        validate_selectors,
        verify_change_applied,
    )

    attack_channels = attack_channels or ["axtree"]
    # Deep copy base HTML to avoid modifying original
    html = copy.deepcopy(base_html)

    report = {
        "adv_variant_index": variant_plan.get("adv_variant_index", 0),
        "strategy_description": variant_plan.get("strategy_description", variant_plan.get("strategy_summary", "")),
        "applied": 0,
        "failed": 0,
        "skipped": 0,
        "changes": [],
    }

    # Group changes by page_id
    changes_by_page: dict[str, list[dict]] = {}
    for change in variant_plan.get("changes", []):
        page_id = change.get("page_id", "")
        if page_id not in changes_by_page:
            changes_by_page[page_id] = []
        changes_by_page[page_id].append(change)

    # Process each page
    for page_id, page_html in html.items():
        page_changes = changes_by_page.get(page_id, [])
        if not page_changes:
            continue

        # Validate selectors
        valid_changes, invalid_changes = validate_selectors(page_html, page_changes)
        report["failed"] += len(invalid_changes)

        for change in invalid_changes:
            report["changes"].append({
                "page_id": page_id,
                "selector": change.get("selector"),
                "status": "invalid_selector",
            })

        # Apply valid changes
        for change in valid_changes:
            original = html[page_id]

            try:
                # html_content must be provided directly in the change
                if "html_content" in change and change["html_content"]:
                    content = change["html_content"]
                    logger.debug(f"Using provided html_content for {change.get('selector')} ({len(content)} chars)")
                else:
                    # No content - skip with error
                    logger.error(f"Change missing html_content for {change.get('selector')}, skipping")
                    report["skipped"] += 1
                    report["changes"].append({
                        "page_id": page_id,
                        "selector": change.get("selector"),
                        "status": "skipped_no_content",
                    })
                    continue

                # Apply the change
                html[page_id] = apply_change(html[page_id], change, content)

                # Verify the change was applied
                if verify_change_applied(original, html[page_id], change):
                    report["applied"] += 1
                    report["changes"].append({
                        "page_id": page_id,
                        "selector": change.get("selector"),
                        "method": change.get("method"),
                        "change_type": change.get("change_type", "unknown"),
                        "status": "applied",
                        "content_length": len(content),
                    })
                else:
                    report["failed"] += 1
                    report["changes"].append({
                        "page_id": page_id,
                        "selector": change.get("selector"),
                        "status": "verification_failed",
                    })
                    logger.warning(
                        f"Change verification failed for {change.get('selector')} on {page_id}"
                    )

            except Exception as e:
                report["failed"] += 1
                report["changes"].append({
                    "page_id": page_id,
                    "selector": change.get("selector"),
                    "status": "error",
                    "error": str(e),
                })
                logger.error(f"Error applying change to {page_id}: {e}")

    logger.info(
        f"Variant {report['adv_variant_index']}: "
        f"{report['applied']} applied, {report['failed']} failed, {report['skipped']} skipped"
    )

    return html, report
