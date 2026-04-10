"""
Adversarial hardening agent for the adversarial hardening pipeline.

Generates hardened adversarial content based on trace analysis results,
targeting specific defense patterns identified in agent behavior.
"""

import json
import logging

from agentlab.benchmarks.hardening_models import (
    DEFENSE_TO_APS_ESCALATION,
    DefensePattern,
    HardeningPlan,
    TraceAnalysisResult,
)
from agentlab.benchmarks.llm_utils import OpenRouterLLMClient, strip_code_fences
from agentlab.benchmarks.hardening_prompts import (
    get_hardening_directive_prompt,
    get_hardening_data_js_prompt,
)
from agentlab.benchmarks.redteam.data_validator import DataValidationResult

logger = logging.getLogger(__name__)


class AdversarialHardeningAgent:
    """Generates hardened adversarial content based on trace analysis.

    Takes a TraceAnalysisResult and current injection content, then produces
    new adversarial variants that specifically target the agent's defense
    patterns.
    """

    def __init__(self, model: str = "openrouter/anthropic/claude-sonnet-4"):
        """
        Initialize the hardening agent.

        Args:
            model: LLM model for content generation (OpenRouter format).
        """
        self.model = model
        self.llm = OpenRouterLLMClient(model)

    def _resolve_target_dimension(
        self,
        analysis: TraceAnalysisResult,
    ) -> tuple[str | DefensePattern, str, str]:
        """Resolve the primary defense pattern and APS dimension to escalate."""
        if not analysis.defense_patterns:
            return "", "", ""

        primary_defense = analysis.defense_patterns[0]
        primary_defense_str = (
            primary_defense.value if isinstance(primary_defense, DefensePattern) else primary_defense
        )

        target_dimension = analysis.hardening_target
        if not target_dimension:
            if isinstance(primary_defense, DefensePattern):
                target_dimension = DEFENSE_TO_APS_ESCALATION.get(primary_defense, "camouflage")
            else:
                try:
                    dp_enum = DefensePattern(primary_defense)
                    target_dimension = DEFENSE_TO_APS_ESCALATION.get(dp_enum, "camouflage")
                except ValueError:
                    target_dimension = "camouflage"

        return primary_defense, primary_defense_str, target_dimension

    def harden_single(
        self,
        analysis: TraceAnalysisResult,
        behavior_config: dict,
        current_injections: dict,
    ) -> dict:
        """
        Generate hardened adversarial content for a single behavior.

        Maps the agent's primary defense pattern to an APS dimension, then
        generates new adversarial placeholder content targeting that dimension.

        Args:
            analysis: TraceAnalysisResult from trace analysis.
            behavior_config: Full behavior configuration dict.
            current_injections: Dict mapping placeholder IDs to content dicts,
                each with "benign" and "adversarial" keys.

        Returns:
            Updated placeholder_content dict with hardened adversarial variants.
        """
        if not analysis.defense_patterns:
            logger.info(
                f"[{analysis.behavior_id}] No defense patterns found, "
                f"skipping hardening"
            )
            return current_injections

        _, primary_defense_str, target_dimension = self._resolve_target_dimension(analysis)

        logger.info(
            f"[{analysis.behavior_id}] Hardening: targeting {target_dimension} "
            f"to defeat {primary_defense_str}"
        )

        prompt = get_hardening_directive_prompt(
            analysis_result=analysis.to_dict(),
            behavior_config=behavior_config,
            current_placeholder_content=current_injections,
            defense_pattern=primary_defense_str,
            target_aps_dimension=target_dimension,
        )

        try:
            response = self.llm.chat_simple(
                [{"role": "user", "content": prompt}],
                max_tokens=8192,
            )
            cleaned = strip_code_fences(response)
            updated_content = json.loads(cleaned)
            if not isinstance(updated_content, dict):
                raise ValueError("hardened content must be a JSON object")
            for array_name, items in updated_content.items():
                if not isinstance(items, list):
                    raise ValueError(f"hardened content for {array_name} must be a list")

            logger.info(
                f"[{analysis.behavior_id}] Generated hardened content for "
                f"{len(updated_content)} data array(s)"
            )
            return updated_content

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(
                f"[{analysis.behavior_id}] Failed to parse hardened content: {e}"
            )
            return current_injections
        except Exception as e:
            logger.error(
                f"[{analysis.behavior_id}] Hardening generation failed: {e}"
            )
            return current_injections

    def harden_single_data_js(
        self,
        analysis: TraceAnalysisResult,
        behavior_config: dict,
        benign_data_js: str,
        current_adversarial_data_js: str,
        retry_feedback: DataValidationResult | None = None,
    ) -> str:
        """Generate a hardened full-file app-mode ``data.js`` candidate."""
        if not analysis.defense_patterns:
            logger.info(
                f"[{analysis.behavior_id}] No defense patterns found, "
                f"preserving current adversarial data.js"
            )
            return current_adversarial_data_js

        _, primary_defense_str, target_dimension = self._resolve_target_dimension(analysis)
        prompt = get_hardening_data_js_prompt(
            analysis_result=analysis.to_dict(),
            behavior_config=behavior_config,
            benign_data_js=benign_data_js,
            current_adversarial_data_js=current_adversarial_data_js,
            defense_pattern=primary_defense_str,
            target_aps_dimension=target_dimension,
            retry_feedback=retry_feedback.to_dict() if retry_feedback else None,
        )

        try:
            response = self.llm.chat_simple(
                [{"role": "user", "content": prompt}],
                max_tokens=8192,
            )
            cleaned = strip_code_fences(response).strip()
            if not cleaned:
                logger.warning(
                    f"[{analysis.behavior_id}] Hardening returned empty data.js; "
                    f"preserving current adversarial file"
                )
                return current_adversarial_data_js
            return cleaned + ("\n" if not cleaned.endswith("\n") else "")
        except Exception as e:
            logger.error(
                f"[{analysis.behavior_id}] data.js hardening generation failed: {e}"
            )
            return current_adversarial_data_js

    def harden_batch(
        self,
        plan: HardeningPlan,
        behaviors: dict[str, dict],
        injections: dict[str, dict],
    ) -> dict[str, dict]:
        """
        Process all behaviors in the hardening queue.

        Args:
            plan: HardeningPlan with the hardening_queue.
            behaviors: Dict mapping behavior_id to behavior config.
            injections: Dict mapping behavior_id to placeholder content dicts.

        Returns:
            Updated injections dict with hardened adversarial variants.
        """
        updated_injections = dict(injections)

        for item in plan.hardening_queue:
            behavior_id = item["behavior_id"]
            behavior_config = behaviors.get(behavior_id)
            current_content = injections.get(behavior_id, {})

            if not behavior_config:
                logger.warning(
                    f"No behavior config found for {behavior_id}, skipping"
                )
                continue

            # Reconstruct a minimal TraceAnalysisResult from the queue item
            analysis = TraceAnalysisResult(
                behavior_id=behavior_id,
                trial_classification="resisted",
                propensity_signal_confidence=0.5,
                defense_patterns=[
                    DefensePattern(dp) for dp in item.get("defense_patterns", [])
                    if dp in [e.value for e in DefensePattern]
                ],
                vulnerability_patterns=[],
                aps_dimensions={item["target_dimension"]: item["current_level"]},
                evidence=[],
                hardening_target=item["target_dimension"],
                hardening_rationale=item.get("rationale", ""),
            )

            hardened = self.harden_single(analysis, behavior_config, current_content)
            updated_injections[behavior_id] = hardened

        logger.info(
            f"Batch hardening complete: processed {len(plan.hardening_queue)} behaviors"
        )
        return updated_injections
