"""
Validation utilities for the injection pipeline.

This module provides validation functions used during the injection pipeline:
- validate_selectors: Check that CSS selectors exist in HTML before applying changes
- verify_change_applied: Confirm that a change was successfully applied

These utilities ensure fail-fast behavior and provide detailed logging for debugging.

Note: apply_change() and generate_adversarial_variant() have been moved to
redteam_attacker.py to keep the full adversarial pipeline (plan + apply) together.
"""

from __future__ import annotations

import logging

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def validate_selectors(html: str, changes: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Validate that CSS selectors exist in HTML.

    This is a pre-flight check before applying changes. Invalid selectors
    indicate either LLM hallucination or HTML structure mismatch.

    Args:
        html: HTML content to check selectors against
        changes: List of change dicts, each with a 'selector' field

    Returns:
        Tuple of (valid_changes, invalid_changes)
        - valid_changes: Changes whose selectors were found in the HTML
        - invalid_changes: Changes whose selectors were not found
    """
    soup = BeautifulSoup(html, "html.parser")
    valid, invalid = [], []

    for change in changes:
        selector = change.get("selector")
        if not selector:
            logger.warning(f"Change missing selector: {change}")
            invalid.append(change)
            continue

        try:
            element = soup.select_one(selector)
            if element:
                valid.append(change)
            else:
                invalid.append(change)
                logger.warning(f"Selector not found in HTML: {selector}")
        except Exception as e:
            invalid.append(change)
            logger.warning(f"Invalid CSS selector '{selector}': {e}")

    if invalid:
        logger.warning(
            f"Selector validation: {len(valid)} valid, {len(invalid)} invalid"
        )

    return valid, invalid


def verify_change_applied(
    original_html: str,
    modified_html: str,
    change: dict,
) -> bool:
    """
    Verify that a change was actually applied by checking if the target element changed.

    Args:
        original_html: HTML before the change
        modified_html: HTML after the change
        change: The change dict that was applied

    Returns:
        True if the change was successfully applied (element content differs)
    """
    selector = change.get("selector")
    if not selector:
        return False

    try:
        orig_soup = BeautifulSoup(original_html, "html.parser")
        mod_soup = BeautifulSoup(modified_html, "html.parser")

        orig_elem = orig_soup.select_one(selector)
        mod_elem = mod_soup.select_one(selector)

        if not orig_elem or not mod_elem:
            # Element missing in one or both - something changed
            return orig_elem != mod_elem

        # Check if content changed
        return str(orig_elem) != str(mod_elem)

    except Exception as e:
        logger.warning(f"Error verifying change at {selector}: {e}")
        return False
