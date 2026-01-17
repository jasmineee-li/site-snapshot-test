"""
Validation utilities for the injection pipeline.

This module provides validation functions used during the injection pipeline:
- validate_selectors: Check that CSS selectors exist in HTML before applying changes
- verify_change_applied: Confirm that a change was successfully applied
- apply_change: Apply a single change to HTML using BeautifulSoup
- generate_adversarial_variant: Orchestrate Phase 3 (apply all changes with validation)

These utilities ensure fail-fast behavior and provide detailed logging for debugging.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from agentlab.benchmarks.redteam_attacker import RedteamAttackerAgent

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
    attacker: "RedteamAttackerAgent",
) -> tuple[dict[str, str], dict]:
    """
    Generate one adversarial HTML variant by applying all changes.

    This is the Phase 3 orchestration function that:
    1. Validates selectors before applying changes
    2. Generates content for each change
    3. Applies changes to HTML
    4. Verifies each change was applied

    Args:
        base_html: Dict mapping page_id to base HTML content
        variant_plan: Variant plan from Phase 2 with 'changes' array
        behavior: Behavior config dict with 'doc', 'target', etc.
        attacker: RedteamAttackerAgent instance for content generation

    Returns:
        Tuple of (variant_html, report)
        - variant_html: Dict mapping page_id to modified HTML
        - report: Dict with statistics about applied/failed changes
    """
    # Deep copy base HTML to avoid modifying original
    html = copy.deepcopy(base_html)

    report = {
        "adv_variant_index": variant_plan.get("adv_variant_index", 0),
        "strategy_description": variant_plan.get("strategy_description", ""),
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
                # Generate content for this change
                content = attacker.generate_change_content(
                    change=change,
                    behavior=behavior,
                    page_html=html[page_id],
                )

                # Apply the change
                html[page_id] = apply_change(html[page_id], change, content)

                # Verify the change was applied
                if verify_change_applied(original, html[page_id], change):
                    report["applied"] += 1
                    report["changes"].append({
                        "page_id": page_id,
                        "selector": change.get("selector"),
                        "method": change.get("method"),
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
