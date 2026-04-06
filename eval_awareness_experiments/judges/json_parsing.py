"""Shared JSON parsing utilities.

Copied verbatim from eval-awareness-testbed/utils/json_parsing.py.
"""

import json
import re


def parse_json_response(response: str) -> dict:
    """Parse JSON from a model response, handling code blocks and embedded JSON."""
    if not response:
        raise ValueError("Empty response")

    cleaned = response.strip()

    # Handle markdown code blocks
    if cleaned.startswith("```"):
        json_match = re.search(r"```json\s*(.*?)\s*```", cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        first_newline = cleaned.find("\n")
        last_fence = cleaned.rfind("```")
        if first_newline != -1 and last_fence > first_newline:
            cleaned = cleaned[first_newline + 1:last_fence].strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in response
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(json_pattern, response, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse JSON from response: {response[:200]}...")


def parse_json_safe(response: str) -> dict | None:
    """Parse JSON, returning None on failure instead of raising."""
    try:
        return parse_json_response(response)
    except ValueError:
        return None
