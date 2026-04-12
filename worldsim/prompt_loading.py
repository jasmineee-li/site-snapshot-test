"""Tiny prompt file loader.

Prompts live in ``worldsim/prompts/*.md``, one per pipeline step. Each file
is the verbatim prompt text from the v5 spec for that step.
"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt file by stem name, e.g. ``load_prompt('profile-site')``.

    Args:
        name: Stem of a file under ``worldsim/prompts/`` (no extension).

    Returns:
        Full prompt text.

    Raises:
        FileNotFoundError: If no matching prompt file exists, with a listing
            of all available prompt names in the error message.
    """
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.md"))
        raise FileNotFoundError(
            f"Prompt {name!r} not found at {path}. Available: {available}"
        )
    return path.read_text(encoding="utf-8")
