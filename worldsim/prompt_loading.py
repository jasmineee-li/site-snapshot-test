"""Tiny prompt file loader.

Prompts live in ``worldsim/prompts/*.md``, one per pipeline step. Each file
is the verbatim prompt text from the v5 spec for that step.
"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt file by stem name, e.g. ``load_prompt('profile-site')``.

    Some prompts contain template variables (e.g. ``{num_tasks}``). This
    function returns raw text — callers are responsible for substitution::

        load_prompt("generate-benign-tasks").format(num_tasks=30)

    For templates containing ``|`` (e.g. ``{pass|fail}`` in
    ``diagnose-benign-failure.md``), use ``str.replace()`` instead of
    ``str.format()`` since ``|`` is invalid in format specs.

    Args:
        name: Stem of a file under ``worldsim/prompts/`` (no extension).

    Returns:
        Full prompt text (unsubstituted).

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
