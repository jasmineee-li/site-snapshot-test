"""Tiny prompt file loader.

Prompts live in ``worldsim/prompts/*.md``, one per pipeline step. Each file
is the verbatim prompt text from the v5 spec for that step.
"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


_VALIDATION_FOOTER_TEMPLATE = (
    "\n\n## Pre-submission validation\n\n"
    "Before you finish, validate your output:\n\n"
    "```bash\n"
    "python /workspace/_validate.py {command}\n"
    "```\n\n"
    "If it reports errors, fix your output and re-run until it prints "
    '`{{"valid": true}}`. Do not finish until validation passes.'
)


def load_prompt(name: str, *, validation_command: str | None = None) -> str:
    """Load a prompt file by stem name, e.g. ``load_prompt('profile-site')``.

    Some prompts contain template variables (e.g. ``{num_tasks}``). This
    function returns raw text — callers are responsible for substitution::

        load_prompt("generate-benign-tasks").format(num_tasks=30)

    For templates containing ``|`` (e.g. ``{pass|fail}`` in
    ``diagnose-benign-failure.md``), use ``str.replace()`` instead of
    ``str.format()`` since ``|`` is invalid in format specs.

    Args:
        name: Stem of a file under ``worldsim/prompts/`` (no extension).
        validation_command: When provided, a validation footer is appended
            instructing Claude Code to run ``_validate.py`` with this
            subcommand before finishing. Only the footer is ``.format()``-ed,
            not the full prompt text (which may contain literal ``{{ }}``
            for JSON examples).

    Returns:
        Full prompt text (unsubstituted), optionally with a validation footer.

    Raises:
        FileNotFoundError: If no matching prompt file exists, with a listing
            of all available prompt names in the error message.
    """
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.md"))
        raise FileNotFoundError(f"Prompt {name!r} not found at {path}. Available: {available}")
    text = path.read_text(encoding="utf-8")
    if validation_command is not None:
        # Only .format() the footer template, not the full prompt text.
        # Prompt files may contain literal {{ }} for JSON examples that
        # would break str.format().
        footer = _VALIDATION_FOOTER_TEMPLATE.format(command=validation_command)
        text += footer
    return text
