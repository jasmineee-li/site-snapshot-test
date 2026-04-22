"""Tiny prompt file loader.

Prompts live in ``worldsim/prompts/*.md``, one per pipeline step. Each file
is the verbatim prompt text from the v5 spec for that step.
"""

from __future__ import annotations

from pathlib import Path

from worldsim.editors._registry import (
    EDITOR_CONTRACT_TABLE_SENTINEL,
    ContractRenderContext,
    render_contract_table,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"


class PromptRenderError(RuntimeError):
    """Raised when a prompt file contains a sentinel the caller did not
    supply rendering context for."""


_VALIDATION_FOOTER_TEMPLATE = (
    "\n\n## Pre-submission validation\n\n"
    "Before you finish, validate your output:\n\n"
    "```bash\n"
    "python /workspace/_validate.py {command}\n"
    "```\n\n"
    "If it reports errors, fix your output and re-run until it prints "
    '`{{"valid": true}}`. Do not finish until validation passes.'
)


def load_prompt(
    name: str,
    *,
    validation_command: str | None = None,
    contract_context: ContractRenderContext | None = None,
) -> str:
    """Load a prompt file by stem name, e.g. ``load_prompt('profile-site')``.

    Some prompts contain template variables (e.g. ``{num_tasks}``). This
    function returns raw text — callers are responsible for substitution::

        load_prompt("generate-benign-tasks").format(num_tasks=30)

    For templates containing ``|`` (e.g. ``{pass|fail}``), use
    ``str.replace()`` instead of ``str.format()`` since ``|`` is invalid
    in format specs.

    Args:
        name: Stem of a file under ``worldsim/prompts/`` (no extension).
        validation_command: When provided, a validation footer is appended
            instructing Claude Code to run ``_validate.py`` with this
            subcommand before finishing. Only the footer is ``.format()``-ed,
            not the full prompt text (which may contain literal ``{{ }}``
            for JSON examples).
        contract_context: When provided, replaces any
            ``<!-- EDITOR_CONTRACT_TABLE -->`` sentinel in the prompt file
            with the rendered editor-method contract for the given site
            and in-shard kinds (see
            :func:`worldsim.editors._registry.render_contract_table`). If
            the sentinel is present but this is ``None``, raises
            :class:`PromptRenderError` — never serve a prompt with the
            sentinel unreplaced.

    Returns:
        Full prompt text (unsubstituted), optionally with a validation footer.

    Raises:
        FileNotFoundError: If no matching prompt file exists, with a listing
            of all available prompt names in the error message.
        PromptRenderError: If the file contains
            ``EDITOR_CONTRACT_TABLE_SENTINEL`` but ``contract_context`` is
            ``None``.
    """
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.md"))
        raise FileNotFoundError(f"Prompt {name!r} not found at {path}. Available: {available}")
    text = path.read_text(encoding="utf-8")

    if EDITOR_CONTRACT_TABLE_SENTINEL in text:
        if contract_context is None:
            raise PromptRenderError(
                f"Prompt {name!r} contains {EDITOR_CONTRACT_TABLE_SENTINEL} "
                "but load_prompt was called without contract_context. Pass a "
                "ContractRenderContext(site=..., kind_anchors=...) so the "
                "sentinel can be rendered."
            )
        rendered = render_contract_table(contract_context).rstrip("\n")
        text = text.replace(EDITOR_CONTRACT_TABLE_SENTINEL, rendered)

    if validation_command is not None:
        # Only .format() the footer template, not the full prompt text.
        # Prompt files may contain literal {{ }} for JSON examples that
        # would break str.format().
        footer = _VALIDATION_FOOTER_TEMPLATE.format(command=validation_command)
        text += footer
    return text
