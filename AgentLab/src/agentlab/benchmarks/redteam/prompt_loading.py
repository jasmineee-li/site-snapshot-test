"""Prompt template and guide loading helpers.

Extracted from ``app_pipeline.py`` to keep prompt-resolution logic in a
dedicated module.  Public API: :func:`ensure_trusted_server_template` and
:func:`load_prompt_template`.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam.app_artifacts import (
    compute_docs_snapshot,
    resolve_docs_source_path,
    resolve_repo_root_path,
)
from agentlab.benchmarks.redteam.utils import sha256_file as _sha256_file

logger = logging.getLogger(__name__)

# Resolve paths relative to *this* file – mirrors the originals in
# ``app_pipeline.py`` so the constants stay identical when this module
# lives in the same package directory.
_PACKAGE_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"
_PROMPTS_DIR = _PACKAGE_DIR / "prompts"
_DEFAULT_DESIGN_GUIDES_DIR = _PACKAGE_DIR / "guides"
_AUTHORING_CONTEXT_DIR = ".redteam_authoring"
_AUTHORING_GUIDES_DIR = Path(_AUTHORING_CONTEXT_DIR) / "guides"
_AUTHORING_MANUALS_DIR = Path(_AUTHORING_CONTEXT_DIR) / "manuals"
_AUTHORING_CLAUDEIGNORE_MARKER = "# managed by agentlab redteam authoring workspace"


# ---------------------------------------------------------------------------
# Server template helpers
# ---------------------------------------------------------------------------


def _server_template_source(template_dir: str | Path | None) -> Path:
    """Return the canonical trusted server template path."""
    if template_dir:
        return Path(template_dir) / "server.py"
    return _TEMPLATES_DIR / "server.py"


def ensure_trusted_server_template(
    app_dir: str | Path,
    template_dir: str | Path | None = None,
) -> str | None:
    """Restore and fail closed if the reserved server template changed."""
    app_dir = Path(app_dir)
    template_path = _server_template_source(template_dir)
    server_path = app_dir / "server.py"

    if not template_path.exists():
        return f"Trusted server template not found: {template_path}"
    if not server_path.exists():
        return f"Reserved runtime file missing: {server_path}"

    if _sha256_file(server_path) == _sha256_file(template_path):
        return None

    shutil.copy2(template_path, server_path)
    return (
        "Reserved runtime file server.py was modified during generation/audit; "
        "restored the trusted template and failed closed."
    )


# ---------------------------------------------------------------------------
# Prompt template loading
# ---------------------------------------------------------------------------


def load_prompt_template(template_path: str | Path, **kwargs: str) -> str:
    """Load a Markdown prompt template and format it with *kwargs*.

    Args:
        template_path: Path to a ``.md`` template file.  Relative paths are
            resolved against the package ``prompts/`` directory.
        **kwargs: Substitution variables for ``str.format()``.

    Returns:
        The formatted prompt string.

    Raises:
        FileNotFoundError: If the template does not exist.
    """
    path = Path(template_path)
    if not path.is_absolute():
        path = _PROMPTS_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    template_text = path.read_text(encoding="utf-8")
    if kwargs:
        template_text = template_text.format(**kwargs)
    return template_text


# ---------------------------------------------------------------------------
# Design-guide resolution
# ---------------------------------------------------------------------------


def _resolve_design_guides_dir(
    design_guides_dir: str | Path | None,
) -> Path:
    """Resolve the directory containing app-generation guide documents."""
    if design_guides_dir:
        return Path(design_guides_dir).resolve()
    return _DEFAULT_DESIGN_GUIDES_DIR


def _require_guide_path(guides_dir: Path, filename: str) -> Path:
    guide_path = guides_dir / filename
    if not guide_path.exists():
        raise FileNotFoundError(
            f"Required redteam guide is missing: {guide_path}. "
            "Install the packaged guide assets or pass --design-guides-dir explicitly."
        )
    return guide_path.resolve()


def _relative_prompt_path(working_dir: Path, target: Path) -> str:
    return f"./{target.relative_to(working_dir).as_posix()}"


def _ensure_authoring_claudeignore(working_dir: Path) -> None:
    claudeignore_path = working_dir / ".claudeignore"
    if claudeignore_path.exists():
        return
    claudeignore_path.write_text(
        "\n".join(
            [
                _AUTHORING_CLAUDEIGNORE_MARKER,
                "results/",
                "__pycache__/",
                "*.pyc",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _stage_authoring_file(
    *,
    source: Path,
    working_dir: Path,
    relative_dir: Path,
) -> str:
    target = working_dir / relative_dir / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    _ensure_authoring_claudeignore(working_dir)
    return _relative_prompt_path(working_dir, target)


def _copy_authoring_tree(source: Path, target: Path) -> None:
    if source.is_symlink():
        raise ValueError(f"Authoring context source must not be a symlink: {source}")
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Authoring context source must not contain symlinks: {path}")
        relative_path = path.relative_to(source)
        destination = target / relative_path
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def _guide_prompt_kwargs(
    design_guides_dir: str | Path | None,
    *,
    working_dir: str | Path | None = None,
) -> dict[str, str]:
    guides_dir = _resolve_design_guides_dir(design_guides_dir)
    paths = {
        "app_design_guide_path": _require_guide_path(guides_dir, "app-design-guide.md"),
        "app_data_guide_path": _require_guide_path(guides_dir, "app-data-guide.md"),
        "app_environment_protocol_path": _require_guide_path(
            guides_dir,
            "app-environment-protocol.md",
        ),
        "app_variant_guide_path": _require_guide_path(guides_dir, "app-variant-guide.md"),
    }
    if working_dir is None:
        return {key: str(path) for key, path in paths.items()}

    working_path = Path(working_dir)
    return {
        key: _stage_authoring_file(
            source=path,
            working_dir=working_path,
            relative_dir=_AUTHORING_GUIDES_DIR,
        )
        for key, path in paths.items()
    }


def _task_guide_prompt_kwargs(
    *,
    working_dir: str | Path | None = None,
) -> dict[str, str]:
    docs_dir = _DEFAULT_DESIGN_GUIDES_DIR
    paths = {
        "function_task_design_guide_path": _require_guide_path(
            docs_dir,
            "function-task-design-guide.md",
        ),
        "real_task_design_guide_path": _require_guide_path(
            docs_dir,
            "real-task-design-guide.md",
        ),
        "verifier_sanity_check_guide_path": _require_guide_path(
            docs_dir,
            "verifier-sanity-check.md",
        ),
    }
    if working_dir is None:
        return {key: str(path) for key, path in paths.items()}

    working_path = Path(working_dir)
    return {
        key: _stage_authoring_file(
            source=path,
            working_dir=working_path,
            relative_dir=_AUTHORING_GUIDES_DIR,
        )
        for key, path in paths.items()
    }


def _manual_corpus_prompt_kwargs(
    behavior_spec: dict[str, Any],
    *,
    repo_root_path: str | Path | None,
    working_dir: str | Path | None = None,
) -> dict[str, str]:
    docs_path = str(behavior_spec.get("docs_path") or "").strip()
    if not docs_path:
        return {"docs_authority_block": ""}

    root = (
        Path(repo_root_path).resolve()
        if repo_root_path is not None
        else resolve_repo_root_path(_PACKAGE_DIR)
    )
    docs_source = resolve_docs_source_path(
        repo_root_path=root,
        docs_path=docs_path,
    )
    if docs_source is None or not docs_source.exists():
        raise FileNotFoundError(
            f"Missing manual corpus for docs_path {docs_path!r} relative to {root}"
        )

    docs_snapshot = compute_docs_snapshot(
        {"docs_path": docs_path},
        repo_root_path=root,
    )
    if working_dir is None:
        rendered_corpus_path = str(docs_source.resolve())
    else:
        working_path = Path(working_dir)
        staged_docs_root = working_path / _AUTHORING_MANUALS_DIR / Path(docs_path)
        _copy_authoring_tree(docs_source, staged_docs_root)
        _ensure_authoring_claudeignore(working_path)
        rendered_corpus_path = _relative_prompt_path(working_path, staged_docs_root)
    return {
        "docs_authority_block": (
            "Read the authoritative user manuals and platform docs before you generate the app:\n"
            f"- Declared docs_path: {docs_path}\n"
            f"- Resolved corpus path: {rendered_corpus_path}\n"
            "## Manual Corpus Snapshot\n"
            f"{json.dumps(docs_snapshot, indent=2, ensure_ascii=False)}\n"
        ),
    }


# ---------------------------------------------------------------------------
# Server template copy
# ---------------------------------------------------------------------------


def _copy_server_template(template_dir: str | Path | None, app_dir: Path) -> None:
    """Copy ``server.py`` from *template_dir* into *app_dir*.

    If *template_dir* is ``None`` or unset, uses the package-level
    ``templates/`` directory.
    """
    if template_dir:
        src = Path(template_dir) / "server.py"
    else:
        src = _TEMPLATES_DIR / "server.py"

    if not src.exists():
        logger.warning(
            "Server template not found at %s — app will need a manually "
            "written server.py",
            src,
        )
        return

    dst = app_dir / "server.py"
    shutil.copy2(src, dst)
    logger.info("Copied server template to %s", dst)
