from __future__ import annotations

from .base import BaseSiteEditor, EditorError
from .gitlab import GitlabEditor
from .reddit import RedditEditor

EDITOR_REGISTRY: dict[tuple[str, str], type[BaseSiteEditor]] = {
    ("webarena_verified", "gitlab"): GitlabEditor,
    ("webarena_verified", "reddit"): RedditEditor,
}

__all__ = [
    "EDITOR_REGISTRY",
    "BaseSiteEditor",
    "EditorError",
    "GitlabEditor",
    "RedditEditor",
]
