from __future__ import annotations

from .base import BaseSiteEditor, EditorError
from .gitlab import GitlabEditor
from .reddit import RedditEditor
from .shopping import ShoppingEditor
from .shopping_admin import ShoppingAdminEditor

EDITOR_REGISTRY: dict[tuple[str, str], type[BaseSiteEditor]] = {
    ("webarena_verified", "gitlab"): GitlabEditor,
    ("webarena_verified", "reddit"): RedditEditor,
    ("webarena_verified", "shopping"): ShoppingEditor,
    ("webarena_verified", "shopping_admin"): ShoppingAdminEditor,
}

__all__ = [
    "BaseSiteEditor",
    "EditorError",
    "EDITOR_REGISTRY",
    "GitlabEditor",
    "RedditEditor",
    "ShoppingEditor",
    "ShoppingAdminEditor",
]
