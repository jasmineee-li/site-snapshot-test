from __future__ import annotations

from ._registry import register_editor as _register_editor
from .base import BaseSiteEditor, EditorError
from .gitlab import GitlabEditor
from .reddit import RedditEditor

EDITOR_REGISTRY: dict[tuple[str, str], type[BaseSiteEditor]] = {
    ("webarena_verified", "gitlab"): GitlabEditor,
    ("webarena_verified", "reddit"): RedditEditor,
}

# Populate the editor-method contract registry at package import time.
# _register_editor walks each editor class's `supported_methods` and reads
# the `_editor_method_spec` attribute attached by `@editor_method` on each
# method. Any drift (missing decorator on a supported method, duplicate
# registration) raises RegistryError here — no pipeline starts with an
# inconsistent registry.
_register_editor(GitlabEditor, site="gitlab")
_register_editor(RedditEditor, site="reddit")

__all__ = [
    "EDITOR_REGISTRY",
    "BaseSiteEditor",
    "EditorError",
    "GitlabEditor",
    "RedditEditor",
]
