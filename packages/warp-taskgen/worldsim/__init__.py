"""Lazy compatibility adapter for the retired ``worldsim`` package name.

The implementation lives exclusively in :mod:`warp_taskgen`.  This package
does not mirror the implementation tree: a small import hook resolves legacy
submodule names to the canonical module object, keeping module state and
monkeypatches shared during the migration window.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_LEGACY_PREFIX = __name__ + "."
_CANONICAL_PREFIX = "warp_taskgen."
_CANONICAL_ROOT = Path(__file__).resolve().parent.parent / "warp_taskgen"


class _LegacyModuleLoader(importlib.abc.Loader):
    """Return the canonical module object for one legacy import."""

    def __init__(self, canonical_name: str) -> None:
        self.canonical_name = canonical_name
        self._canonical_metadata: dict[str, object] | None = None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = importlib.import_module(self.canonical_name)
        self._canonical_metadata = {
            key: getattr(module, key, None)
            for key in (
                "__name__",
                "__package__",
                "__loader__",
                "__spec__",
                "__file__",
                "__cached__",
                "__path__",
            )
        }
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module: ModuleType) -> None:
        # The canonical import performed all initialization.  Keeping this a
        # no-op is what prevents a second implementation graph from forming.
        if self._canonical_metadata is None:
            return None
        for key, value in self._canonical_metadata.items():
            if value is None:
                continue
            setattr(module, key, value)

    def get_code(self, fullname: str):
        """Provide the temporary ``python -m worldsim.main`` command alias."""

        if self.canonical_name != "warp_taskgen.main":
            return compile("", f"<{fullname} compatibility adapter>", "exec")
        return compile(
            "from warp_taskgen.main import main as _main\nraise SystemExit(_main())\n",
            "<worldsim.main compatibility adapter>",
            "exec",
        )

    def get_filename(self, fullname: str) -> str:
        return f"<compatibility adapter for {self.canonical_name}>"


class _LegacyModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(_LEGACY_PREFIX):
            return None
        canonical_name = _CANONICAL_PREFIX + fullname[len(_LEGACY_PREFIX) :]
        if not _canonical_module_exists(canonical_name):
            return None
        loader = _LegacyModuleLoader(canonical_name)
        spec = importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=loader.get_filename(fullname),
            is_package=_is_package(canonical_name),
        )
        return spec


def _is_package(canonical_name: str) -> bool:
    """Check package metadata without importing any canonical module."""

    relative = canonical_name.removeprefix("warp_taskgen.").replace(".", "/")
    return (_CANONICAL_ROOT / relative / "__init__.py").is_file()


def _canonical_module_exists(canonical_name: str) -> bool:
    """Return whether a canonical module/package exists on disk."""

    relative = canonical_name.removeprefix("warp_taskgen.").replace(".", "/")
    return (_CANONICAL_ROOT / f"{relative}.py").is_file() or (
        _CANONICAL_ROOT / relative / "__init__.py"
    ).is_file()


_FINDER = _LegacyModuleFinder()
if not any(isinstance(finder, _LegacyModuleFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _FINDER)

__all__ = ["__version__"]
__version__ = "0.1.0"

# Make the package root itself one canonical object too.  Importing the root
# does not import any child module; submodule names remain lazy through the
# finder above.
_CANONICAL_PACKAGE = importlib.import_module("warp_taskgen")
sys.modules[__name__] = _CANONICAL_PACKAGE
