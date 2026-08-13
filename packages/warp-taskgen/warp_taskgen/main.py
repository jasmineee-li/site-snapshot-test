"""Compatibility entrypoint for the WorldSim CLI."""

from __future__ import annotations

from typing import Any

# ruff: noqa: F403
from warp_taskgen.cli import *
from warp_taskgen.cli import _impl as _legacy_impl

globals().update(
    {name: value for name, value in vars(_legacy_impl).items() if not name.startswith("__")}
)

_ORIGINAL_IMPL_FUNCS = {
    "build_parser": _legacy_impl.build_parser,
    "main": _legacy_impl.main,
    "_dispatch_inspect": _legacy_impl._dispatch_inspect,
    "_dispatch_phase": _legacy_impl._dispatch_phase,
    "_dispatch_preflight": _legacy_impl._dispatch_preflight,
    "_dispatch_resume": _legacy_impl._dispatch_resume,
    "_dispatch_status": _legacy_impl._dispatch_status,
    "_dispatch_task_bank": _legacy_impl._dispatch_task_bank,
}


def _sync_legacy_patches() -> None:
    for name, value in globals().items():
        if name.startswith("__") or name in {"_legacy_impl", "_sync_legacy_patches"}:
            continue
        if name in _ORIGINAL_IMPL_FUNCS:
            facade_wrapper = _FACADE_WRAPPERS.get(name)
            if value is facade_wrapper:
                setattr(_legacy_impl, name, _ORIGINAL_IMPL_FUNCS[name])
            elif value is not _ORIGINAL_IMPL_FUNCS[name]:
                setattr(_legacy_impl, name, value)
            continue
        if hasattr(_legacy_impl, name):
            setattr(_legacy_impl, name, value)


def build_parser() -> Any:
    _sync_legacy_patches()
    return _legacy_impl.build_parser()


def main(argv: list[str] | None = None) -> int:
    _sync_legacy_patches()
    return _legacy_impl.main(argv)


def _dispatch_resume(*args: Any, **kwargs: Any) -> int:
    _sync_legacy_patches()
    return _legacy_impl._dispatch_resume(*args, **kwargs)


def _dispatch_phase(*args: Any, **kwargs: Any) -> int:
    _sync_legacy_patches()
    return _legacy_impl._dispatch_phase(*args, **kwargs)


def _dispatch_preflight(*args: Any, **kwargs: Any) -> int:
    _sync_legacy_patches()
    return _legacy_impl._dispatch_preflight(*args, **kwargs)


def _dispatch_status(*args: Any, **kwargs: Any) -> int:
    _sync_legacy_patches()
    return _legacy_impl._dispatch_status(*args, **kwargs)


def _dispatch_inspect(*args: Any, **kwargs: Any) -> int:
    _sync_legacy_patches()
    return _legacy_impl._dispatch_inspect(*args, **kwargs)


def _dispatch_task_bank(*args: Any, **kwargs: Any) -> int:
    _sync_legacy_patches()
    return _legacy_impl._dispatch_task_bank(*args, **kwargs)


_FACADE_WRAPPERS = {
    "build_parser": build_parser,
    "main": main,
    "_dispatch_inspect": _dispatch_inspect,
    "_dispatch_phase": _dispatch_phase,
    "_dispatch_preflight": _dispatch_preflight,
    "_dispatch_resume": _dispatch_resume,
    "_dispatch_status": _dispatch_status,
    "_dispatch_task_bank": _dispatch_task_bank,
}


if __name__ == "__main__":
    raise SystemExit(main())
