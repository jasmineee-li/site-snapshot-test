"""Compatibility entrypoint for the WorldSim CLI."""

from __future__ import annotations

from typing import Any

# ruff: noqa: F403
from worldsim.cli import *
from worldsim.cli import _impl as _legacy_impl

globals().update(
    {
        name: value
        for name, value in vars(_legacy_impl).items()
        if not name.startswith("__")
    }
)


def _sync_legacy_patches() -> None:
    for name, value in globals().items():
        if name.startswith("__") or name in {"_legacy_impl", "_sync_legacy_patches"}:
            continue
        if hasattr(_legacy_impl, name):
            setattr(_legacy_impl, name, value)


def build_parser() -> Any:
    _sync_legacy_patches()
    return _legacy_impl.build_parser()


def main(argv: list[str] | None = None) -> int:
    _sync_legacy_patches()
    return _legacy_impl.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
