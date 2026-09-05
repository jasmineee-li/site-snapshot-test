"""Executable entrypoint for the WARP Taskgen CLI."""

from __future__ import annotations

from warp_taskgen.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
