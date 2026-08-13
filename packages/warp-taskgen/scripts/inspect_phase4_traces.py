#!/usr/bin/env python3
"""Compatibility wrapper for `uv run python -m warp_taskgen.main trace ...`."""

from warp_taskgen.phase_4.trace_inspection_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
