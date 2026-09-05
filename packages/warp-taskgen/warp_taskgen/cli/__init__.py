"""Public package surface for the WARP Taskgen CLI."""

from __future__ import annotations

# ruff: noqa: F401
# ``env`` must load first: it runs the dotenv bootstrap before ``args`` and its
# siblings import ``warp_taskgen.config``.
from warp_taskgen.cli import env
from warp_taskgen.cli.args import build_parser
from warp_taskgen.cli.dispatch import main
