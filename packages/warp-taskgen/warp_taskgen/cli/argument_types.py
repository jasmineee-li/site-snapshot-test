"""Argparse ``type=`` validators shared by the WARP Taskgen CLI parsers."""

from __future__ import annotations

import argparse
import math


def _positive_int(value: str) -> int:
    """Argparse type for positive integer CLI flags."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    """Argparse type for 0-or-greater integer CLI flags."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _non_negative_float(value: str) -> float:
    """Argparse type for finite zero-or-greater durations."""

    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite number >= 0")
    return parsed
