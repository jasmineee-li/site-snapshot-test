"""Phase 2c retry-delay dependency seam."""

from __future__ import annotations

import asyncio


async def phase_2c_retry_sleep(delay_seconds: float) -> None:
    """Wait for a Phase 2c retry without owning a global clock seam."""

    await asyncio.sleep(delay_seconds)


__all__ = ["phase_2c_retry_sleep"]
