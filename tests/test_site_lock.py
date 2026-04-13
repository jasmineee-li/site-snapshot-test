from __future__ import annotations

import asyncio
import time

import pytest

from worldsim.site_lock import reset_locks, site_lock


@pytest.fixture(autouse=True)
def _clean_locks():
    reset_locks()
    yield
    reset_locks()


@pytest.mark.asyncio
async def test_same_site_serializes():
    """Two tasks for the same site must not overlap."""
    timeline: list[tuple[str, str]] = []

    async def work(label: str) -> None:
        async with site_lock("shopping"):
            timeline.append((label, "enter"))
            await asyncio.sleep(0.05)
            timeline.append((label, "exit"))

    await asyncio.gather(work("a"), work("b"))

    # One must finish before the other starts — no interleaving.
    enters = [i for i, (_, ev) in enumerate(timeline) if ev == "enter"]
    exits = [i for i, (_, ev) in enumerate(timeline) if ev == "exit"]
    # The second enter must come after the first exit.
    assert enters[1] > exits[0]


@pytest.mark.asyncio
async def test_different_sites_run_in_parallel():
    """Tasks on different sites must overlap in time."""
    timestamps: dict[str, list[float]] = {}

    async def work(site: str) -> None:
        async with site_lock(site):
            timestamps[site] = [time.monotonic()]
            await asyncio.sleep(0.05)
            timestamps[site].append(time.monotonic())

    await asyncio.gather(work("shopping"), work("gitlab"))

    # They should overlap: the second task's start should be before the first
    # task's end (allowing for tiny scheduling jitter).
    shopping_start, shopping_end = timestamps["shopping"]
    gitlab_start, gitlab_end = timestamps["gitlab"]
    # At least one must have started before the other ended.
    assert shopping_start < gitlab_end and gitlab_start < shopping_end


@pytest.mark.asyncio
async def test_exception_releases_lock():
    """An exception inside site_lock must release the lock."""

    with pytest.raises(RuntimeError, match="boom"):
        async with site_lock("shopping"):
            raise RuntimeError("boom")

    # The lock should be available again — this must not hang.
    acquired = False
    async with site_lock("shopping"):
        acquired = True

    assert acquired
