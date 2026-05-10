from __future__ import annotations

import asyncio
import time

import pytest

from worldsim.site_lock import reset_locks, site_lock, task_lock


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


@pytest.mark.asyncio
async def test_task_lock_allows_same_site_parallelism_on_distinct_instances():
    """Bound tasks with distinct reset endpoints should run in parallel."""
    timestamps: dict[str, list[float]] = {}

    async def work(reset_endpoint: str) -> None:
        task = {
            "site": "shopping",
            "sites": ["shopping"],
            "_worldsim_runtime": {
                "reset_endpoints": [reset_endpoint],
                "sites": ["shopping"],
            },
        }
        async with task_lock(task):
            timestamps[reset_endpoint] = [time.monotonic()]
            await asyncio.sleep(0.05)
            timestamps[reset_endpoint].append(time.monotonic())

    await asyncio.gather(
        work("http://shopping-1.test/init"),
        work("http://shopping-2.test/init"),
    )

    first_start, first_end = timestamps["http://shopping-1.test/init"]
    second_start, second_end = timestamps["http://shopping-2.test/init"]
    assert first_start < second_end and second_start < first_end


@pytest.mark.asyncio
async def test_task_lock_serializes_on_shared_secondary_endpoint():
    """Tasks that share any reset endpoint must not overlap."""
    timeline: list[tuple[str, str]] = []

    async def work(label: str, endpoints: list[str]) -> None:
        task = {
            "sites": ["shopping", "gitlab"],
            "_worldsim_runtime": {
                "reset_endpoints": endpoints,
                "sites": ["shopping", "gitlab"],
            },
        }
        async with task_lock(task):
            timeline.append((label, "enter"))
            await asyncio.sleep(0.05)
            timeline.append((label, "exit"))

    await asyncio.gather(
        work("a", ["http://shopping-1.test/init", "http://gitlab.test/init"]),
        work("b", ["http://shopping-2.test/init", "http://gitlab.test/init"]),
    )

    enters = [i for i, (_, ev) in enumerate(timeline) if ev == "enter"]
    exits = [i for i, (_, ev) in enumerate(timeline) if ev == "exit"]
    assert enters[1] > exits[0]
