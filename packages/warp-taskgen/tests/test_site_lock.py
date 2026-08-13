from __future__ import annotations

import asyncio

import pytest

from warp_taskgen.site_lock import reset_locks, site_lock, task_lock


@pytest.fixture(autouse=True)
def _clean_locks():
    reset_locks()
    yield
    reset_locks()


@pytest.mark.asyncio
async def test_same_site_serializes():
    """Two tasks for the same site must not overlap."""
    timeline: list[tuple[str, str]] = []
    first_entered = asyncio.Event()
    second_attempting = asyncio.Event()
    second_entered = asyncio.Event()
    release_first = asyncio.Event()

    async def first_work() -> None:
        async with site_lock("shopping"):
            timeline.append(("a", "enter"))
            first_entered.set()
            await release_first.wait()
            timeline.append(("a", "exit"))

    async def second_work() -> None:
        await first_entered.wait()
        second_attempting.set()
        async with site_lock("shopping"):
            second_entered.set()
            timeline.append(("b", "enter"))
            timeline.append(("b", "exit"))

    first_task = asyncio.create_task(first_work())
    second_task = asyncio.create_task(second_work())
    await second_attempting.wait()
    assert not second_entered.is_set()
    release_first.set()
    await asyncio.gather(first_task, second_task)

    # One must finish before the other starts — no interleaving.
    enters = [i for i, (_, ev) in enumerate(timeline) if ev == "enter"]
    exits = [i for i, (_, ev) in enumerate(timeline) if ev == "exit"]
    # The second enter must come after the first exit.
    assert enters[1] > exits[0]


@pytest.mark.asyncio
async def test_different_sites_run_in_parallel():
    """Tasks on different sites must overlap in time."""
    entered = {
        "shopping": asyncio.Event(),
        "gitlab": asyncio.Event(),
    }

    async def work(site: str, other_site: str) -> None:
        async with site_lock(site):
            entered[site].set()
            await entered[other_site].wait()

    await asyncio.wait_for(
        asyncio.gather(work("shopping", "gitlab"), work("gitlab", "shopping")),
        timeout=1,
    )


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
    first_endpoint = "http://shopping-1.test/init"
    second_endpoint = "http://shopping-2.test/init"
    entered = {
        first_endpoint: asyncio.Event(),
        second_endpoint: asyncio.Event(),
    }

    async def work(reset_endpoint: str, other_endpoint: str) -> None:
        task = {
            "site": "shopping",
            "sites": ["shopping"],
            "_worldsim_runtime": {
                "reset_endpoints": [reset_endpoint],
                "sites": ["shopping"],
            },
        }
        async with task_lock(task):
            entered[reset_endpoint].set()
            await entered[other_endpoint].wait()

    await asyncio.wait_for(
        asyncio.gather(
            work(first_endpoint, second_endpoint),
            work(second_endpoint, first_endpoint),
        ),
        timeout=1,
    )


@pytest.mark.asyncio
async def test_task_lock_serializes_on_shared_secondary_endpoint():
    """Tasks that share any reset endpoint must not overlap."""
    timeline: list[tuple[str, str]] = []
    first_entered = asyncio.Event()
    second_attempting = asyncio.Event()
    second_entered = asyncio.Event()
    release_first = asyncio.Event()

    async def work(label: str, endpoints: list[str], *, hold: bool) -> None:
        task = {
            "sites": ["shopping", "gitlab"],
            "_worldsim_runtime": {
                "reset_endpoints": endpoints,
                "sites": ["shopping", "gitlab"],
            },
        }
        if not hold:
            await first_entered.wait()
            second_attempting.set()
        async with task_lock(task):
            if hold:
                first_entered.set()
            else:
                second_entered.set()
            timeline.append((label, "enter"))
            if hold:
                await release_first.wait()
            timeline.append((label, "exit"))

    first_task = asyncio.create_task(
        work(
            "a",
            ["http://shopping-1.test/init", "http://gitlab.test/init"],
            hold=True,
        )
    )
    second_task = asyncio.create_task(
        work(
            "b",
            ["http://shopping-2.test/init", "http://gitlab.test/init"],
            hold=False,
        )
    )
    await second_attempting.wait()
    assert not second_entered.is_set()
    release_first.set()
    await asyncio.gather(first_task, second_task)

    enters = [i for i, (_, ev) in enumerate(timeline) if ev == "enter"]
    exits = [i for i, (_, ev) in enumerate(timeline) if ev == "exit"]
    assert enters[1] > exits[0]
