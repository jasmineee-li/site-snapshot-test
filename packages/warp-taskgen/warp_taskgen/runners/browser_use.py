"""Browser Use runner adapter."""

from __future__ import annotations

from collections.abc import Callable

from warp_taskgen.agent_runtime import AgentRunner


def make_agent_factory(*args: object, **kwargs: object) -> Callable[[], AgentRunner]:
    """Delegate to the existing Browser Use factory.

    Kept in a runner module so selection code can treat Browser Use like every
    other runtime without moving the mature implementation in this slice.
    """

    from warp_taskgen.agent_config import make_agent_factory as _make_agent_factory

    kwargs.pop("runner", None)
    return _make_agent_factory(*args, runner="browser_use", **kwargs)
