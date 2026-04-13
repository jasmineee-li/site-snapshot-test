from __future__ import annotations

import pytest

from worldsim.runners.agentlab import AgentLabAgentWrapper


@pytest.mark.asyncio
async def test_agentlab_wrapper_satisfies_worker_lifecycle_contract():
    agent = AgentLabAgentWrapper(agent_args=object(), model="demo-model")

    await agent.setup("http://example.test")
    await agent.teardown()
