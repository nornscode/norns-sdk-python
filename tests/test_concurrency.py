"""Tasks on one connection execute concurrently.

A long-lived worker (a connector) serves many runs at once. A slow tool
or LLM call must not block the receive loop — before this guarantee,
every task on the connection serialized behind whichever one was slowest.
"""

import asyncio
import json

from norns import Agent, Norns, tool
from norns.client import websockets  # patched below

from tests.fakes import FakeWS, tool_task


def test_slow_tool_does_not_block_other_tasks(monkeypatch):
    events = []
    release = None  # set inside the loop

    @tool
    async def slow_tool(label: str) -> str:
        """Wait until released."""
        events.append(("start", label))
        await release.wait()
        events.append(("end", label))
        return label

    agent = Agent(name="t", system_prompt="t", tools=[slow_tool])
    norns = Norns("http://localhost:4000", api_key="nrn_test")

    fake = FakeWS()
    monkeypatch.setattr(websockets, "connect", lambda url: fake)

    async def scenario():
        nonlocal release
        release = asyncio.Event()

        tools_by_name = {t.name: t for t in agent.tools}
        serve = asyncio.create_task(norns._connect_and_serve(agent, "w1", tools_by_name))

        fake.incoming.put_nowait(tool_task("t1", "slow_tool", label="a"))
        fake.incoming.put_nowait(tool_task("t2", "slow_tool", label="b"))

        # Both tools must *start* while neither has finished — under the
        # old serial loop the second start waited for the first end, and
        # this hangs forever (t2 never starts while t1 blocks on release).
        for _ in range(100):
            if len([e for e in events if e[0] == "start"]) == 2:
                break
            await asyncio.sleep(0.01)
        assert events == [("start", "a"), ("start", "b")]

        release.set()
        r1 = await asyncio.wait_for(fake.results.get(), 2)
        r2 = await asyncio.wait_for(fake.results.get(), 2)
        assert {r1["task_id"], r2["task_id"]} == {"t1", "t2"}
        assert {r1["status"], r2["status"]} == {"ok"}

        fake.incoming.put_nowait(None)
        await asyncio.wait_for(serve, 2)

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_disconnect_cancels_in_flight_tasks(monkeypatch):
    state = {}
    cancelled = []

    @tool
    async def hanging_tool() -> str:
        """Run until cancelled."""
        state["started"].set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return "never"

    agent = Agent(name="t", system_prompt="t", tools=[hanging_tool])
    norns = Norns("http://localhost:4000", api_key="nrn_test")

    fake = FakeWS()
    monkeypatch.setattr(websockets, "connect", lambda url: fake)

    async def scenario():
        state["started"] = asyncio.Event()

        tools_by_name = {t.name: t for t in agent.tools}
        serve = asyncio.create_task(norns._connect_and_serve(agent, "w1", tools_by_name))

        fake.incoming.put_nowait(tool_task("t1", "hanging_tool"))
        await asyncio.wait_for(state["started"].wait(), 2)

        # Connection ends while the tool is still running.
        fake.incoming.put_nowait(None)
        await asyncio.wait_for(serve, 2)

        # Give the cancellation a tick to propagate.
        await asyncio.sleep(0)
        assert cancelled == [True]

    asyncio.run(asyncio.wait_for(scenario(), 10))
