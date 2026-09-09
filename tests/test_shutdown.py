"""Graceful shutdown: drain, finish in-flight work, leave, return.

A connector under a supervisor (volund, a Fly reconciler) gets stopped and
restarted routinely. On SIGTERM/SIGINT or `shutdown()` the worker must tell
Norns to stop sending it work, finish what it holds, report the results,
leave the channel, and let `run()` return — instead of dying mid-task.
"""

import asyncio
import os
import signal

from norns import Agent, Norns, tool
from norns.client import websockets  # patched below

from tests.fakes import FakeWS, tool_task


def make(monkeypatch, tools, **run_opts):
    agent = Agent(name="t", system_prompt="t", tools=tools)
    norns = Norns("http://localhost:4000", api_key="nrn_test")
    norns._shutdown_timeout = run_opts.get("shutdown_timeout", 5.0)
    fake = FakeWS()
    monkeypatch.setattr(websockets, "connect", lambda url: fake)
    return agent, norns, fake


async def wait_until(pred, timeout=2):
    for _ in range(int(timeout / 0.01)):
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met in time")


def test_shutdown_drains_in_flight_work_then_leaves(monkeypatch):
    state = {}

    @tool
    async def slow_tool() -> str:
        """Wait until released."""
        state["started"].set()
        await state["release"].wait()
        return "finished"

    agent, norns, fake = make(monkeypatch, [slow_tool])

    async def scenario():
        state["started"], state["release"] = asyncio.Event(), asyncio.Event()
        run = asyncio.create_task(norns._run_loop(agent, "w1"))

        fake.incoming.put_nowait(tool_task("t1", "slow_tool"))
        await asyncio.wait_for(state["started"].wait(), 2)

        norns.shutdown()
        await wait_until(lambda: "drain" in fake.events())
        # Still waiting on the tool: no leave, no result, loop still running.
        assert "phx_leave" not in fake.events()
        assert fake.results.empty()
        assert not run.done()

        state["release"].set()
        result = await asyncio.wait_for(fake.results.get(), 2)
        assert result["task_id"] == "t1" and result["status"] == "ok"

        await asyncio.wait_for(run, 2)
        assert fake.events()[-2:] == ["tool_result", "phx_leave"] or fake.events()[-1] == "phx_leave"
        assert fake.closed

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_task_arriving_during_drain_is_still_served(monkeypatch):
    state = {}

    @tool
    async def slow_tool(label: str) -> str:
        """Wait until released."""
        state["started"][label] = True
        await state["release"].wait()
        return label

    agent, norns, fake = make(monkeypatch, [slow_tool])

    async def scenario():
        state["started"], state["release"] = {}, asyncio.Event()
        run = asyncio.create_task(norns._run_loop(agent, "w1"))

        fake.incoming.put_nowait(tool_task("t1", "slow_tool", label="a"))
        await wait_until(lambda: "a" in state["started"])

        norns.shutdown()
        await wait_until(lambda: "drain" in fake.events())
        # Dispatched before Norns processed the drain: ours to finish.
        fake.incoming.put_nowait(tool_task("t2", "slow_tool", label="b"))
        await wait_until(lambda: "b" in state["started"])

        state["release"].set()
        await asyncio.wait_for(run, 2)
        delivered = {fake.results.get_nowait()["task_id"] for _ in range(fake.results.qsize())}
        assert delivered == {"t1", "t2"}
        assert fake.events()[-1] == "phx_leave"

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_shutdown_deadline_drops_stuck_tasks(monkeypatch):
    state = {"cancelled": False}

    @tool
    async def stuck_tool() -> str:
        """Never finishes."""
        state["started"].set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        return "never"

    agent, norns, fake = make(monkeypatch, [stuck_tool], shutdown_timeout=0.1)

    async def scenario():
        state["started"] = asyncio.Event()
        run = asyncio.create_task(norns._run_loop(agent, "w1"))
        fake.incoming.put_nowait(tool_task("t1", "stuck_tool"))
        await asyncio.wait_for(state["started"].wait(), 2)

        norns.shutdown()
        await asyncio.wait_for(run, 2)
        await asyncio.sleep(0)

        assert state["cancelled"]
        assert fake.results.empty()
        assert fake.events()[-1] == "phx_leave"

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_shutdown_during_reconnect_backoff_returns_promptly(monkeypatch):
    agent = Agent(name="t", system_prompt="t", tools=[])
    norns = Norns("http://localhost:4000", api_key="nrn_test")

    def refuse(url):
        raise ConnectionError("refused")

    monkeypatch.setattr(websockets, "connect", refuse)

    async def scenario():
        run = asyncio.create_task(norns._run_loop(agent, "w1"))
        await asyncio.sleep(0.05)  # inside the 3s backoff
        norns.shutdown()
        await asyncio.wait_for(run, 1)

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_sigterm_starts_a_drain(monkeypatch):
    agent, norns, fake = make(monkeypatch, [])

    async def scenario():
        run = asyncio.create_task(norns._run_loop(agent, "w1"))
        await wait_until(lambda: "phx_join" in fake.events())

        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(run, 2)
        assert fake.events()[-2:] == ["drain", "phx_leave"]

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_shutdown_before_run_is_a_noop():
    norns = Norns("http://localhost:4000", api_key="nrn_test")
    norns.shutdown()
