"""A side effect happens once, however many times the call is dispatched.

Core names every side-effecting call with a key derived from things that
survive a crash — run, step, tool call id, gard — and re-dispatches the
call if the result never reached it. It cannot know whether the effect
landed; only the worker that ran it knows. So the worker keeps what it
did, and answers the second dispatch from that.
"""

import asyncio

from norns import Agent, Norns, tool
from norns.client import MAX_REMEMBERED_RESULTS, websockets  # patched below

from tests.fakes import FakeWS, tool_task


def charging_agent(charges):
    @tool
    def charge_card(amount: int) -> str:
        """Take money."""
        charges.append(amount)
        return f"charged {amount}"

    return Agent(name="t", system_prompt="t", tools=[charge_card])


def serve(monkeypatch, agent, script):
    """Run `script(fake)` against a worker connected to a fake socket."""
    fake = FakeWS()
    monkeypatch.setattr(websockets, "connect", lambda url: fake)
    norns = Norns("http://localhost:4000", api_key="nrn_test")

    async def scenario():
        tools_by_name = {t.name: t for t in agent.tools}
        task = asyncio.create_task(norns._connect_and_serve(agent, "w1", tools_by_name))
        await script(fake, norns)
        fake.incoming.put_nowait(None)
        await asyncio.wait_for(task, 2)

    asyncio.run(asyncio.wait_for(scenario(), 10))
    return norns


def test_the_same_key_twice_charges_once(monkeypatch):
    charges = []

    async def script(fake, _norns):
        fake.incoming.put_nowait(tool_task("t1", "charge_card", idempotency_key="k1", amount=10))
        first = await asyncio.wait_for(fake.results.get(), 2)

        # The result never reached core, so core asks again. Same call, same key.
        fake.incoming.put_nowait(tool_task("t2", "charge_card", idempotency_key="k1", amount=10))
        second = await asyncio.wait_for(fake.results.get(), 2)

        assert first["status"] == "ok"
        assert first["result"] == "charged 10"
        assert "duplicate" not in first

        # Same answer, and flagged, so core can record it rather than
        # showing what looks like a second charge.
        assert second["result"] == "charged 10"
        assert second["duplicate"] is True
        assert second["task_id"] == "t2"

    serve(monkeypatch, charging_agent(charges), script)
    assert charges == [10]


def test_without_a_key_nothing_is_remembered(monkeypatch):
    charges = []

    async def script(fake, norns):
        fake.incoming.put_nowait(tool_task("t1", "charge_card", amount=10))
        await asyncio.wait_for(fake.results.get(), 2)
        fake.incoming.put_nowait(tool_task("t2", "charge_card", amount=10))
        second = await asyncio.wait_for(fake.results.get(), 2)

        # No key means core did not consider this side-effecting. Twice asked
        # is twice meant.
        assert "duplicate" not in second
        assert norns._completed == {}

    serve(monkeypatch, charging_agent(charges), script)
    assert charges == [10, 10]


def test_a_failure_is_not_remembered(monkeypatch):
    attempts = []

    @tool
    def flaky(n: int) -> str:
        """Fail once, then work."""
        attempts.append(n)
        if len(attempts) == 1:
            raise RuntimeError("network went away")
        return "ok"

    agent = Agent(name="t", system_prompt="t", tools=[flaky])

    async def script(fake, _norns):
        fake.incoming.put_nowait(tool_task("t1", "flaky", idempotency_key="k1", n=1))
        first = await asyncio.wait_for(fake.results.get(), 2)
        fake.incoming.put_nowait(tool_task("t2", "flaky", idempotency_key="k1", n=1))
        second = await asyncio.wait_for(fake.results.get(), 2)

        assert first["status"] == "error"
        # A retry of something that did not happen should happen.
        assert second["status"] == "ok"
        assert "duplicate" not in second

    serve(monkeypatch, agent, script)
    assert len(attempts) == 2


def test_the_memory_is_bounded(monkeypatch):
    charges = []

    async def script(fake, norns):
        for i in range(MAX_REMEMBERED_RESULTS + 10):
            fake.incoming.put_nowait(tool_task(f"t{i}", "charge_card", idempotency_key=f"k{i}", amount=1))
            await asyncio.wait_for(fake.results.get(), 2)

        # A worker lives for days; this is a crash guard, not a cache.
        assert len(norns._completed) == MAX_REMEMBERED_RESULTS
        assert "k0" not in norns._completed
        assert f"k{MAX_REMEMBERED_RESULTS + 9}" in norns._completed

    serve(monkeypatch, charging_agent(charges), script)
