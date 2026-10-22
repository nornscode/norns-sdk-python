"""A worker on a scoped key (nrnw_) joins without registering its agents,
declares only the capabilities it was given, and stops when the key is
revoked."""

import asyncio
import json

import pytest

from norns import Agent, KeyRevoked, Norns
from norns.client import websockets  # patched below

from tests.fakes import FakeWS


@pytest.fixture
def agent():
    return Agent(name="box-worker", system_prompt="test")


def _noop_loop(self, agents, wid):
    async def coro():
        pass

    return coro()


def test_scoped_key_skips_agent_registration(agent, monkeypatch, caplog):
    def unreachable(self, a):
        raise AssertionError("a scoped key must not touch the REST API")

    monkeypatch.setattr(Norns, "_ensure_agent", unreachable)
    monkeypatch.setattr(Norns, "_run_loop", _noop_loop)

    norns = Norns("http://localhost:4000", api_key="nrnw_abc")
    with caplog.at_level("INFO", logger="norns"):
        norns.run(agent)

    assert "skipping agent registration" in caplog.text


def test_tenant_key_still_registers(agent, monkeypatch):
    ensured = []
    monkeypatch.setattr(Norns, "_ensure_agent", lambda self, a: ensured.append(a.name))
    monkeypatch.setattr(Norns, "_run_loop", _noop_loop)

    Norns("http://localhost:4000", api_key="nrn_abc").run(agent)

    assert ensured == ["box-worker"]


def test_capabilities_default_to_both(agent):
    norns = Norns("http://localhost:4000", api_key="nrnw_abc")
    assert norns._join_payload(agent, "w1")["capabilities"] == ["llm", "tools"]


def test_capabilities_land_in_the_join_payload(agent, monkeypatch):
    monkeypatch.setattr(Norns, "_run_loop", _noop_loop)
    norns = Norns("http://localhost:4000", api_key="nrnw_abc")

    norns.run(agent, capabilities=["tools"])

    assert norns._join_payload(agent, "w1")["capabilities"] == ["tools"]


def test_capabilities_are_checked_before_connecting(agent, monkeypatch):
    monkeypatch.setattr(Norns, "_run_loop", _noop_loop)
    norns = Norns("http://localhost:4000", api_key="nrnw_abc")

    with pytest.raises(ValueError, match="capabilities"):
        norns.run(agent, capabilities=["shell"])
    with pytest.raises(ValueError, match="capabilities"):
        norns.run(agent, capabilities=[])


def test_key_revoked_stops_the_worker(agent, monkeypatch):
    fake = FakeWS()
    monkeypatch.setattr(websockets, "connect", lambda url: fake)
    norns = Norns("http://localhost:4000", api_key="nrnw_abc")

    async def scenario():
        run = asyncio.create_task(norns._run_loop(agent, "w1"))
        fake.incoming.put_nowait(json.dumps([None, None, "worker:lobby", "key_revoked", {}]))
        with pytest.raises(KeyRevoked):
            await asyncio.wait_for(run, timeout=2)

    asyncio.run(scenario())
