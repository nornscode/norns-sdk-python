"""One worker, several agents: every agent ensured, one join, the union of
their tools."""

import asyncio
import json

import httpx
import pytest
import respx

from norns import Agent, Norns, tool
from norns.client import _llm_provider, _tools_by_name, websockets  # websockets patched below

from tests.fakes import FakeWS, tool_task

BASE_URL = "http://localhost:4000"


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return f"read {path}"


@tool(side_effect=True)
def write_file(path: str) -> str:
    """Write a file."""
    return f"wrote {path}"


def team():
    lead = Agent(
        name="lead",
        tools=[read_file, write_file],
        subagents={"mode": "allowlist", "allowed_agents": ["explore"]},
    )
    explore = Agent(
        name="explore",
        tools=[read_file],
        allowed_tools=["read_file"],
        subagent_conversation="per_launch",
    )
    return lead, explore


@pytest.fixture
def norns():
    return Norns(BASE_URL, api_key="nrn_test")


def test_join_payload_lists_every_agent_and_the_union_of_tools(norns):
    payload = norns._join_payload(list(team()), "w1")

    assert [a["name"] for a in payload["agents"]] == ["lead", "explore"]
    # read_file belongs to both and is registered once.
    assert [t["name"] for t in payload["tools"]] == ["read_file", "write_file"]


def test_one_agent_joins_as_before(norns):
    lead, _ = team()
    assert norns._join_payload(lead, "w1") == norns._join_payload([lead], "w1")


def test_a_tuple_of_agents_works(norns):
    assert len(norns._join_payload(team(), "w1")["agents"]) == 2


def test_different_tools_with_one_name_raise_before_connecting(norns, monkeypatch):
    @tool(name="read_file")
    def other_read(path: str) -> str:
        """A different read."""
        return ""

    def unreachable(*args):
        raise AssertionError("reached the server")

    monkeypatch.setattr(Norns, "_ensure_agent", unreachable)
    monkeypatch.setattr(websockets, "connect", unreachable)

    with pytest.raises(ValueError, match="read_file"):
        norns.run([Agent(name="a", tools=[read_file]), Agent(name="b", tools=[other_read])])


def test_duplicate_agent_names_raise(norns):
    with pytest.raises(ValueError, match="unique"):
        norns._join_payload([Agent(name="a"), Agent(name="a")], "w1")


def test_run_ensures_every_agent(norns, monkeypatch):
    ensured, served = [], []
    monkeypatch.setattr(Norns, "_ensure_agent", lambda self, a: ensured.append(a.name))

    async def noop_loop(self, agents, wid):
        served.append([a.name for a in agents])

    monkeypatch.setattr(Norns, "_run_loop", noop_loop)

    norns.run(list(team()))

    assert ensured == ["lead", "explore"]
    assert served == [["lead", "explore"]]


def test_tool_tasks_dispatch_across_every_agents_tools(norns, monkeypatch):
    fake = FakeWS()
    monkeypatch.setattr(websockets, "connect", lambda url: fake)
    norns._shutdown_timeout = 2.0

    async def scenario():
        run = asyncio.create_task(norns._run_loop(list(team()), "w1"))
        fake.incoming.put_nowait(tool_task("t1", "read_file", path="a"))
        fake.incoming.put_nowait(tool_task("t2", "write_file", path="b"))

        results = {}
        for _ in range(2):
            r = await asyncio.wait_for(fake.results.get(), 2)
            results[r["task_id"]] = r["result"]
        assert results == {"t1": "read a", "t2": "wrote b"}

        join = fake.sent[0]
        assert join[3] == "phx_join"
        assert [a["name"] for a in join[4]["agents"]] == ["lead", "explore"]

        norns.shutdown()
        await asyncio.wait_for(run, 2)

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_agents_relying_on_llm_provider_must_share_one():
    a = Agent(name="a", llm_provider="anthropic")
    b = Agent(name="b", model="gpt-4o", llm_provider="openai")
    with pytest.raises(ValueError, match="llm_provider"):
        _llm_provider([a, b])

    # A provider prefix takes the model out of the vote.
    b = Agent(name="b", model="openai/gpt-4o", llm_provider="openai")
    assert _llm_provider([a, b]) == "anthropic"


def test_the_same_tool_in_several_agents_is_one_tool():
    lead, explore = team()
    assert _tools_by_name([lead, explore]) == {"read_file": read_file, "write_file": write_file}


@respx.mock
def test_ensure_agent_sends_policies_only_when_set(norns):
    respx.get(f"{BASE_URL}/api/v1/agents").mock(
        return_value=httpx.Response(200, json={"data": [{"id": 1, "name": "lead"}]})
    )
    put = respx.put(f"{BASE_URL}/api/v1/agents/1").mock(return_value=httpx.Response(200, json={"data": {}}))
    post = respx.post(f"{BASE_URL}/api/v1/agents").mock(
        return_value=httpx.Response(201, json={"data": {"id": 2}})
    )

    lead, explore = team()
    norns._ensure_agent(lead)
    norns._ensure_agent(explore)

    lead_config = json.loads(put.calls[0].request.content)["model_config"]
    assert lead_config["subagents"] == {"mode": "allowlist", "allowed_agents": ["explore"]}
    assert "tools" not in lead_config
    assert "subagent_conversation" not in lead_config

    explore_config = json.loads(post.calls[0].request.content)["model_config"]
    assert explore_config["tools"] == {"mode": "allowlist", "allowed_tools": ["read_file"]}
    assert explore_config["subagent_conversation"] == "per_launch"
    assert "subagents" not in explore_config
