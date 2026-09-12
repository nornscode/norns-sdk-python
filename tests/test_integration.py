"""Integration tests against a live Norns server.

Requires:
    NORNS_URL  — e.g. http://localhost:4001
    NORNS_API_KEY — tenant API key
    ANTHROPIC_API_KEY — for LLM calls in worker tests

Skip automatically if not set.
"""

import os
import threading
import time
import uuid

import pytest

from norns import Agent, Norns, NornsClient, tool

NORNS_URL = os.environ.get("NORNS_URL")
NORNS_API_KEY = os.environ.get("NORNS_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# The SDK's own default, so these cannot drift onto a model the provider
# has retired — which is what happened to the version this replaced. Set
# NORNS_TEST_MODEL to run them against another one.
MODEL = os.environ.get("NORNS_TEST_MODEL") or Agent.model

pytestmark = pytest.mark.skipif(
    not NORNS_URL or not NORNS_API_KEY,
    reason="NORNS_URL and NORNS_API_KEY not set",
)


@pytest.fixture
def client():
    c = NornsClient(NORNS_URL, api_key=NORNS_API_KEY)
    yield c
    c.close()


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _create_agent(name: str) -> Agent:
    """Create an agent definition and ensure it exists on the server."""
    agent = Agent(
        name=name,
        model=MODEL,
        system_prompt="You are a test agent. Reply concisely.",
    )
    worker = Norns(NORNS_URL, api_key=NORNS_API_KEY)
    worker._ensure_agent(agent)
    return agent


# --- Agent Management ---


def test_list_agents(client):
    agents = client.list_agents()
    assert isinstance(agents, list)


def test_create_and_get_agent(client):
    name = _unique_name("test-agent")
    _create_agent(name)
    found = client.get_agent(name)
    assert found.name == name


# --- Full round-trip: worker + client ---


@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set")
def test_send_message_and_complete(client):
    """Send a message, have a worker process it, verify completion."""
    name = _unique_name("test-roundtrip")

    @tool
    def echo(text: str) -> str:
        """Echo back the input."""
        return text

    agent_def = Agent(
        name=name,
        model=MODEL,
        system_prompt="You are a test agent. Use the echo tool with the user's exact message, then reply with what it returned.",
        tools=[echo],
    )

    worker = Norns(NORNS_URL, api_key=NORNS_API_KEY)
    worker._ensure_agent(agent_def)

    # Start worker in background thread
    def run_worker():
        try:
            worker.run(agent_def)
        except Exception:
            pass

    worker_thread = threading.Thread(target=run_worker, daemon=True)
    worker_thread.start()
    time.sleep(2)  # let the worker connect

    agent = client.get_agent(name)

    # Send and wait for completion
    result = client.send_message(agent.id, "hello", wait=True, timeout=60)
    assert result.status == "completed"
    assert result.output is not None
    assert len(result.output) > 0

    # Verify run details
    run = client.get_run(result.run_id)
    assert run.run_id == result.run_id
    assert run.status == "completed"

    # Verify events were logged
    events = client.get_events(result.run_id)
    assert isinstance(events, list)
    assert len(events) > 0
    event_types = {e.event_type for e in events}
    assert "llm_response" in event_types


# --- Conversations ---


def test_list_conversations(client):
    agents = client.list_agents()
    if not agents:
        pytest.skip("No agents registered on server")
    convos = client.list_conversations(agents[0].id)
    assert isinstance(convos, list)
