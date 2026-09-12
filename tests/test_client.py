"""Tests for NornsClient."""

import json

import httpx
import pytest
import respx

from norns import NornsClient
from norns.models import AgentResponse, ConversationResponse, EventResponse, RunResponse


BASE_URL = "http://localhost:4000"


@pytest.fixture
def client():
    c = NornsClient(BASE_URL, api_key="nrn_test123")
    yield c
    c.close()


# --- Auth ---


@respx.mock
def test_auth_header(client):
    route = respx.get(f"{BASE_URL}/api/v1/agents").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    client.list_agents()
    assert route.calls[0].request.headers["authorization"] == "Bearer nrn_test123"


# --- Agents ---


@respx.mock
def test_list_agents(client):
    respx.get(f"{BASE_URL}/api/v1/agents").mock(
        return_value=httpx.Response(200, json={"data": [
            {"id": 1, "name": "bot-1", "status": "active", "model": "claude-sonnet-5",
             "mode": "task", "system_prompt": "You help.", "max_steps": 50},
            {"id": 2, "name": "bot-2", "status": "active", "model": "claude-sonnet-5",
             "mode": "conversation", "system_prompt": "", "max_steps": 100},
        ]})
    )
    agents = client.list_agents()
    assert len(agents) == 2
    assert isinstance(agents[0], AgentResponse)
    assert agents[0].name == "bot-1"
    assert agents[1].mode == "conversation"


@respx.mock
def test_get_agent_by_id(client):
    respx.get(f"{BASE_URL}/api/v1/agents/1").mock(
        return_value=httpx.Response(200, json={"data": {
            "id": 1, "name": "bot-1", "status": "active", "model": "claude-sonnet-5",
            "mode": "task", "system_prompt": "You help.", "max_steps": 50,
        }})
    )
    agent = client.get_agent(1)
    assert agent.id == 1
    assert agent.name == "bot-1"


@respx.mock
def test_get_agent_by_name(client):
    respx.get(f"{BASE_URL}/api/v1/agents").mock(
        return_value=httpx.Response(200, json={"data": [
            {"id": 1, "name": "bot-1", "status": "active", "model": "claude-sonnet-5",
             "mode": "task", "system_prompt": "", "max_steps": 50},
            {"id": 2, "name": "support-bot", "status": "active", "model": "claude-sonnet-5",
             "mode": "conversation", "system_prompt": "", "max_steps": 50},
        ]})
    )
    agent = client.get_agent("support-bot")
    assert agent.id == 2


@respx.mock
def test_get_agent_by_name_not_found(client):
    respx.get(f"{BASE_URL}/api/v1/agents").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    with pytest.raises(ValueError, match="Agent not found"):
        client.get_agent("nonexistent")


# --- Send Message ---


@respx.mock
def test_send_message_fire_and_forget(client):
    respx.get(f"{BASE_URL}/api/v1/agents").mock(
        return_value=httpx.Response(200, json={"data": [
            {"id": 1, "name": "bot", "status": "active", "model": "m", "mode": "task",
             "system_prompt": "", "max_steps": 50},
        ]})
    )
    route = respx.post(f"{BASE_URL}/api/v1/agents/1/messages").mock(
        return_value=httpx.Response(202, json={"run_id": 42, "status": "accepted"})
    )

    result = client.send_message("bot", "Hello!")
    assert result.run_id == 42
    assert result.status == "accepted"
    assert result.output is None

    body = route.calls[0].request.read()
    import json
    parsed = json.loads(body)
    assert parsed["content"] == "Hello!"


@respx.mock
def test_send_message_with_conversation_key(client):
    respx.post(f"{BASE_URL}/api/v1/agents/1/messages").mock(
        return_value=httpx.Response(202, json={"run_id": 43, "status": "accepted"})
    )
    result = client.send_message(1, "Hi", conversation_key="slack:U01")
    assert result.conversation_key == "slack:U01"


@respx.mock
def test_send_message_wait(client):
    respx.post(f"{BASE_URL}/api/v1/agents/1/messages").mock(
        return_value=httpx.Response(202, json={"run_id": 42, "status": "accepted"})
    )
    respx.get(f"{BASE_URL}/api/v1/runs/42").mock(
        return_value=httpx.Response(200, json={"data": {
            "id": 42, "status": "completed", "output": "Here's your answer",
            "agent_id": 1, "conversation_id": 1, "trigger_type": "message",
            "inserted_at": "2025-01-01T00:00:00Z",
        }})
    )

    result = client.send_message(1, "Hello!", wait=True, timeout=5)
    assert result.status == "completed"
    assert result.output == "Here's your answer"


@respx.mock
def test_send_message_wait_timeout(client):
    respx.post(f"{BASE_URL}/api/v1/agents/1/messages").mock(
        return_value=httpx.Response(202, json={"run_id": 42, "status": "accepted"})
    )
    respx.get(f"{BASE_URL}/api/v1/runs/42").mock(
        return_value=httpx.Response(200, json={"data": {
            "id": 42, "status": "running", "output": None,
            "agent_id": 1, "conversation_id": 1, "trigger_type": "message",
            "inserted_at": "2025-01-01T00:00:00Z",
        }})
    )

    with pytest.raises(TimeoutError):
        client.send_message(1, "Hello!", wait=True, timeout=0.1)


# --- Run Inspection ---


@respx.mock
def test_get_run(client):
    respx.get(f"{BASE_URL}/api/v1/runs/42").mock(
        return_value=httpx.Response(200, json={"data": {
            "id": 42, "status": "completed", "output": "Done",
            "agent_id": 1, "conversation_id": 5, "trigger_type": "message",
            "inserted_at": "2025-01-01T00:00:00Z",
        }})
    )
    run = client.get_run(42)
    assert isinstance(run, RunResponse)
    assert run.run_id == 42
    assert run.status == "completed"
    assert run.output == "Done"


@respx.mock
def test_get_run_parses_waiting_for(client):
    respx.get(f"{BASE_URL}/api/v1/runs/43").mock(
        return_value=httpx.Response(200, json={"data": {
            "id": 43, "status": "waiting", "output": None,
            "agent_id": 1, "conversation_id": 5, "trigger_type": "message",
            "inserted_at": "2025-01-01T00:00:00Z",
            "waiting_for": {
                "question": "Book the 7pm show?",
                "tool_call_id": "call_ask",
                "asked_at": "2025-01-01T00:00:00Z",
            },
        }})
    )
    run = client.get_run(43)
    assert run.is_waiting
    assert run.waiting_for.question == "Book the 7pm show?"
    assert run.waiting_for.tool_call_id == "call_ask"


@respx.mock
def test_get_run_without_waiting_for(client):
    respx.get(f"{BASE_URL}/api/v1/runs/44").mock(
        return_value=httpx.Response(200, json={"data": {
            "id": 44, "status": "running", "output": None,
            "agent_id": 1, "conversation_id": None, "trigger_type": "message",
            "inserted_at": "2025-01-01T00:00:00Z",
        }})
    )
    run = client.get_run(44)
    assert run.waiting_for is None
    assert not run.is_waiting


@respx.mock
def test_send_message_wait_stops_on_waiting(client):
    """A parked agent is waiting on the caller — don't poll until timeout."""
    respx.post(f"{BASE_URL}/api/v1/agents/1/messages").mock(
        return_value=httpx.Response(202, json={"run_id": 45, "status": "accepted"})
    )
    respx.get(f"{BASE_URL}/api/v1/runs/45").mock(
        return_value=httpx.Response(200, json={"data": {
            "id": 45, "status": "waiting", "output": None,
            "agent_id": 1, "conversation_id": None, "trigger_type": "message",
            "inserted_at": "2025-01-01T00:00:00Z",
            "waiting_for": {"question": "7pm or 8pm?", "tool_call_id": "call_ask"},
        }})
    )

    result = client.send_message(1, "Book me a table", wait=True, timeout=5)
    assert result.is_waiting
    assert result.status == "waiting"
    assert result.waiting_for.question == "7pm or 8pm?"


@respx.mock
def test_reply(client):
    route = respx.post(f"{BASE_URL}/api/v1/runs/45/reply").mock(
        return_value=httpx.Response(202, json={"status": "accepted", "run_id": 45})
    )
    client.reply(45, "7pm")
    assert route.called
    assert json.loads(route.calls[0].request.content) == {"answer": "7pm"}


@respx.mock
def test_get_events(client):
    respx.get(f"{BASE_URL}/api/v1/runs/42/events").mock(
        return_value=httpx.Response(200, json={"data": [
            {"id": 1, "sequence": 1, "event_type": "llm_response",
             "payload": {"step": 1}, "source": "worker", "inserted_at": "2025-01-01T00:00:00Z"},
            {"id": 2, "sequence": 2, "event_type": "tool_call",
             "payload": {"name": "search"}, "source": "worker", "inserted_at": "2025-01-01T00:00:01Z"},
        ]})
    )
    events = client.get_events(42)
    assert len(events) == 2
    assert isinstance(events[0], EventResponse)
    assert events[0].event_type == "llm_response"
    assert events[1].payload == {"name": "search"}


# --- Conversations ---


@respx.mock
def test_list_conversations(client):
    respx.get(f"{BASE_URL}/api/v1/agents/1/conversations").mock(
        return_value=httpx.Response(200, json={"data": [
            {"id": 10, "agent_id": 1, "key": "slack:U01", "message_count": 5, "token_estimate": 1200},
        ]})
    )
    convos = client.list_conversations(1)
    assert len(convos) == 1
    assert isinstance(convos[0], ConversationResponse)
    assert convos[0].key == "slack:U01"


@respx.mock
def test_get_conversation(client):
    respx.get(f"{BASE_URL}/api/v1/agents/1/conversations/slack:U01").mock(
        return_value=httpx.Response(200, json={"data": {
            "id": 10, "agent_id": 1, "key": "slack:U01", "message_count": 5, "token_estimate": 1200,
        }})
    )
    convo = client.get_conversation(1, "slack:U01")
    assert convo.key == "slack:U01"
    assert convo.message_count == 5


@respx.mock
def test_delete_conversation(client):
    route = respx.delete(f"{BASE_URL}/api/v1/agents/1/conversations/slack:U01").mock(
        return_value=httpx.Response(204)
    )
    client.delete_conversation(1, "slack:U01")
    assert route.called


# --- Error Handling ---


@respx.mock
def test_error_handling_404(client):
    respx.get(f"{BASE_URL}/api/v1/agents/999").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.get_agent(999)


@respx.mock
def test_error_handling_500(client):
    respx.get(f"{BASE_URL}/api/v1/agents").mock(
        return_value=httpx.Response(500, json={"error": "internal"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.list_agents()


def test_max_tokens_comes_from_the_task_or_falls_back():
    """A turn that reaches the ceiling comes back cut off, so the agent's
    own ceiling has to reach the worker."""
    from norns.client import DEFAULT_MAX_TOKENS, _max_tokens

    assert _max_tokens({"max_tokens": 32000}) == 32000
    assert _max_tokens({}) == DEFAULT_MAX_TOKENS
    assert _max_tokens({"max_tokens": None}) == DEFAULT_MAX_TOKENS
    assert _max_tokens({"max_tokens": 0}) == DEFAULT_MAX_TOKENS
    assert _max_tokens({"max_tokens": "lots"}) == DEFAULT_MAX_TOKENS
    assert DEFAULT_MAX_TOKENS > 4096
