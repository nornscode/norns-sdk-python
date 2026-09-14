"""Tests for agent definition and tool decorator."""

import pytest

from norns import Agent, tool


def test_tool_decorator_basic():
    @tool
    def search(query: str) -> str:
        """Search the web."""
        return f"results for {query}"

    assert search.name == "search"
    assert search.description == "Search the web."
    assert search.side_effect is False
    assert search.input_schema["properties"]["query"]["type"] == "string"
    assert search.input_schema["required"] == ["query"]


def test_tool_decorator_with_options():
    @tool(name="send_email", side_effect=True)
    def email(to: str, subject: str, body: str) -> str:
        """Send an email."""
        return "sent"

    assert email.name == "send_email"
    assert email.side_effect is True
    assert set(email.input_schema["required"]) == {"to", "subject", "body"}


def test_tool_decorator_optional_params():
    @tool
    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

    assert greet.input_schema["required"] == ["name"]
    assert "greeting" in greet.input_schema["properties"]


def test_tool_type_inference():
    @tool
    def mixed(text: str, count: int, ratio: float, flag: bool) -> str:
        return "ok"

    props = mixed.input_schema["properties"]
    assert props["text"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["ratio"]["type"] == "number"
    assert props["flag"]["type"] == "boolean"


def test_tool_handler_callable():
    @tool
    def echo(msg: str) -> str:
        return msg

    result = echo.handler(msg="hello")
    assert result == "hello"


def test_tool_registration_format():
    @tool(side_effect=True)
    def dangerous(target: str) -> str:
        """Do something risky."""
        return "done"

    reg = dangerous.to_registration()
    assert reg["name"] == "dangerous"
    assert reg["description"] == "Do something risky."
    assert reg["side_effect"] is True
    assert "input_schema" in reg


def test_agent_defaults():
    agent = Agent(name="test")
    assert agent.model == "claude-sonnet-5"
    assert agent.mode == "task"
    assert agent.on_failure == "retry_last_step"
    assert agent.max_steps == 50
    assert agent.tools == []


def test_agent_with_tools():
    @tool
    def search(query: str) -> str:
        """Search."""
        return ""

    agent = Agent(name="bot", tools=[search])
    assert len(agent.tools) == 1


def test_agent_registration_format():
    @tool
    def search(query: str) -> str:
        """Search."""
        return ""

    agent = Agent(
        name="bot",
        model="claude-haiku-4-5-20251001",
        system_prompt="You are helpful.",
        tools=[search],
        mode="conversation",
    )

    reg = agent.to_registration()
    assert reg["name"] == "bot"
    assert reg["model"] == "claude-haiku-4-5-20251001"
    assert reg["system_prompt"] == "You are helpful."
    assert reg["mode"] == "conversation"
    assert reg["tools"] == ["search"]


def test_agent_policy_fields_default_to_unset():
    agent = Agent(name="test")
    assert agent.allowed_tools is None
    assert agent.subagents is None
    assert agent.subagent_conversation is None


def test_allowed_tools_must_name_the_agents_tools():
    @tool
    def search(query: str) -> str:
        """Search."""
        return ""

    Agent(name="bot", tools=[search], allowed_tools=["search"])
    with pytest.raises(ValueError, match="send_email"):
        Agent(name="bot", tools=[search], allowed_tools=["search", "send_email"])
    with pytest.raises(ValueError, match="built-ins"):
        Agent(name="bot", tools=[search], allowed_tools=["ask_human"])


def test_subagent_conversation_is_validated():
    for value in ("per_launch", "per_parent"):
        assert Agent(name="bot", subagent_conversation=value).subagent_conversation == value
    with pytest.raises(ValueError, match="per_parent"):
        Agent(name="bot", subagent_conversation="shared")
