"""Tests for neutral <-> LiteLLM format translation."""

import json
from unittest.mock import MagicMock

from norns.client import _to_litellm_tools, _from_litellm_response


def test_to_litellm_tools():
    tools = _to_litellm_tools([{
        "name": "search",
        "description": "Search the web",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }])
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "search"
    assert tools[0]["function"]["parameters"] == {"type": "object", "properties": {"q": {"type": "string"}}}


def test_to_litellm_tools_multiple():
    tools = _to_litellm_tools([
        {"name": "a", "description": "Tool A", "parameters": {}},
        {"name": "b", "description": "Tool B", "parameters": {}},
    ])
    assert len(tools) == 2
    assert tools[0]["function"]["name"] == "a"
    assert tools[1]["function"]["name"] == "b"


def _make_response(content="Hello!", finish_reason="stop", tool_calls=None,
                   prompt_tokens=100, completion_tokens=20):
    """Build a mock LiteLLM response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def test_from_litellm_text_response():
    resp = _make_response(content="Hello!", finish_reason="stop")
    result = _from_litellm_response(resp)
    assert result["status"] == "ok"
    assert result["content"] == "Hello!"
    assert result["finish_reason"] == "stop"
    assert result["usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert "tool_calls" not in result


def test_from_litellm_tool_call_response():
    tc = MagicMock()
    tc.id = "tc_1"
    tc.function.name = "search"
    tc.function.arguments = json.dumps({"q": "weather"})

    resp = _make_response(content="Let me check.", finish_reason="tool_calls", tool_calls=[tc])
    result = _from_litellm_response(resp)
    assert result["content"] == "Let me check."
    assert result["finish_reason"] == "tool_call"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0] == {
        "id": "tc_1",
        "name": "search",
        "arguments": {"q": "weather"},
    }


def test_from_litellm_tool_call_dict_arguments():
    """LiteLLM sometimes returns arguments as dict instead of JSON string."""
    tc = MagicMock()
    tc.id = "tc_1"
    tc.function.name = "search"
    tc.function.arguments = {"q": "weather"}

    resp = _make_response(content="", finish_reason="tool_calls", tool_calls=[tc])
    result = _from_litellm_response(resp)
    assert result["tool_calls"][0]["arguments"] == {"q": "weather"}


def test_from_litellm_length_finish():
    resp = _make_response(content="Truncated...", finish_reason="length")
    result = _from_litellm_response(resp)
    assert result["finish_reason"] == "length"


def test_from_litellm_none_content():
    resp = _make_response(content=None, finish_reason="stop")
    result = _from_litellm_response(resp)
    assert result["content"] == ""


def test_from_litellm_multiple_tool_calls():
    tc1 = MagicMock()
    tc1.id = "tc_1"
    tc1.function.name = "search"
    tc1.function.arguments = json.dumps({"q": "a"})

    tc2 = MagicMock()
    tc2.id = "tc_2"
    tc2.function.name = "lookup"
    tc2.function.arguments = json.dumps({"id": "123"})

    resp = _make_response(content="", finish_reason="tool_calls", tool_calls=[tc1, tc2])
    result = _from_litellm_response(resp)
    assert len(result["tool_calls"]) == 2
    assert result["tool_calls"][0]["name"] == "search"
    assert result["tool_calls"][1]["name"] == "lookup"


# --- Opaque content: the worker composes, renders, elides, decides ---

from norns.client import (  # noqa: E402
    _compose_system_prompt,
    _elide_old_tool_results,
    _final_output,
    _render_message,
    _to_litellm_messages,
)


def test_compose_system_prompt_appends_summary_and_date():
    prompt = _compose_system_prompt({
        "system_prompt": "You help.",
        "summary": "User likes cats.",
        "date": "2026-09-09",
    })
    assert prompt == "You help.\n\nSummary of earlier conversation: User likes cats.\n\nCurrent date: 2026-09-09."


def test_compose_system_prompt_verbatim_without_envelope_fields():
    assert _compose_system_prompt({"system_prompt": "You help."}) == "You help."
    assert _compose_system_prompt({"system_prompt": "You help.", "summary": ""}) == "You help."


def test_render_message_passes_plain_messages_through():
    msg = {"role": "user", "content": "hi"}
    assert _render_message(msg) is msg


def test_render_inherited_context():
    out = _render_message({"role": "user", "kind": "inherited_context", "content": {"ticket_id": "T-123"}})
    assert out["role"] == "user"
    assert out["content"] == '[Inherited context from parent agent]\n{"ticket_id": "T-123"}'
    assert "kind" not in out


def test_render_system_results():
    base = {"role": "tool", "tool_call_id": "c1", "name": "x", "content": ""}
    cases = [
        (("timer_completed", {}, ""), "Timer completed."),
        (("tool_denied", {"tool_name": "send_email"}, ""), "Tool 'send_email' is not in this agent's allowed tools."),
        (("subagent_denied", {"agent_name": "kid", "reason": "disabled"}, ""), "This agent is not permitted to launch sub-agents."),
        (("subagent_denied", {"agent_name": "kid", "reason": "max_depth", "max_depth": 3}, ""),
            "Sub-agent nesting limit reached (max depth 3). Do the work in this agent instead of delegating further."),
        (("subagent_denied", {"agent_name": "kid", "reason": "not_allowlisted"}, ""), "Agent 'kid' is not in this agent's allowed sub-agents."),
        (("subagent_list_denied", {}, ""), "Listing agents is not permitted for this agent."),
        (("subagent_not_found", {"agent_name": "kid"}, ""), "Agent 'kid' not found"),
        (("subagent_self", {"agent_name": "me"}, ""), "Cannot launch self as a sub-agent"),
        (("subagent_missing", {"run_id": 42}, ""), "Sub-agent run 42 no longer exists, so its result cannot be recovered."),
        (("subagent_launch_failed", {"agent_name": "kid", "reason": ":busy"}, ""), "Failed to launch agent 'kid': :busy"),
    ]
    for (kind, data, content), expected in cases:
        out = _render_message({**base, "kind": kind, "data": data, "content": content})
        assert out["content"] == expected, kind


def test_render_subagent_outcomes_keep_content_and_envelope_apart():
    done = _render_message({"role": "tool", "kind": "subagent_completed", "data": {"run_id": 7, "status": "completed"}, "content": "42"})
    assert json.loads(done["content"]) == {"run_id": 7, "status": "completed", "output": "42"}
    failed = _render_message({"role": "tool", "kind": "subagent_failed", "data": {"run_id": 8, "status": "failed"}, "content": "boom"})
    assert json.loads(failed["content"]) == {"run_id": 8, "status": "failed", "error": "boom"}
    listed = _render_message({"role": "tool", "kind": "list_agents", "data": {"agents": [{"name": "a", "purpose": ""}]}, "content": ""})
    assert json.loads(listed["content"]) == [{"name": "a", "purpose": ""}]


def test_elide_old_tool_results_caps_only_aged_results():
    long = "x" * 500
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "t", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "c1", "content": long},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "name": "t", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "c2", "content": long},
    ]
    out = _elide_old_tool_results(messages)
    assert out[2]["content"] == "x" * 200 + "...(truncated)"
    assert out[4]["content"] == long  # recent results stay whole
    assert _elide_old_tool_results(messages[:4]) == messages[:4]  # four or fewer: untouched


def test_final_output_falls_back_to_last_substantive_assistant_turn():
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "Here's what I found.", "tool_calls": [{"id": "c1", "name": "t", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    assert _final_output(messages, "") == "Here's what I found."
    assert _final_output(messages, "Done.") == "Done."
    assert _final_output([{"role": "user", "content": "go"}], "  ") == "  "


def test_to_litellm_messages_encodes_structured_content_it_cannot_render():
    block = {"$enc": "v1", "kid": "k", "n": "n", "ct": "c"}
    out = _to_litellm_messages([{"role": "user", "content": block}])
    assert json.loads(out[0]["content"]) == block


def test_handle_llm_task_composes_renders_and_reports_final_output(monkeypatch):
    import asyncio
    from norns.client import Norns

    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _make_response(content="", finish_reason="stop")

    monkeypatch.setattr("norns.client.litellm.completion", fake_completion)
    client = Norns("http://localhost:4000", api_key="k")
    client._llm_provider = "anthropic"

    task = {
        "model": "claude-sonnet-5",
        "system_prompt": "You help.",
        "summary": "Prior chat.",
        "date": "2026-09-09",
        "messages": [
            {"role": "user", "content": "wait a sec"},
            {"role": "assistant", "content": "Waiting.", "tool_calls": [{"id": "c1", "name": "wait", "arguments": {"seconds": 1}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "wait", "kind": "timer_completed", "data": {}, "content": ""},
        ],
        "tools": [],
    }
    result = asyncio.run(client._handle_llm_task(task))

    assert captured["messages"][0] == {"role": "system", "content": "You help.\n\nSummary of earlier conversation: Prior chat.\n\nCurrent date: 2026-09-09."}
    assert captured["messages"][-1]["content"] == "Timer completed."
    assert result["status"] == "ok"
    assert result["final_output"] == "Waiting."


def test_compaction_prompt_and_messages():
    from norns.client import COMPACTION_INSTRUCTION, _compaction_messages, _compose_compaction_prompt

    task = {
        "purpose": "compact",
        "system_prompt": "You are Sleipnir.",
        "summary": "Earlier: fixed add().",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "tool", "tool_call_id": "c1", "name": "wait", "kind": "timer_completed", "data": {}, "content": ""},
        ],
    }
    assert _compose_compaction_prompt(task) == "You are Sleipnir.\n\nSummary of earlier conversation: Earlier: fixed add()."
    assert _compose_compaction_prompt({"system_prompt": "p"}) == "p"

    messages = _compaction_messages(task)
    assert messages[-1] == {"role": "user", "content": COMPACTION_INSTRUCTION}
    assert messages[1]["content"] == "Timer completed."
    assert "kind" not in messages[1]


def test_elision_skipped_when_core_manages_context():
    from norns.client import TOOL_RESULT_CAP, _messages_for_task

    big = "x" * (TOOL_RESULT_CAP + 50)
    messages = [
        {"role": "user", "content": "a"},
        {"role": "tool", "tool_call_id": "c1", "name": "t", "content": big},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    assert len(_messages_for_task({"messages": messages})[1]["content"]) < len(big)
    assert _messages_for_task({"messages": messages, "context_policy": {"compact_at": 1, "keep": 1}})[1]["content"] == big


def test_agent_registers_context_policy():
    from norns.agent import Agent

    agent = Agent(name="a", context_policy={"compact_at": 100_000, "keep": 30}, context_strategy="none")
    assert agent.to_registration()["context_policy"] == {"compact_at": 100_000, "keep": 30}
    assert Agent(name="b").to_registration()["context_policy"] is None
