# SDK Remaining Work

## What's implemented

### Worker (`Norns` class) ✓
- WebSocket connection to `/worker` with Phoenix channel protocol
- Agent + tool registration on join
- `llm_task` handling (calls Anthropic API)
- `tool_task` handling (calls local tool functions, sync + async)
- Heartbeat, auto-reconnect
- Graceful shutdown: SIGTERM/SIGINT or `shutdown()` drains (Norns `drain` event), finishes in-flight tasks up to a deadline, leaves
- `@tool` decorator with JSON Schema inference from type hints
- `Agent` dataclass with all AgentDef fields

### Client (`NornsClient` class) ✓
- HTTP client with auth (`httpx`)
- `list_agents()`, `get_agent(id_or_name)`
- `send_message()` with fire-and-forget and `wait=True` (polling)
- `get_run()`, `get_events()`
- `list_conversations()`, `get_conversation()`, `delete_conversation()`
- `stream()` via WebSocket (generator yielding `StreamEvent`)
- Response models (`RunResponse`, `EventResponse`, etc.)
- Error handling, context manager

### Tests ✓
- 29 tests passing (agent, schema, client with HTTP mocking)

---

## What's NOT implemented yet

### Server-side (Norns runtime repo)

**1. Agent lookup by name** — NICE TO HAVE

The client resolves names by listing all agents and filtering. This works but is O(n). Add `GET /api/v1/agents?name=support-bot` or make `GET /api/v1/agents/:id` accept a name string.

### SDK (this repo)

**3. End-to-end integration test** — IMPORTANT

No test actually connects to a running Norns instance. We have unit tests with HTTP mocking, but no proof that the SDK talks to the real server correctly. Need:
- A test that starts Norns (via docker compose), creates a tenant, runs a worker, sends a message via the client, and verifies the result
- Can be a separate test suite that requires `NORNS_URL` to be set

**4. Worker: handle agent registration response** — MINOR

The worker sends agents in the join payload but doesn't check if the server accepted them. Should validate the join reply and log/raise if registration failed.

**5. Worker: rate limit handling** — MINOR

When the Anthropic API returns 429, the worker returns the error to the orchestrator. But the worker could handle retries locally (with backoff) instead of pushing the problem back to the orchestrator. This would make the rate limit invisible to Norns.

**7. Pydantic support for tool schemas** — NICE TO HAVE

Currently schemas are inferred from type hints only. If a user passes a Pydantic model as a type hint, the SDK should use `model.model_json_schema()` to generate the schema. Check if Pydantic is installed and use it opportunistically.

**8. Multiple agents per worker** — NICE TO HAVE

Currently `norns.run(agent)` takes one agent. Should support `norns.run([agent1, agent2])` for workers that handle multiple agent types.

### Documentation

**9. Publish to PyPI** — when ready for external users

**10. Add examples directory** — concrete examples:
- `examples/slack_bot.py` — Slack integration
- `examples/research_agent.py` — multi-step research
- `examples/support_bot.py` — customer support with tools

---

## Priority order

1. **End-to-end integration test** (proves the whole thing works)
2. **Worker graceful shutdown** (SIGINT handling)
3. **Examples** (shows people how to use it)
4. Everything else
