# norns-sdk

[![CI](https://github.com/nornscode/norns-sdk-python/actions/workflows/ci.yml/badge.svg)](https://github.com/nornscode/norns-sdk-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/norns-sdk)](https://pypi.org/project/norns-sdk/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Python SDK for [Norns](https://github.com/nornscode/norns).

```bash
pip install norns-sdk
```

The SDK has two parts. `Norns` is the worker — it connects to the server, registers your agent, and sits in a loop handling tasks. `NornsClient` is for sending messages and reading results from application code (your Slack bot, web backend, CLI, etc).

## Worker

```python
import os
from norns import Norns, Agent, tool

@tool
def search_docs(query: str) -> str:
    """Search product documentation."""
    return db.vector_search(query)

@tool(side_effect=True)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a customer."""
    smtp.send(to=to, subject=subject, body=body)
    return f"Email sent to {to}"

agent = Agent(
    name="support-bot",
    model="claude-sonnet-5",
    system_prompt="You are a customer support agent. Look up docs and help customers.",
    tools=[search_docs, send_email],
    mode="conversation",
    on_failure="retry_last_step",
)

norns = Norns("http://localhost:4000", api_key=os.environ["NORNS_API_KEY"])
norns.run(agent)  # LLM API keys read from env (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
```

`norns.run()` connects via WebSocket, registers the agent and tools, then blocks forever handling `llm_task` and `tool_task` dispatches. LLM calls go through [LiteLLM](https://github.com/BerriAI/litellm), so any supported provider works. Norns never sees your API keys — your worker makes all external calls.

### Gards

A [gard](https://github.com/nornscode/norns/blob/main/docs/gards.md) pins all of a run's tool dispatch to one worker — worker affinity for coding agents and other filesystem-bound work. Create one (`nornsctl gards create` prints the claim token once), then claim it:

```python
norns.run(agent, gard=3, claim_token="tok_...")
```

A worker in a gard serves only runs bound to that gard, and vice versa. Tool handlers can expose service ports for the dashboard — the gard is inferred from the connection:

```python
norns.register_port(3000, name="react", url="http://localhost:3000")
```

A fatally rejected claim (bad token, destroyed gard) raises `JoinError` instead of reconnect-looping; `GardDestroyed` is raised if the gard is destroyed while the worker is connected.

### Shutdown

On SIGTERM or SIGINT the worker drains instead of dying mid-task: it tells Norns to stop sending it work, finishes the tasks it already holds, reports their results, leaves the channel, and `run()` returns. Tasks still running after `shutdown_timeout` seconds (default 30, or `NORNS_SHUTDOWN_TIMEOUT`) are dropped and Norns re-dispatches them. A second signal exits immediately.

```python
norns.run(agent, shutdown_timeout=60)
```

`norns.shutdown()` requests the same drain from code — a tool handler or another thread can call it. New work that arrives while a worker drains queues until its replacement connects, so a connector restarted by a supervisor loses nothing.

## Client

```python
import os
from norns import NornsClient

client = NornsClient("http://localhost:4000", api_key=os.environ["NORNS_API_KEY"])

# Fire-and-forget
run = client.send_message("support-bot", "Where's my order?")
# run.run_id, run.status == "accepted"

# Wait for completion
result = client.send_message("support-bot", "Where's my order?", wait=True, timeout=30)
print(result.output)

# Multi-turn with a conversation key
result = client.send_message("support-bot", "And the tracking number?",
                             conversation_key="slack:U01ABC", wait=True)

# Inspect a run
run = client.get_run(42)
events = client.get_events(42)

# Stream events as they happen
for event in client.stream("support-bot", "Research quantum computing"):
    if event.type == "completed":
        print(event.data.get("output", "")[:80])
        break
```

## Human-in-the-loop

An agent can call the built-in `ask_human` tool to pause and ask a question. The run parks with status `"waiting"` until someone answers, and survives a restart while parked.

```python
result = client.send_message("support-bot", "Book me a table", wait=True)

if result.is_waiting:
    print(result.waiting_for.question)      # "7pm or 8pm?"
    client.reply(result.run_id, "7pm")
```

`wait=True` returns as soon as the agent parks — it's waiting on you, so it won't progress on its own. Sending the agent another message answers the question too, which is usually what a chat or Slack client wants; `reply()` targets one specific run.

## Tools

The `@tool` decorator infers JSON Schema from type hints. The docstring becomes the tool description the LLM sees.

```python
@tool
def lookup_customer(email: str) -> str:
    """Look up a customer by email."""
    customer = db.query("SELECT * FROM customers WHERE email = ?", email)
    return f"Found: {customer['name']} ({customer['plan']})"
```

Mark side-effecting tools so Norns can enforce idempotency on replay:

```python
@tool(side_effect=True)
def charge_card(customer_id: str, amount: float) -> str:
    """Charge a credit card."""
    result = stripe.charges.create(customer=customer_id, amount=int(amount * 100))
    return f"Charged ${amount}: {result['id']}"
```

Async handlers work too:

```python
@tool
async def fetch_page(url: str) -> str:
    """Fetch a web page."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()
```

## Agent options

```python
agent = Agent(
    name="my-agent",
    model="claude-sonnet-5",
    system_prompt="You are helpful.",
    tools=[search, send_email],
    mode="conversation",             # "task" or "conversation"
    checkpoint_policy="on_tool_call",  # "every_step", "on_tool_call", "manual"
    context_window=20,
    max_steps=50,
    on_failure="retry_last_step",    # "stop" or "retry_last_step"
    max_tokens=8192,                 # ceiling on one response
)
```

`max_tokens` is the ceiling on a single response, not on the history.
Leave it unset and the worker uses 8192; raise it for an agent whose
turns are long, such as one writing a whole file in a single turn. A
turn that reaches the ceiling comes back truncated — the run completes,
and the `llm_response` event carries `finish_reason: "length"` so a
client can say so.

## Docs

- [Messaging client design](docs/messaging-client-design.md)
- [Remaining work](docs/remaining-work.md)

## License

MIT
