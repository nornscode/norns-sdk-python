"""Norns client — worker and client for the Norns durable agent runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from collections import OrderedDict
from collections.abc import Generator
from typing import Any

import httpx
import litellm
import websockets

from norns.agent import Agent, ToolDef
from norns.models import (
    AgentResponse,
    ConversationResponse,
    EventResponse,
    MessageResult,
    RunResponse,
    StreamEvent,
    WaitingFor,
)

MAX_REMEMBERED_RESULTS = 512

logger = logging.getLogger("norns")


class JoinError(Exception):
    """The worker channel join was rejected for a non-retryable reason
    (invalid claim token, destroyed or missing gard, bad registration)."""


class JoinRetryable(Exception):
    """The join failed for a reason a reconnect can fix (e.g. the previous
    incarnation's disconnect hasn't been processed yet)."""


class GardDestroyed(Exception):
    """The gard this worker claimed was destroyed while it was connected."""


class Norns:
    """Client for the Norns durable agent runtime.

    Usage:
        norns = Norns("http://localhost:4000", api_key="nrn_...")

        @tool
        def search(query: str) -> str:
            ...

        agent = Agent(name="bot", tools=[search], ...)
        norns.run(agent, llm_api_key="sk-ant-...")
    """

    def __init__(self, url: str, *, api_key: str | None = None):
        self.url = url.rstrip("/")
        self.api_key = api_key or os.environ.get("NORNS_API_KEY", "")
        self._ws_url = self.url.replace("http://", "ws://").replace("https://", "wss://")
        self._gard: str | int | None = None
        self._claim_token: str | None = None
        self._ws = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown: asyncio.Event | None = None
        self._shutdown_timeout = 30.0
        # Results of side-effecting calls, by the idempotency key core issued.
        # Bounded: a worker lives for days and this is a crash guard, not a
        # cache, so the oldest entries are the ones least likely to be asked
        # for again.
        self._completed: OrderedDict[str, dict] = OrderedDict()

    def run(
        self,
        agent: Agent,
        *,
        llm_api_key: str | None = None,
        worker_id: str | None = None,
        gard: str | int | None = None,
        claim_token: str | None = None,
        shutdown_timeout: float | None = None,
    ):
        """Connect as a worker, register the agent, and handle tasks until
        told to stop.

        Auto-creates the agent via REST if it doesn't exist yet.
        This blocks — like a Temporal worker.

        On SIGTERM or SIGINT (or `shutdown()`) the worker drains: it tells
        Norns to stop sending it work, finishes the tasks it already holds
        — up to `shutdown_timeout` seconds (default 30, or
        NORNS_SHUTDOWN_TIMEOUT) — reports their results, leaves the channel,
        and returns. Tasks still running at the deadline are dropped; Norns
        re-dispatches them. A second signal exits immediately.

        Pass gard and claim_token (both from POST /api/v1/gards, usually
        handed to the worker by the provisioner) to claim a gard: all tool
        dispatch for runs bound to that gard comes to this worker, and only
        to it. A worker without a gard only serves no-gard runs.

        LLM API keys are read from environment variables by LiteLLM
        (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.). The llm_api_key
        parameter is accepted for backwards compatibility but ignored.
        """
        # Suppress noisy LiteLLM logs
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self._ensure_agent(agent)
        # NORNS_GARD / NORNS_GARD_CLAIM_TOKEN mirror NORNS_WORKER_ID below:
        # a provisioner (volund) hands the worker its gard through the
        # environment so scaffolded workers need no code changes.
        self._gard = gard or os.environ.get("NORNS_GARD")
        self._claim_token = claim_token or os.environ.get("NORNS_GARD_CLAIM_TOKEN")

        # NORNS_WORKER_ID lets a provisioner give the worker a stable
        # identity (volund sets it to the deployment name) so operators
        # can join "container running" with "worker connected".
        wid = (
            worker_id
            or os.environ.get("NORNS_WORKER_ID")
            or f"python-worker-{uuid.uuid4().hex[:8]}"
        )

        if shutdown_timeout is None:
            shutdown_timeout = float(os.environ.get("NORNS_SHUTDOWN_TIMEOUT", "30"))
        self._shutdown_timeout = shutdown_timeout

        try:
            asyncio.run(self._run_loop(agent, wid))
        except KeyboardInterrupt:
            # Only reachable where signal handlers can't be installed
            # (Windows, non-main thread): there is no drain, just exit.
            logger.info("Worker shutting down.")

    def shutdown(self):
        """Ask a running worker to drain and stop. Safe from any thread —
        a tool handler can call it, and so can a test. `run()` returns
        once the drain completes."""
        if self._loop is None or self._shutdown is None:
            return
        self._loop.call_soon_threadsafe(self._shutdown.set)

    def register_port(
        self,
        internal_port: int,
        *,
        name: str = "",
        url: str | None = None,
        protocol: str = "http",
    ):
        """Register a service port with Norns for dashboard visibility.

        Requires the worker to be running in a gard — the gard is inferred
        from the connection, so there's nothing to re-specify here. Call this
        from a (sync) tool handler after starting a service; it's
        best-effort, and a rejected registration (e.g. a disallowed URL
        scheme) is logged by the message loop rather than raised.
        """
        if self._gard is None:
            raise RuntimeError("register_port requires the worker to be running in a gard")
        if self._ws is None or self._loop is None:
            raise RuntimeError("register_port requires a connected worker (call from a tool handler)")

        payload = {"internal_port": internal_port, "name": name, "protocol": protocol}
        if url:
            payload["url"] = url

        future = asyncio.run_coroutine_threadsafe(
            self._push("register_port", payload), self._loop
        )
        future.result(timeout=10)

    def _ensure_agent(self, agent: Agent):
        """Create or update the agent via REST API.

        If the agent already exists, updates it to match the current definition.
        This ensures code changes (system_prompt, model, etc.) are always picked up.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        body = {
            "name": agent.name,
            "system_prompt": agent.system_prompt,
            "status": "idle",
            "model": agent.model,
            "max_steps": agent.max_steps,
            "model_config": {
                "mode": agent.mode,
                "checkpoint_policy": agent.checkpoint_policy,
                "context_strategy": agent.context_strategy,
                "context_window": agent.context_window,
                "on_failure": agent.on_failure,
                "context_policy": agent.context_policy,
                "max_tokens": agent.max_tokens,
            },
        }

        with httpx.Client(base_url=self.url, headers=headers) as client:
            resp = client.get("/api/v1/agents")
            resp.raise_for_status()
            existing = resp.json().get("data", [])

            for a in existing:
                if a["name"] == agent.name:
                    resp = client.put(f"/api/v1/agents/{a['id']}", json=body)
                    resp.raise_for_status()
                    logger.info(f"Updated agent '{agent.name}' (id={a['id']})")
                    return

            resp = client.post("/api/v1/agents", json=body)
            resp.raise_for_status()
            created = resp.json()["data"]
            logger.info(f"Created agent '{agent.name}' (id={created['id']})")

    async def _run_loop(self, agent: Agent, worker_id: str):
        """Main event loop: connect, register, handle tasks, reconnect on
        failure, until shutdown is requested."""
        tools_by_name = {t.name: t for t in agent.tools}
        self._llm_provider = agent.llm_provider
        self._loop = asyncio.get_running_loop()
        self._shutdown = asyncio.Event()
        installed = self._install_signal_handlers()

        try:
            while not self._shutdown.is_set():
                try:
                    await self._connect_and_serve(agent, worker_id, tools_by_name)
                except (JoinError, GardDestroyed):
                    # Retrying can never succeed — surface it instead of spinning.
                    raise
                except JoinRetryable as e:
                    if self._shutdown.is_set():
                        break
                    logger.warning(f"{e}. Retrying in 3s...")
                    await self._sleep_unless_shutdown(3)
                except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                    if self._shutdown.is_set():
                        break
                    logger.warning(f"Connection lost: {e}. Reconnecting in 3s...")
                    await self._sleep_unless_shutdown(3)
                except Exception as e:
                    if self._shutdown.is_set():
                        break
                    logger.error(f"Unexpected error: {e}. Reconnecting in 5s...")
                    await self._sleep_unless_shutdown(5)
        finally:
            for sig in installed:
                self._loop.remove_signal_handler(sig)
            self._loop = None
            self._shutdown = None

        logger.info("Worker stopped.")

    def _install_signal_handlers(self) -> list[signal.Signals]:
        """Route SIGTERM/SIGINT to a drain. Returns the signals hooked;
        empty where the loop can't do it (Windows, non-main thread)."""
        loop = asyncio.get_running_loop()
        installed = []
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._on_signal, sig)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            installed.append(sig)
        return installed

    def _on_signal(self, sig: signal.Signals):
        logger.info(f"Received {sig.name}: shutting down")
        if self._shutdown is not None:
            self._shutdown.set()
        # A second delivery gets the default action (exit now).
        try:
            self._loop.remove_signal_handler(sig)
            signal.signal(sig, signal.SIG_DFL)
        except Exception:
            pass

    async def _sleep_unless_shutdown(self, seconds: float):
        try:
            await asyncio.wait_for(self._shutdown.wait(), seconds)
        except asyncio.TimeoutError:
            pass

    def _join_payload(self, agent: Agent, worker_id: str) -> dict:
        payload = {
            "worker_id": worker_id,
            "tools": [t.to_registration() for t in agent.tools],
            "capabilities": ["llm", "tools"],
            "agents": [agent.to_registration()],
        }

        if self._gard is not None:
            payload["gard"] = self._gard
            payload["claim_token"] = self._claim_token

        return payload

    async def _connect_and_serve(
        self,
        agent: Agent,
        worker_id: str,
        tools_by_name: dict[str, ToolDef],
    ):
        """Single connection lifecycle: connect, join, handle messages."""
        ws_url = f"{self._ws_url}/worker/websocket?token={self.api_key}&vsn=2.0.0"

        async with websockets.connect(ws_url) as ws:
            logger.info(f"Connected to {self.url}")
            self._ref_counter = 1

            # Phoenix channel join
            join_msg = json.dumps(
                [None, "1", "worker:lobby", "phx_join", self._join_payload(agent, worker_id)]
            )
            await ws.send(join_msg)

            response = await ws.recv()
            resp_data = json.loads(response)
            self._check_join_reply(resp_data)

            if self._gard is not None:
                logger.info(f"Worker {worker_id} ready (gard {self._gard})")
            else:
                logger.info(f"Worker {worker_id} ready")

            self._ws = ws

            # Heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat(ws))

            # Each task runs as its own asyncio task: a slow tool or LLM
            # call must not block the receive loop, or every other task on
            # this connection (parallel tool calls, other runs) serializes
            # behind it.
            in_flight: set[asyncio.Task] = set()

            def spawn(coro):
                task = asyncio.create_task(coro)
                in_flight.add(task)
                task.add_done_callback(in_flight.discard)

            # Drains and closes the socket once shutdown is requested; the
            # receive loop below then ends because the connection is gone.
            drain_task = asyncio.create_task(self._drain_on_shutdown(ws, in_flight))

            try:
                async for raw_msg in ws:
                    msg = json.loads(raw_msg)
                    # Phoenix message format: [join_ref, ref, topic, event, payload]
                    if not isinstance(msg, list) or len(msg) < 5:
                        continue

                    _join_ref, _ref, _topic, event, payload = msg

                    if event == "llm_task":
                        spawn(self._run_llm_task(ws, payload))

                    elif event == "tool_task":
                        spawn(self._run_tool_task(ws, payload, tools_by_name))

                    elif event == "phx_reply" and isinstance(payload, dict) and payload.get("status") == "error":
                        # e.g. a rejected register_port — surface it, don't die
                        logger.warning(f"Server rejected a push: {payload.get('response')}")

                    elif event == "gard_destroyed":
                        logger.error("This worker's gard was destroyed — shutting down")
                        raise GardDestroyed(f"gard {self._gard} was destroyed")

                    elif event == "phx_error":
                        logger.error(f"Channel error: {payload}")
                        break

                    elif event == "phx_close":
                        logger.info("Channel closed by server")
                        break

            finally:
                self._ws = None
                heartbeat_task.cancel()
                drain_task.cancel()
                # In-flight results can't be delivered on the next
                # connection — the orchestrator re-dispatches on disconnect
                # and idempotency skips completed side effects. (After a
                # drain this set is empty unless the deadline passed.)
                for task in in_flight:
                    task.cancel()

    async def _drain_on_shutdown(self, ws, in_flight: set[asyncio.Task]):
        if self._shutdown is None:  # serving outside run(): nothing to wait for
            return
        await self._shutdown.wait()

        logger.info(
            f"Draining: {len(in_flight)} in-flight task(s), "
            f"up to {self._shutdown_timeout:g}s"
        )
        # Tell Norns to stop dispatching to us. In-flight tasks stay ours;
        # new ones queue for our replacement.
        try:
            await self._push("drain", {})
        except Exception as e:
            logger.warning(f"Could not send drain ({e}); Norns will re-dispatch on disconnect")

        # Wait for what we hold, including anything that raced in before
        # the drain took effect, until the deadline.
        deadline = time.monotonic() + self._shutdown_timeout
        while in_flight:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.wait(set(in_flight), timeout=remaining)

        if in_flight:
            logger.warning(
                f"{len(in_flight)} task(s) still running at the deadline; "
                "dropping them — Norns will re-dispatch"
            )
            for task in in_flight:
                task.cancel()
        else:
            logger.info("Drained; leaving")

        try:
            await self._push("phx_leave", {})
            await ws.close()
        except Exception as e:
            logger.debug(f"Error while leaving: {e}")

    def _check_join_reply(self, msg):
        """Raise on a failed channel join instead of pretending we're ready.

        A gard claim rejection arrives here. `already_claimed` is retryable —
        on a quick reconnect, the server may not have processed the old
        connection's disconnect yet. Everything else (bad token, destroyed or
        missing gard) is fatal: retrying can never succeed.
        """
        if not isinstance(msg, list) or len(msg) < 5:
            return

        payload = msg[4]
        if not isinstance(payload, dict) or payload.get("status") != "error":
            return

        response = payload.get("response") or {}
        reason = response.get("reason", "join rejected")

        if reason == "already_claimed":
            raise JoinRetryable(f"gard claim: {reason}")

        raise JoinError(f"worker join rejected: {reason}")

    async def _push(self, event: str, payload: dict):
        """Send a channel push on the active connection."""
        self._ref_counter += 1
        msg = json.dumps(["1", str(self._ref_counter), "worker:lobby", event, payload])
        await self._ws.send(msg)

    async def _run_llm_task(self, ws, payload: dict):
        """Handle one llm_task to completion: execute, log, send the result."""
        tools_list = [t.get("name") for t in payload.get("tools", [])]
        logger.info(
            f"LLM call → {payload.get('model', '?')} "
            f"({len(payload.get('messages', []))} messages, {len(tools_list)} tools)"
        )
        result = await self._handle_llm_task(payload)
        finish = result.get("finish_reason", result.get("status", "?"))
        logger.info(f"LLM done → {finish}")
        await self._try_send_result(ws, payload, result)

    async def _run_tool_task(self, ws, payload: dict, tools_by_name: dict[str, ToolDef]):
        """Handle one tool_task to completion: execute, log, send the result."""
        tool_name = payload.get("tool_name", "?")
        logger.info(f"Tool call → {tool_name}")
        result = await self._handle_tool_task(payload, tools_by_name)
        status = result.get("status", "?")
        preview = str(result.get("result", result.get("error", "")))[:80]
        logger.info(f"Tool done → {tool_name}: {status} {preview}")
        await self._try_send_result(ws, payload, result)

    async def _try_send_result(self, ws, task: dict, result: dict):
        """Send a result, tolerating a connection that died mid-task."""
        try:
            await self._send_result(ws, task, result)
        except Exception as e:
            logger.warning(
                f"Could not deliver result for task {task.get('task_id', '?')} "
                f"({e}); the orchestrator re-dispatches it on reconnect"
            )

    async def _handle_llm_task(self, task: dict) -> dict:
        """Execute an LLM call via LiteLLM.

        Receives provider-neutral format from Norns, calls the LLM via LiteLLM
        (which handles provider-specific translation), and returns the result
        in neutral format.
        """
        try:
            raw_model = task.get("model", "claude-sonnet-5")
            # LiteLLM expects provider/model format
            if "/" not in raw_model:
                model = f"{self._llm_provider}/{raw_model}"
            else:
                model = raw_model
            if task.get("purpose") == "compact":
                return await self._handle_compaction_task(task, model)

            # Core sends the def's prompt verbatim and never writes prose for
            # the model: the worker composes the prompt, renders kinded
            # messages, elides old tool results, and decides the final output.
            system_prompt = _compose_system_prompt(task)
            messages = _messages_for_task(task)
            tools = task.get("tools", [])

            # Prepend system prompt as a system message
            llm_messages: list[dict] = []
            if system_prompt:
                llm_messages.append({"role": "system", "content": system_prompt})
            llm_messages.extend(_to_litellm_messages(messages))

            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": _max_tokens(task),
                "messages": llm_messages,
            }
            if tools:
                kwargs["tools"] = _to_litellm_tools(tools)

            response = await asyncio.to_thread(litellm.completion, **kwargs)
            result = _from_litellm_response(response)
            result["final_output"] = _final_output(messages, result.get("content", ""))
            return result

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {"status": "error", "error": str(e)}

    async def _handle_compaction_task(self, task: dict, model: str) -> dict:
        """Fold the history core sent into a summary it will carry forward.

        Core never reads the messages or the summary; this worker writes
        the prose and returns it as content.
        """
        llm_messages: list[dict] = []
        system_prompt = _compose_compaction_prompt(task)
        if system_prompt:
            llm_messages.append({"role": "system", "content": system_prompt})
        llm_messages.extend(_to_litellm_messages(_compaction_messages(task)))

        response = await asyncio.to_thread(
            litellm.completion, model=model, max_tokens=_max_tokens(task), messages=llm_messages
        )
        result = _from_litellm_response(response)
        return {
            "status": "ok",
            "content": result.get("content", ""),
            "finish_reason": "stop",
            "usage": result.get("usage", {}),
        }

    async def _handle_tool_task(self, task: dict, tools: dict[str, ToolDef]) -> dict:
        """Execute a tool call, unless we already did this exact one.

        A key on the task means core considers the call side-effecting and
        has named it: same run, same step, same tool call, same key, however
        many times it is dispatched. Core cannot know whether the effect
        landed — if the result never reached it, it re-dispatches — so this
        worker is the only party that can answer, and it answers from what
        it kept rather than doing the thing twice.
        """
        tool_name = task.get("tool_name", task.get("name", ""))
        input_data = task.get("input", {})
        key = task.get("idempotency_key")

        if key and key in self._completed:
            logger.info(f"Tool {tool_name}: already done ({key}), reusing the result")
            return dict(self._completed[key])

        tool = tools.get(tool_name)
        if tool is None:
            return {"status": "error", "error": f"Unknown tool: {tool_name}"}

        try:
            # Run the handler — support both sync and async
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**input_data)
            else:
                result = await asyncio.to_thread(tool.handler, **input_data)

            return self._remember(key, {"status": "ok", "result": str(result)})

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            # Failures are not remembered: a retry of something that did not
            # happen should happen.
            return {"status": "error", "error": str(e)}

    def _remember(self, key: str | None, result: dict) -> dict:
        if not key:
            return result

        self._completed[key] = dict(result, duplicate=True)
        while len(self._completed) > MAX_REMEMBERED_RESULTS:
            self._completed.popitem(last=False)

        return result

    async def _send_result(self, ws, task: dict, result: dict):
        """Send a task result back to the orchestrator."""
        task_id = task.get("task_id", "")
        result["task_id"] = task_id

        self._ref_counter += 1
        ref = str(self._ref_counter)
        msg = json.dumps(["1", ref, "worker:lobby", "tool_result", result])
        await ws.send(msg)
        logger.debug(f"Sent result for task {task_id}: {result.get('status')}")

    async def _heartbeat(self, ws):
        """Send Phoenix heartbeat to keep the connection alive."""
        ref = 100
        while True:
            await asyncio.sleep(30)
            ref += 1
            msg = json.dumps([None, str(ref), "phoenix", "heartbeat", {}])
            try:
                await ws.send(msg)
            except Exception:
                break


class NornsClient:
    """Client for interacting with Norns agents.

    This is the client — it sends messages and queries results.
    For running a worker, use the Norns class instead.

    Usage:
        client = NornsClient("http://localhost:4000", api_key="nrn_...")
        run = client.send_message("support-bot", "Hello!")
        result = client.send_message("support-bot", "Hello!", wait=True)
    """

    def __init__(self, url: str, *, api_key: str | None = None):
        self.base_url = url.rstrip("/")
        self.api_key = api_key or os.environ.get("NORNS_API_KEY", "")
        self._ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def close(self):
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make an authenticated HTTP request."""
        response = self._client.request(method, path, **kwargs)
        response.raise_for_status()
        return response

    # --- Agent Management ---

    def list_agents(self) -> list[AgentResponse]:
        """List all agents."""
        resp = self._request("GET", "/api/v1/agents")
        return [_parse_agent(a) for a in resp.json()["data"]]

    def get_agent(self, id_or_name: int | str) -> AgentResponse:
        """Get an agent by ID or name.

        If a string is passed, resolves by listing agents and filtering by name.
        """
        if isinstance(id_or_name, int):
            resp = self._request("GET", f"/api/v1/agents/{id_or_name}")
            return _parse_agent(resp.json()["data"])

        agents = self.list_agents()
        for agent in agents:
            if agent.name == id_or_name:
                return agent
        raise ValueError(f"Agent not found: {id_or_name}")

    def _resolve_agent_id(self, agent: int | str) -> int:
        """Resolve an agent identifier to an integer ID."""
        if isinstance(agent, int):
            return agent
        return self.get_agent(agent).id

    # --- Sending Messages ---

    def send_message(
        self,
        agent: int | str,
        content: str,
        *,
        conversation_key: str | None = None,
        wait: bool = False,
        timeout: float = 30,
    ) -> MessageResult:
        """Send a message to an agent.

        Args:
            agent: Agent ID (int) or name (str).
            content: The message content.
            conversation_key: Optional key for multi-turn conversations.
            wait: If True, poll until the run completes or times out.
            timeout: Seconds to wait when wait=True.

        Returns:
            MessageResult with run_id, status, and output (if wait=True and completed).
        """
        agent_id = self._resolve_agent_id(agent)
        body: dict[str, Any] = {"content": content}
        if conversation_key is not None:
            body["conversation_key"] = conversation_key

        resp = self._request("POST", f"/api/v1/agents/{agent_id}/messages", json=body)
        data = resp.json()
        run_id = data["run_id"]
        status = data.get("status", "accepted")

        if not wait:
            return MessageResult(
                run_id=run_id,
                status=status,
                output=None,
                conversation_key=conversation_key,
            )

        # Poll until the run stops progressing, or we time out.
        # "waiting" is not terminal, but the agent is parked on a question and
        # won't move until someone answers — so it stops the wait too.
        deadline = time.monotonic() + timeout
        poll_interval = 0.5
        while time.monotonic() < deadline:
            run = self.get_run(run_id)
            if run.status in ("completed", "failed", "error", "waiting"):
                return MessageResult(
                    run_id=run_id,
                    status=run.status,
                    output=run.output,
                    conversation_key=conversation_key,
                    waiting_for=run.waiting_for,
                )
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 3.0)

        raise TimeoutError(f"Run {run_id} did not complete within {timeout}s")

    # --- Run Inspection ---

    def get_run(self, run_id: int) -> RunResponse:
        """Get details of a run."""
        resp = self._request("GET", f"/api/v1/runs/{run_id}")
        data = resp.json()["data"]
        return RunResponse(
            run_id=data["id"],
            status=data["status"],
            output=data.get("output"),
            agent_id=data["agent_id"],
            conversation_id=data.get("conversation_id"),
            trigger_type=data.get("trigger_type", "message"),
            inserted_at=data["inserted_at"],
            waiting_for=WaitingFor.from_dict(data.get("waiting_for")),
        )

    def reply(self, run_id: int, answer: str) -> None:
        """Answer a run parked on an ``ask_human`` question.

        Sending the agent a normal message with ``send_message()`` does the
        same thing and is usually what a conversational client wants. Use
        ``reply()`` to answer one specific run — for example when an agent has
        several conversations parked at once.

        Example:
            result = client.send_message("support-bot", "Book it", wait=True)
            if result.is_waiting:
                print(result.waiting_for.question)
                client.reply(result.run_id, "yes, go ahead")
        """
        self._request("POST", f"/api/v1/runs/{run_id}/reply", json={"answer": answer})

    def get_events(self, run_id: int) -> list[EventResponse]:
        """Get the event log for a run."""
        resp = self._request("GET", f"/api/v1/runs/{run_id}/events")
        return [
            EventResponse(
                id=e["id"],
                sequence=e["sequence"],
                event_type=e["event_type"],
                payload=e.get("payload", {}),
                source=e.get("source", ""),
                inserted_at=e["inserted_at"],
            )
            for e in resp.json()["data"]
        ]

    # --- Conversations ---

    def list_conversations(self, agent: int | str) -> list[ConversationResponse]:
        """List conversations for an agent."""
        agent_id = self._resolve_agent_id(agent)
        resp = self._request("GET", f"/api/v1/agents/{agent_id}/conversations")
        return [
            ConversationResponse(
                id=c["id"],
                agent_id=c["agent_id"],
                key=c["key"],
                message_count=c.get("message_count", 0),
                token_estimate=c.get("token_estimate", 0),
            )
            for c in resp.json()["data"]
        ]

    def get_conversation(self, agent: int | str, key: str) -> ConversationResponse:
        """Get a specific conversation by key."""
        agent_id = self._resolve_agent_id(agent)
        resp = self._request("GET", f"/api/v1/agents/{agent_id}/conversations/{key}")
        c = resp.json()["data"]
        return ConversationResponse(
            id=c["id"],
            agent_id=c["agent_id"],
            key=c["key"],
            message_count=c.get("message_count", 0),
            token_estimate=c.get("token_estimate", 0),
        )

    def delete_conversation(self, agent: int | str, key: str) -> None:
        """Delete a conversation (reset)."""
        agent_id = self._resolve_agent_id(agent)
        self._request("DELETE", f"/api/v1/agents/{agent_id}/conversations/{key}")

    # --- Streaming ---

    def stream(
        self,
        agent: int | str,
        content: str,
        *,
        conversation_key: str | None = None,
        timeout: float = 120,
    ) -> Generator[StreamEvent, None, None]:
        """Send a message and stream events as they happen.

        Yields StreamEvent objects until the run completes or errors.
        """
        agent_id = self._resolve_agent_id(agent)

        # Send the message first (fire-and-forget)
        body: dict[str, Any] = {"content": content}
        if conversation_key is not None:
            body["conversation_key"] = conversation_key
        resp = self._request("POST", f"/api/v1/agents/{agent_id}/messages", json=body)
        run_id = resp.json()["run_id"]

        # Stream via WebSocket
        yield from _stream_events(self._ws_url, self.api_key, agent_id, run_id, timeout)


def _stream_events(
    ws_url: str,
    api_key: str,
    agent_id: int,
    run_id: int,
    timeout: float,
) -> Generator[StreamEvent, None, None]:
    """Connect to Phoenix WebSocket and yield events for a run."""
    url = f"{ws_url}/socket/websocket?token={api_key}&vsn=2.0.0"
    topic = f"agent:{agent_id}"
    ref_counter = 0

    def next_ref() -> str:
        nonlocal ref_counter
        ref_counter += 1
        return str(ref_counter)

    with websockets.sync.client.connect(url) as ws:
        ws.settimeout(timeout)

        # Join the agent channel
        join_ref = next_ref()
        join_msg = json.dumps([join_ref, next_ref(), topic, "phx_join", {"run_id": run_id}])
        ws.send(join_msg)

        # Wait for join reply
        reply = json.loads(ws.recv())
        if isinstance(reply, list) and len(reply) >= 5 and reply[3] == "phx_reply":
            status = reply[4].get("status")
            if status != "ok":
                raise ConnectionError(f"Failed to join channel {topic}: {reply[4]}")

        # Read events
        while True:
            try:
                raw = ws.recv()
            except TimeoutError:
                raise TimeoutError(f"Stream timed out after {timeout}s")

            msg = json.loads(raw)
            if not isinstance(msg, list) or len(msg) < 5:
                continue

            _join_ref, _ref, _topic, event, payload = msg

            if event in ("phx_reply", "phx_close", "phx_error", "heartbeat"):
                if event == "phx_error":
                    yield StreamEvent(type="error", data=payload)
                    return
                if event == "phx_close":
                    return
                continue

            stream_event = StreamEvent(type=event, data=payload)
            yield stream_event

            if event in ("completed", "error"):
                return


def _parse_agent(data: dict) -> AgentResponse:
    """Parse an agent dict from the API into an AgentResponse."""
    return AgentResponse(
        id=data["id"],
        name=data["name"],
        status=data.get("status", "active"),
        model=data.get("model", ""),
        mode=data.get("mode", "task"),
        system_prompt=data.get("system_prompt", ""),
        max_steps=data.get("max_steps", 50),
    )


# Chars kept of a tool result once it has aged out of the last two messages.
# A context-cost policy, applied here because the worker is what holds the
# plaintext; core forwards results untouched.
TOOL_RESULT_CAP = 200


def _compose_system_prompt(task: dict) -> str:
    """The prompt the model sees: the def's prompt verbatim, then the
    conversation summary and the date the orchestrator put in the envelope."""
    prompt = task.get("system_prompt") or ""
    summary = task.get("summary")
    date = task.get("date")
    if isinstance(summary, str) and summary:
        prompt += "\n\nSummary of earlier conversation: " + summary
    if date:
        prompt += f"\n\nCurrent date: {date}."
    return prompt


def _render_message(msg: dict) -> dict:
    """Render a kinded message — one the orchestrator resolved itself — to
    plain content. Messages without a kind pass through untouched."""
    kind = msg.get("kind")
    if not kind:
        return msg
    out = {k: v for k, v in msg.items() if k not in ("kind", "data")}
    out["content"] = _render_kind(kind, msg.get("data") or {}, msg.get("content"))
    return out


def _encode(value) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _render_kind(kind: str, data: dict, content) -> str:
    if kind == "inherited_context":
        return "[Inherited context from parent agent]\n" + _encode(content)
    if kind == "timer_completed":
        return "Timer completed."
    if kind == "tool_denied":
        return f"Tool '{data.get('tool_name')}' is not in this agent's allowed tools."
    if kind == "subagent_denied":
        reason = data.get("reason")
        if reason == "disabled":
            return "This agent is not permitted to launch sub-agents."
        if reason == "max_depth":
            return (
                f"Sub-agent nesting limit reached (max depth {data.get('max_depth')}). "
                "Do the work in this agent instead of delegating further."
            )
        return f"Agent '{data.get('agent_name')}' is not in this agent's allowed sub-agents."
    if kind == "subagent_list_denied":
        return "Listing agents is not permitted for this agent."
    if kind == "subagent_not_found":
        return f"Agent '{data.get('agent_name')}' not found"
    if kind == "subagent_self":
        return "Cannot launch self as a sub-agent"
    if kind == "subagent_missing":
        return f"Sub-agent run {data.get('run_id')} no longer exists, so its result cannot be recovered."
    if kind == "subagent_launch_failed":
        return f"Failed to launch agent '{data.get('agent_name')}': {data.get('reason')}"
    if kind == "subagent_completed":
        return json.dumps({"run_id": data.get("run_id"), "status": "completed", "output": content or ""})
    if kind == "subagent_failed":
        return json.dumps({"run_id": data.get("run_id"), "status": "failed", "error": content or ""})
    if kind == "list_agents":
        return json.dumps(data.get("agents") or [])
    if content in (None, ""):
        return _encode(data)
    return _encode(content)


COMPACTION_INSTRUCTION = (
    "Write a summary of the conversation so far for your own future reference. "
    "You will continue the same task with only this summary and the most recent "
    "messages, so keep every fact you would need: the task and its constraints, "
    "decisions made and why, files and identifiers touched, what was tried and "
    "failed, what remains to be done, and anything the user asked for that is not "
    "finished. Fold in the earlier summary if there is one. Write prose or terse "
    "notes, no preamble, no commentary about summarising."
)


def _compose_compaction_prompt(task: dict) -> str:
    """The def's prompt (so the summary is written from the agent's point of
    view) plus the earlier summary, for a purpose="compact" task."""
    prompt = task.get("system_prompt") or ""
    summary = task.get("summary")
    if isinstance(summary, str) and summary:
        prompt += "\n\nSummary of earlier conversation: " + summary
    return prompt


def _compaction_messages(task: dict) -> list[dict]:
    """The folded history, rendered, then the instruction."""
    rendered = [_render_message(m) for m in task.get("messages", [])]
    return rendered + [{"role": "user", "content": COMPACTION_INSTRUCTION}]


# One response's ceiling when the agent does not set one. Higher than a
# provider's own floor because a coding agent writes files in a single
# turn, and a turn that reaches the ceiling comes back cut off.
DEFAULT_MAX_TOKENS = 8192


def _max_tokens(task: dict) -> int:
    """The agent's cap, from the task envelope, or ours."""
    value = task.get("max_tokens")
    return value if isinstance(value, int) and value > 0 else DEFAULT_MAX_TOKENS


def _messages_for_task(task: dict) -> list[dict]:
    """Rendered messages, elided unless core manages the context itself."""
    rendered = [_render_message(m) for m in task.get("messages", [])]
    if task.get("context_policy"):
        return rendered
    return _elide_old_tool_results(rendered)


def _elide_old_tool_results(messages: list[dict]) -> list[dict]:
    """Cap tool results older than the last two messages."""
    if len(messages) <= 4:
        return messages
    old, recent = messages[:-2], messages[-2:]
    out = []
    for msg in old:
        content = msg.get("content")
        if msg.get("role") == "tool" and isinstance(content, str) and len(content) > TOOL_RESULT_CAP:
            msg = {**msg, "content": content[:TOOL_RESULT_CAP] + "...(truncated)"}
        out.append(msg)
    return out + recent


def _final_output(messages: list[dict], content) -> str:
    """The run's output when the model stops. A turn can say something
    substantive alongside a tool call and then end with an empty "stop" turn;
    fall back to the last non-empty assistant text rather than losing it."""
    text = content if isinstance(content, str) else ""
    if text.strip():
        return text
    for msg in reversed(messages):
        c = msg.get("content")
        if msg.get("role") == "assistant" and isinstance(c, str) and c.strip():
            return c
    return text


def _to_litellm_messages(messages: list[dict]) -> list[dict]:
    """Translate neutral-format messages to LiteLLM/OpenAI format.

    Neutral format tool_calls use {id, name, arguments (object)}.
    OpenAI format expects {id, type: "function", function: {name, arguments (JSON string)}}.
    """
    result = []
    for msg in messages:
        role = msg.get("role", "")

        if role == "assistant" and msg.get("tool_calls"):
            converted_calls = []
            for tc in msg["tool_calls"]:
                args = tc.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args)
                converted_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": args,
                    },
                })
            result.append({
                "role": "assistant",
                "content": msg.get("content", "") or None,
                "tool_calls": converted_calls,
            })
        elif role == "tool":
            # Ensure tool results have the right field name for LiteLLM
            result.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", msg.get("id", "")),
                "content": msg.get("content", ""),
            })
        else:
            content = msg.get("content")
            if isinstance(content, dict):
                # An opaque block this worker holds no key for, or structured
                # content: send it encoded rather than crash the call.
                msg = {**msg, "content": json.dumps(content)}
            result.append(msg)

    return result


def _to_litellm_tools(tools: list[dict]) -> list[dict]:
    """Translate neutral tool definitions to LiteLLM/OpenAI format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            },
        }
        for t in tools
    ]


def _from_litellm_response(response) -> dict:
    """Translate a LiteLLM response to Norns neutral wire format."""
    choice = response.choices[0]
    message = choice.message

    content = message.content or ""

    tool_calls: list[dict] = []
    if message.tool_calls:
        for tc in message.tool_calls:
            arguments = tc.function.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": arguments,
            })

    # LiteLLM normalizes finish_reason to OpenAI values
    finish_reason_map = {
        "stop": "stop",
        "tool_calls": "tool_call",
        "length": "length",
        "content_filter": "stop",
    }
    finish_reason = finish_reason_map.get(choice.finish_reason, choice.finish_reason)

    result: dict[str, Any] = {
        "status": "ok",
        "content": content,
        "finish_reason": finish_reason,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    }
    if tool_calls:
        result["tool_calls"] = tool_calls

    return result
