"""Test doubles shared by the worker tests."""

import asyncio
import json


class FakeWS:
    """Stands in for a websockets connection: async context manager,
    async iterator of queued incoming frames, captures sent frames."""

    def __init__(self):
        self.incoming = asyncio.Queue()
        self.results = asyncio.Queue()
        self.sent = []  # every frame the worker sent, decoded
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def send(self, raw):
        if self.closed:
            raise ConnectionError("socket closed")
        msg = json.loads(raw)
        self.sent.append(msg)
        if msg[3] == "tool_result":
            self.results.put_nowait(msg[4])

    async def recv(self):
        # The join reply.
        return json.dumps([None, "1", "worker:lobby", "phx_reply", {"status": "ok", "response": {}}])

    async def close(self):
        self.closed = True
        self.incoming.put_nowait(None)

    def events(self):
        return [m[3] for m in self.sent]

    def __aiter__(self):
        return self

    async def __anext__(self):
        frame = await self.incoming.get()
        if frame is None:
            raise StopAsyncIteration
        return frame


def tool_task(task_id, tool_name, **input_data):
    return json.dumps(
        [None, "1", "worker:lobby", "tool_task",
         {"task_id": task_id, "tool_name": tool_name, "input": input_data}]
    )
