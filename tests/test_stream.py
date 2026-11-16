"""Tests for NornsClient.stream against a local websocket server."""

import json
import threading

import httpx
import respx
from websockets.sync.server import serve

from norns import NornsClient


def _phoenix_server(frames):
    """A websocket server that acks the channel join, then sends `frames`."""
    joined = []

    def handler(ws):
        join = json.loads(ws.recv())
        joined.append(join)
        ws.send(json.dumps([join[0], join[1], join[2], "phx_reply", {"status": "ok", "response": {}}]))
        for frame in frames:
            ws.send(json.dumps([join[0], None, join[2], frame[0], frame[1]]))
        ws.recv()  # hold the connection until the client closes it

    server = serve(handler, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, joined


@respx.mock(assert_all_mocked=False)
def test_stream_yields_events_until_completed(respx_mock):
    respx_mock.route(host="127.0.0.1", path="/api/v1/agents/7/messages").mock(
        return_value=httpx.Response(202, json={"run_id": 42, "status": "accepted"})
    )
    server, joined = _phoenix_server([
        ("llm_response", {"content": "thinking"}),
        ("completed", {"output": "done"}),
        ("llm_response", {"content": "never read"}),
    ])
    try:
        port = server.socket.getsockname()[1]
        with NornsClient(f"http://127.0.0.1:{port}", api_key="nrn_test") as client:
            events = list(client.stream(7, "hi", timeout=5))
    finally:
        server.shutdown()

    assert [e.type for e in events] == ["llm_response", "completed"]
    assert events[-1].data == {"output": "done"}
    assert joined[0][2:] == ["agent:7", "phx_join", {"run_id": 42}]


@respx.mock(assert_all_mocked=False)
def test_stream_times_out_when_nothing_arrives(respx_mock):
    import pytest

    respx_mock.route(host="127.0.0.1", path="/api/v1/agents/7/messages").mock(
        return_value=httpx.Response(202, json={"run_id": 42, "status": "accepted"})
    )
    server, _joined = _phoenix_server([])
    try:
        port = server.socket.getsockname()[1]
        with NornsClient(f"http://127.0.0.1:{port}", api_key="nrn_test") as client:
            with pytest.raises(TimeoutError, match="timed out"):
                list(client.stream(7, "hi", timeout=0.3))
    finally:
        server.shutdown()
