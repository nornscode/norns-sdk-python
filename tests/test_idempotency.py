"""A side effect happens once, however many times the call is dispatched.

Core names every side-effecting call with a key derived from things that
survive a crash — run, step, tool call id, gard — and re-dispatches the
call if the result never reached it. It cannot know whether the effect
landed; only the worker that ran it knows. So the worker keeps what it
did, and answers the second dispatch from that.
"""

import asyncio
import os

import pytest

from norns import Agent, Norns, tool
from norns.client import MAX_REMEMBERED_RESULTS, websockets  # patched below

from tests.fakes import FakeWS, tool_task


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Each test gets its own worker state. Real keys carry a run id and are
    unique for all time; the short keys here would otherwise leak between
    runs — which is itself the store doing its job."""
    monkeypatch.setenv("NORNS_STATE_DIR", str(tmp_path / "state"))


def charging_agent(charges):
    @tool
    def charge_card(amount: int) -> str:
        """Take money."""
        charges.append(amount)
        return f"charged {amount}"

    return Agent(name="t", system_prompt="t", tools=[charge_card])


def serve(monkeypatch, agent, script):
    """Run `script(fake)` against a worker connected to a fake socket."""
    fake = FakeWS()
    monkeypatch.setattr(websockets, "connect", lambda url: fake)
    norns = Norns("http://localhost:4000", api_key="nrn_test")

    async def scenario():
        tools_by_name = {t.name: t for t in agent.tools}
        task = asyncio.create_task(norns._connect_and_serve(agent, "w1", tools_by_name))
        await script(fake, norns)
        fake.incoming.put_nowait(None)
        await asyncio.wait_for(task, 2)

    asyncio.run(asyncio.wait_for(scenario(), 10))
    return norns


def test_the_same_key_twice_charges_once(monkeypatch):
    charges = []

    async def script(fake, _norns):
        fake.incoming.put_nowait(tool_task("t1", "charge_card", idempotency_key="k1", amount=10))
        first = await asyncio.wait_for(fake.results.get(), 2)

        # The result never reached core, so core asks again. Same call, same key.
        fake.incoming.put_nowait(tool_task("t2", "charge_card", idempotency_key="k1", amount=10))
        second = await asyncio.wait_for(fake.results.get(), 2)

        assert first["status"] == "ok"
        assert first["result"] == "charged 10"
        assert "duplicate" not in first

        # Same answer, and flagged, so core can record it rather than
        # showing what looks like a second charge.
        assert second["result"] == "charged 10"
        assert second["duplicate"] is True
        assert second["task_id"] == "t2"

    serve(monkeypatch, charging_agent(charges), script)
    assert charges == [10]


def test_without_a_key_nothing_is_remembered(monkeypatch):
    charges = []

    async def script(fake, norns):
        fake.incoming.put_nowait(tool_task("t1", "charge_card", amount=10))
        await asyncio.wait_for(fake.results.get(), 2)
        fake.incoming.put_nowait(tool_task("t2", "charge_card", amount=10))
        second = await asyncio.wait_for(fake.results.get(), 2)

        # No key means core did not consider this side-effecting. Twice asked
        # is twice meant.
        assert "duplicate" not in second
        assert len(norns._completed) == 0

    serve(monkeypatch, charging_agent(charges), script)
    assert charges == [10, 10]


def test_a_failure_is_not_remembered(monkeypatch):
    attempts = []

    @tool
    def flaky(n: int) -> str:
        """Fail once, then work."""
        attempts.append(n)
        if len(attempts) == 1:
            raise RuntimeError("network went away")
        return "ok"

    agent = Agent(name="t", system_prompt="t", tools=[flaky])

    async def script(fake, _norns):
        fake.incoming.put_nowait(tool_task("t1", "flaky", idempotency_key="k1", n=1))
        first = await asyncio.wait_for(fake.results.get(), 2)
        fake.incoming.put_nowait(tool_task("t2", "flaky", idempotency_key="k1", n=1))
        second = await asyncio.wait_for(fake.results.get(), 2)

        assert first["status"] == "error"
        # A retry of something that did not happen should happen.
        assert second["status"] == "ok"
        assert "duplicate" not in second

    serve(monkeypatch, agent, script)
    assert len(attempts) == 2


def test_the_memory_is_bounded(monkeypatch):
    charges = []

    async def script(fake, norns):
        for i in range(MAX_REMEMBERED_RESULTS + 10):
            fake.incoming.put_nowait(tool_task(f"t{i}", "charge_card", idempotency_key=f"k{i}", amount=1))
            await asyncio.wait_for(fake.results.get(), 2)

        # A worker lives for days; this is a crash guard, not a cache.
        assert len(norns._completed) == MAX_REMEMBERED_RESULTS
        assert "k0" not in norns._completed
        assert f"k{MAX_REMEMBERED_RESULTS + 9}" in norns._completed

    serve(monkeypatch, charging_agent(charges), script)


def test_the_memory_outlives_the_worker(monkeypatch):
    """The case the key exists for: the worker dies holding the call.

    Core re-dispatches it to whatever comes back under the same worker id, so
    what that worker remembers has to have survived the process — otherwise it
    does the side effect a second time, which is the whole thing we are trying
    not to do.
    """
    charges = []
    agent = charging_agent(charges)

    async def first_life(fake, _norns):
        fake.incoming.put_nowait(tool_task("t1", "charge_card", idempotency_key="k1", amount=10))
        await asyncio.wait_for(fake.results.get(), 2)

    serve(monkeypatch, agent, first_life)
    assert charges == [10]

    async def second_life(fake, norns):
        # A different process entirely, same worker id, same key.
        fake.incoming.put_nowait(tool_task("t2", "charge_card", idempotency_key="k1", amount=10))
        result = await asyncio.wait_for(fake.results.get(), 2)
        assert result["duplicate"] is True
        assert result["result"] == "charged 10"
        assert "k1" in norns._completed

    serve(monkeypatch, agent, second_life)
    assert charges == [10]


def test_an_unwritable_state_dir_does_not_stop_the_worker(monkeypatch, tmp_path):
    """A crash guard that crashes the worker is worse than no crash guard."""
    blocked = tmp_path / "nope"
    blocked.write_text("not a directory")
    monkeypatch.setenv("NORNS_STATE_DIR", str(blocked))

    charges = []

    async def script(fake, norns):
        fake.incoming.put_nowait(tool_task("t1", "charge_card", idempotency_key="k1", amount=10))
        result = await asyncio.wait_for(fake.results.get(), 2)
        assert result["status"] == "ok"
        # Still deduplicates in memory for as long as this process lives.
        assert "k1" in norns._completed

    serve(monkeypatch, charging_agent(charges), script)
    assert charges == [10]


def test_the_default_state_dir_is_per_user(monkeypatch, tmp_path):
    """The system temp dir is shared; another user's file must not be able to
    tell this worker a call was already done."""
    import os
    import tempfile

    from norns.client import _completed_calls_path

    monkeypatch.delenv("NORNS_STATE_DIR")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    path = _completed_calls_path("w1")
    assert path.parent != tmp_path / "norns-worker-state"
    assert path.parent.parent == tmp_path
    if hasattr(os, "getuid"):
        assert str(os.getuid()) in path.parent.name

    from norns.client import _CompletedCalls

    _CompletedCalls(path).add("k1", {"status": "ok"})
    if hasattr(os, "getuid"):
        assert (path.parent.stat().st_mode & 0o777) == 0o700


posix_only = pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership and modes")


def _forge(state_dir, worker_id="w1"):
    """What another local user would plant: a record saying k1 was done."""
    import json

    state_dir.mkdir(exist_ok=True)
    (state_dir / f"{worker_id}.completed.json").write_text(
        json.dumps({"k1": {"status": "ok", "result": "charged 10"}})
    )


def _charges_once_despite_forgery(monkeypatch):
    charges = []

    async def script(fake, norns):
        fake.incoming.put_nowait(tool_task("t1", "charge_card", idempotency_key="k1", amount=10))
        result = await asyncio.wait_for(fake.results.get(), 2)
        assert "duplicate" not in result
        assert result["result"] == "charged 10"

    serve(monkeypatch, charging_agent(charges), script)
    # The call really ran: the forged record was not believed.
    return charges == [10]


@posix_only
def test_a_state_dir_others_can_write_is_refused(monkeypatch, tmp_path, caplog):
    """Pre-created with loose permissions, it could hold forged records."""
    state = tmp_path / "state"
    _forge(state)
    state.chmod(0o777)
    assert _charges_once_despite_forgery(monkeypatch)
    assert "lets other users in" in caplog.text


@posix_only
def test_a_state_dir_with_group_read_is_refused(monkeypatch, tmp_path, caplog):
    state = tmp_path / "state"
    _forge(state)
    state.chmod(0o750)
    assert _charges_once_despite_forgery(monkeypatch)
    assert "lets other users in" in caplog.text


def test_a_symlinked_state_dir_is_refused(monkeypatch, tmp_path, caplog):
    real = tmp_path / "elsewhere"
    _forge(real)
    real.chmod(0o700)
    (tmp_path / "state").symlink_to(real)
    assert _charges_once_despite_forgery(monkeypatch)
    assert "is a symlink" in caplog.text


@posix_only
def test_a_state_dir_owned_by_someone_else_is_refused(monkeypatch, tmp_path, caplog):
    import os

    state = tmp_path / "state"
    _forge(state)
    state.chmod(0o700)
    monkeypatch.setattr(os, "getuid", lambda: os.stat(state).st_uid + 1)
    assert _charges_once_despite_forgery(monkeypatch)
    assert "not this user" in caplog.text


@posix_only
def test_a_state_file_others_can_write_is_ignored(monkeypatch, tmp_path, caplog):
    state = tmp_path / "state"
    _forge(state)
    state.chmod(0o700)
    (state / "w1.completed.json").chmod(0o666)
    assert _charges_once_despite_forgery(monkeypatch)
    assert "lets other users in" in caplog.text


def test_a_symlinked_state_file_is_ignored(monkeypatch, tmp_path, caplog):
    elsewhere = tmp_path / "elsewhere"
    _forge(elsewhere)
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    (state / "w1.completed.json").symlink_to(elsewhere / "w1.completed.json")
    assert _charges_once_despite_forgery(monkeypatch)
    assert "is a symlink" in caplog.text


@posix_only
def test_a_private_state_dir_is_trusted_and_files_are_written_0600(monkeypatch, tmp_path):
    """The checks must not refuse the dir this worker made itself, nor the
    0644 files earlier versions wrote into it."""
    from norns.client import _CompletedCalls

    state = tmp_path / "state"
    _forge(state)
    state.chmod(0o700)
    (state / "w1.completed.json").chmod(0o644)
    calls = _CompletedCalls(state / "w1.completed.json")
    assert "k1" in calls
    calls.add("k2", {"status": "ok"})
    assert ((state / "w1.completed.json").stat().st_mode & 0o777) == 0o600
