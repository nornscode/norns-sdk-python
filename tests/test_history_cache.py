"""The per-run history cache behind `features: ["message_cache"]`."""

from norns.client import _HistoryCache


def full(run_id, messages, **extra):
    return {"task_id": "t", "run_id": run_id, "messages": messages, **extra}


def delta(run_id, base_count, drop, append):
    return {
        "task_id": "t",
        "run_id": run_id,
        "messages_delta": {"base_count": base_count, "drop": drop, "append": append},
    }


def test_a_full_task_is_remembered_and_a_delta_applied_to_it():
    cache = _HistoryCache()

    first = cache.resolve(full(1, [{"role": "user", "content": "hi"}]))
    assert first["messages"] == [{"role": "user", "content": "hi"}]

    second = cache.resolve(delta(1, 1, 0, [{"role": "assistant", "content": "yo"}]))
    assert second["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    assert "messages_delta" not in second
    assert second["task_id"] == "t"

    # A dropped prefix (compaction, a sliding window).
    third = cache.resolve(delta(1, 2, 1, [{"role": "user", "content": "more"}]))
    assert [m["content"] for m in third["messages"]] == ["yo", "more"]


def test_a_delta_on_an_unknown_or_wrong_sized_base_is_a_miss():
    cache = _HistoryCache()
    assert cache.resolve(delta(7, 1, 0, [])) is None

    cache.resolve(full(7, [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]))
    assert cache.resolve(delta(7, 1, 0, [])) is None
    # The miss did not disturb what is held.
    assert cache.resolve(delta(7, 2, 0, []))["messages"] == [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]


def test_compaction_tasks_leave_the_cache_alone():
    cache = _HistoryCache()
    cache.resolve(full(1, [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]))

    compact = cache.resolve(full(1, [{"role": "user", "content": "a"}], purpose="compact"))
    assert compact["messages"] == [{"role": "user", "content": "a"}]

    assert cache.resolve(delta(1, 2, 0, []))["messages"] == [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]


def test_the_run_used_longest_ago_is_evicted_first():
    cache = _HistoryCache(max_runs=2)
    for run_id in (1, 2, 3):
        cache.resolve(full(run_id, [{"role": "user", "content": str(run_id)}]))

    assert cache.resolve(delta(1, 1, 0, [])) is None
    assert cache.resolve(delta(2, 1, 0, [])) is not None
    assert cache.resolve(delta(3, 1, 0, [])) is not None


def test_the_join_payload_declares_the_feature():
    from norns import Agent, Norns

    client = Norns("http://localhost:4000", api_key="k")
    payload = client._join_payload(Agent(name="bot"), "w")
    assert payload["features"] == ["message_cache"]
