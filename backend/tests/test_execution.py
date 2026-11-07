"""Serverless execution model: inline runs, reconnection vs interruption, deadlines, disconnects."""

import asyncio
import json
import threading
import time
from dataclasses import replace

import pytest
from d1_emulator import D1Emulator, d1_store

from databeacon.agent.testing import scripted_ecommerce_llm
from databeacon.runs import RunManager, RunStore

pytestmark = pytest.mark.anyio
Q = "Why did our revenue decline last quarter, and which segments contributed?"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def emu(tmp_path):
    return D1Emulator(tmp_path / "d1.sqlite")


def _settings(**kw):
    from databeacon.config import get_settings

    return replace(get_settings(), **kw)


async def _drain(gen):
    return [m async for m in gen]


async def test_inline_run_streams_and_saves_everything(registry, dataset, emu):
    mgr = RunManager(registry, _settings(execution_mode="inline", run_deadline_s=270),
                     llm_factory=lambda: scripted_ecommerce_llm(hallucinate_first=False), store=d1_store(emu))
    msgs = await _drain(mgr.stream(dataset.id, Q))
    assert msgs[0]["kind"] == "run" and msgs[-1] == {"kind": "done", "status": "completed"}
    run_id = msgs[0]["run_id"]
    # a different instance (empty memory) sees the finished run, its saved events and its PDF
    other = RunManager(registry, _settings(execution_mode="inline"), store=d1_store(emu))
    row = other.get(run_id)
    assert row["status"] == "completed" and row["execution"] == "inline"
    assert row["result"]["synthesis"]["answer"]["primary_entity"] == "Enterprise"
    saved = other.store.events(run_id)
    assert [m for _, m in saved][-1]["kind"] == "done"
    assert len(saved) >= len(msgs) - 1
    pdf = other.report_bytes(run_id)
    assert pdf and pdf[:4] == b"%PDF"


async def test_reconnect_reads_saved_progress_while_running_elsewhere(registry, dataset, emu):
    store = d1_store(emu)
    gate = threading.Event()
    policy = scripted_ecommerce_llm(hallucinate_first=False)
    real = policy.responder

    held = threading.Event()

    def slow(messages, tools_bound):  # hold the run mid-investigation until the viewer has looked
        if tools_bound and sum(1 for m in messages if m.type == "tool") >= 2:
            held.set()
            for _ in range(200):
                if gate.is_set():
                    break
                time.sleep(0.05)
        return real(messages, tools_bound)

    policy.responder = slow
    executor = RunManager(registry, _settings(execution_mode="inline", run_deadline_s=270),
                          llm_factory=lambda: policy, store=store)
    gen = executor.stream(dataset.id, Q)
    first = await gen.__anext__()
    run_id = first["run_id"]
    consumer = asyncio.create_task(_drain(gen))  # keep reading the stream, as a browser would
    for _ in range(100):
        if held.is_set():
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(1.2)  # let the persister flush
    viewer = RunManager(registry, _settings(execution_mode="inline"), store=d1_store(emu))  # another instance
    row = viewer.get(run_id)
    assert row["status"] == "running" and row["connection"] == "reconnected"
    assert row["result"]["partial"] and row["result"]["timeline"]
    gate.set()
    rest = await consumer
    assert rest[-1]["kind"] == "done"
    assert viewer.get(run_id)["status"] == "completed"


async def test_stopped_executor_is_reported_interrupted_not_resumed(registry, dataset, emu):
    store = RunStore(d1_store(emu))
    store.create("deadbeef0001", dataset.id, Q, owner=dataset.owner, execution="inline", deadline_at=time.time() + 270)
    store.append_events("deadbeef0001", [(0, time.time(), {"kind": "event", "event": {
        "id": "e1", "type": "node", "name": "plan", "status": "ok", "started_at": "2026-01-01T00:00:00"}})])
    with emu.lock:  # pretend the last heartbeat was 5 minutes ago: the instance died
        import sqlite3

        sqlite3.connect(emu.path).execute("UPDATE runs SET heartbeat_at = ? WHERE id = 'deadbeef0001'",
                                          (time.time() - 300,)).connection.commit()
    viewer = RunManager(registry, _settings(execution_mode="inline", run_stale_s=120), store=d1_store(emu))
    row = viewer.get("deadbeef0001")
    assert row["status"] == "interrupted"
    assert row["interruption"]["resumable"] is False and "stopped" in row["interruption"]["reason"]
    assert row["result"]["partial"] and row["result"]["timeline"][0]["name"] == "plan"  # saved progress kept


async def test_new_instance_does_not_interrupt_live_runs_elsewhere(registry, dataset, emu):
    store = RunStore(d1_store(emu))
    store.create("feedface0001", dataset.id, Q, owner=dataset.owner, execution="inline")
    RunManager(registry, _settings(), store=d1_store(emu))  # a cold start must not touch other runs
    row = RunManager(registry, _settings(run_stale_s=120), store=d1_store(emu)).get("feedface0001")
    assert row["status"] == "running" and row["connection"] == "reconnected"


async def test_deadline_still_produces_a_complete_answer(registry, dataset, emu):
    from databeacon.agent.graph import AnalystAgent

    mgr = RunManager(registry, _settings(execution_mode="inline", run_deadline_s=270),
                     llm_factory=lambda: scripted_ecommerce_llm(hallucinate_first=False), store=d1_store(emu))
    orig = AnalystAgent.DEADLINE_RESERVE_S
    AnalystAgent.DEADLINE_RESERVE_S = 269.9  # time is (almost) up as soon as the agent starts acting
    try:
        msgs = await _drain(mgr.stream(dataset.id, Q))
    finally:
        AnalystAgent.DEADLINE_RESERVE_S = orig
    run = mgr.get(msgs[0]["run_id"])
    names = [e["name"] for e in run["result"]["timeline"]]
    assert "deadline_reached" in names
    assert run["result"]["synthesis"] is not None and run["result"]["report"]["ok"]  # answer and PDF written
    assert run["status"] in ("completed", "completed_with_issues")


async def test_hard_time_limit_marks_run_interrupted(registry, dataset, emu):
    def stuck():
        policy = scripted_ecommerce_llm()
        real = policy.responder

        def hang(messages, tools_bound):
            time.sleep(1.5)
            return real(messages, tools_bound)

        policy.responder = hang
        return policy

    mgr = RunManager(registry, _settings(execution_mode="inline", run_deadline_s=0.5, run_deadline_grace_s=0.5),
                     llm_factory=stuck, store=d1_store(emu))
    msgs = await _drain(mgr.stream(dataset.id, Q))
    assert msgs[-1] == {"kind": "done", "status": "interrupted"}
    assert any(m.get("kind") == "event" and m["event"]["name"] == "deadline_exceeded" for m in msgs)
    assert mgr.get(msgs[0]["run_id"])["status"] == "interrupted"


def test_inline_api_streams_and_survives_client_disconnect(registry, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from databeacon.api.main import create_app
    from databeacon.config import get_settings, reset_settings_cache

    monkeypatch.setenv("DATABEACON_EXECUTION", "inline")
    reset_settings_cache()
    try:
        s = get_settings()
        mgr = RunManager(registry, s, llm_factory=lambda: scripted_ecommerce_llm(hallucinate_first=False))
        with TestClient(create_app(mgr)) as c:
            ds = c.post("/api/datasets/sample").json()
            # read only the first two events, then hang up
            with c.stream("POST", "/api/runs", json={"dataset_id": ds["id"], "question": Q}) as r:
                assert r.headers["content-type"].startswith("text/event-stream")
                assert "content-length" not in r.headers  # a fixed length breaks streaming on real servers
                lines = []
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        lines.append(json.loads(line[6:]))
                    if len(lines) >= 2:
                        break
            run_id = lines[0]["run_id"]
            for _ in range(100):
                row = c.get(f"/api/runs/{run_id}").json()
                if row["status"] != "running":
                    break
                time.sleep(0.05)
            assert row["status"] == "completed"  # the run kept going and was saved
            assert c.get(f"/api/runs/{run_id}/report.pdf").content[:4] == b"%PDF"
            events = c.get(f"/api/runs/{run_id}/events").text
            assert '"kind": "done"' in events
    finally:
        monkeypatch.delenv("DATABEACON_EXECUTION")
        reset_settings_cache()
