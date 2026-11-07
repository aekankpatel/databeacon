"""Storage backends: D1 HTTP contract, atomic budget reservations and rate limits under concurrency."""

import multiprocessing as mp
import threading
import time

import pytest
from d1_emulator import D1Emulator, d1_store

from databeacon.budget import NeuronBudget
from databeacon.storage.base import SQLiteStore, StorageError


@pytest.fixture
def emu(tmp_path):
    return D1Emulator(tmp_path / "d1.sqlite")


def test_d1_request_and_response_contract(emu):
    s = d1_store(emu)
    s.ensure_schema("t", ["CREATE TABLE IF NOT EXISTS t (k TEXT PRIMARY KEY, v INTEGER)"])
    assert s.execute("INSERT INTO t VALUES (?, ?)", ("a", 1)) == 1
    assert s.batch([("INSERT INTO t VALUES (?, ?)", ("b", 2)), ("UPDATE t SET v = v + 1", ())]) == [1, 2]
    assert s.query("SELECT k, v FROM t ORDER BY k") == [{"k": "a", "v": 2}, {"k": "b", "v": 3}]
    assert emu.bodies[1] == {"sql": "INSERT INTO t VALUES (?, ?)", "params": ["a", 1]}
    assert "batch" in emu.bodies[2] and len(emu.bodies[2]["batch"]) == 2


def test_d1_batch_is_all_or_nothing(emu):
    s = d1_store(emu)
    s.ensure_schema("t", ["CREATE TABLE IF NOT EXISTS t (k TEXT PRIMARY KEY)"])
    with pytest.raises(StorageError):
        s.batch([("INSERT INTO t VALUES ('x')", ()), ("INSERT INTO t VALUES ('x')", ())])  # 2nd violates PK
    assert s.query("SELECT COUNT(*) AS n FROM t")[0]["n"] == 0


def test_d1_retries_transient_errors_but_not_auth(emu):
    s = d1_store(emu)
    s.ensure_schema("t", ["CREATE TABLE IF NOT EXISTS t (k TEXT)"])
    emu.fail_next = [503, 429]
    assert s.execute("INSERT INTO t VALUES ('a')") == 1  # third attempt succeeds
    emu.fail_next = [503, 503, 503]
    with pytest.raises(StorageError, match="after 3 attempts|503"):
        s.execute("INSERT INTO t VALUES ('b')")
    from databeacon.storage.d1 import D1Store

    bad = D1Store("acct", "db", token="x", base_url="https://d1.test/client/v4", transport=emu.transport())
    bad._headers["Authorization"] = "Token nope"
    with pytest.raises(StorageError, match="Authentication"):
        bad.query("SELECT 1")


def _hammer_reservations(budget, n_threads, each, results):
    def worker():
        got = []
        for _ in range(each):
            got.append(budget.reserve(100))
        results.extend(got)

    ts = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


@pytest.mark.parametrize("backend", ["sqlite", "d1"])
def test_concurrent_reservations_never_exceed_budget(tmp_path, emu, backend):
    store = SQLiteStore(tmp_path / "u.sqlite") if backend == "sqlite" else d1_store(emu)
    budget = NeuronBudget(store=store, daily_limit=1000)
    results: list = []
    _hammer_reservations(budget, n_threads=8, each=6, results=results)  # 48 attempts x 100 for a 1,000 budget
    granted = [r for r in results if r]
    assert len(granted) == 10, len(granted)
    assert budget.remaining() == 0
    assert budget.reserve(1) is None


def _proc_reserve(path, q):
    b = NeuronBudget(path=path, daily_limit=1000)
    q.put(sum(1 for _ in range(10) if b.reserve(100)))


def test_reservations_atomic_across_processes(tmp_path):
    """Separate processes sharing one SQLite ledger - the same race serverless instances have on D1."""
    path = tmp_path / "shared.sqlite"
    NeuronBudget(path=path, daily_limit=1000)
    q = mp.get_context("spawn").Queue()
    procs = [mp.get_context("spawn").Process(target=_proc_reserve, args=(path, q)) for _ in range(5)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert sum(q.get() for _ in procs) == 10


def test_record_releases_reservation_atomically(emu):
    b = NeuronBudget(store=d1_store(emu), daily_limit=1000)
    rid = b.reserve(400)
    assert b.remaining() == 600
    b.record("cloudflare", "@cf/openai/gpt-oss-120b", 1000, 100, reported_neurons=40.0, reservation_id=rid)
    assert b.remaining() == pytest.approx(960)
    assert b.used() == pytest.approx(40)


def test_shared_ledger_is_visible_to_every_instance(emu):
    a = NeuronBudget(store=d1_store(emu), daily_limit=1000)
    b = NeuronBudget(store=d1_store(emu), daily_limit=1000)  # a second "serverless instance"
    a.record("cloudflare", "m", 0, 0, reported_neurons=900)
    assert b.remaining() == pytest.approx(100)
    a.block_provider("cloudflare", "4006 seen by instance A")
    assert b.blocked() and "instance A" in b.blocked()


@pytest.mark.parametrize("backend", ["sqlite", "d1"])
def test_shared_rate_limit_is_atomic(tmp_path, emu, backend):
    from fastapi import HTTPException

    from databeacon.api.security import SharedRateLimiter

    store = SQLiteStore(tmp_path / "r.sqlite") if backend == "sqlite" else d1_store(emu)
    lim = SharedRateLimiter(store)
    allowed, denied = [], []

    def worker():
        for _ in range(5):
            try:
                lim.hit("runs:1.2.3.4", 7, 3600, "analyses")
                allowed.append(1)
            except HTTPException as exc:
                assert exc.status_code == 429 and int(exc.headers["Retry-After"]) > 0
                denied.append(1)

    ts = [threading.Thread(target=worker) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(allowed) == 7 and len(denied) == 23
    lim.hit("runs:5.6.7.8", 7, 3600, "analyses")  # other keys unaffected
    # a second limiter instance (another serverless instance) sees the same counts
    with pytest.raises(HTTPException):
        SharedRateLimiter(store).hit("runs:1.2.3.4", 7, 3600, "analyses")


def test_rate_window_expires(tmp_path):
    from databeacon.api.security import SharedRateLimiter

    lim = SharedRateLimiter(SQLiteStore(tmp_path / "r.sqlite"))
    lim.hit("k", 1, 1, "x")
    time.sleep(1.1)
    lim.hit("k", 1, 1, "x")  # the earlier hit is outside the 1 s window
