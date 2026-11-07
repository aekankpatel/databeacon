"""Analysis run lifecycle: execution, event persistence, reconnection and interruption detection.

Two execution modes (``DATABEACON_EXECUTION``):

* ``background`` (default for local/Docker): ``POST /api/runs`` starts the agent as a task in this
  long-lived process and returns at once; clients follow progress over SSE.
* ``inline`` (serverless, e.g. Vercel): the investigation runs *inside* the request that started it
  and the response streams its events. No background process or instance-local state is relied on.

In both modes every event is appended to the shared store (local SQLite or Cloudflare D1) together
with a heartbeat, so any instance can serve the saved progress. That is **reconnecting**: reading what
has been saved while the original executor keeps working. If the executor stops (time limit, crash,
redeploy, frozen instance) its heartbeat goes stale and the run is marked ``interrupted``. An
interrupted investigation cannot be resumed from the middle - the agent's in-memory state is gone - so
the only way forward is a new run (optionally linked with ``rerun_of``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from .agent.graph import AnalystAgent
from .config import Settings, get_settings
from .datasets import DatasetRegistry
from .llm import build_chat_model
from .mcp_client import MCPToolHub
from .observability import RunRecorder
from .storage.base import SQLiteStore, SQLStore, StorageError
from .storage.dataset_codec import split_chunks

log = logging.getLogger("databeacon.runs")
FINAL = {"completed", "completed_with_issues", "completed_with_errors", "failed", "needs_clarification",
         "interrupted"}
BLOB_CHUNK = 512 * 1024


def _now() -> str:
    return datetime.now(UTC).isoformat()


class RunStore:
    SCHEMA = [
        """CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, question TEXT NOT NULL, status TEXT NOT NULL,
            created_at TEXT NOT NULL, finished_at TEXT, result TEXT)""",
        "CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs(dataset_id, created_at)",
        """CREATE TABLE IF NOT EXISTS run_events (
            run_id TEXT NOT NULL, seq INTEGER NOT NULL, ts REAL NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY (run_id, seq))""",
        """CREATE TABLE IF NOT EXISTS blobs (
            key TEXT NOT NULL, idx INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY (key, idx))""",
    ]
    # columns added after the first release (local stores are migrated in place)
    EXTRA_COLUMNS = {"owner": "TEXT NOT NULL DEFAULT 'local'", "heartbeat_at": "REAL", "execution": "TEXT",
                     "deadline_at": "REAL", "rerun_of": "TEXT", "interruption": "TEXT"}

    def __init__(self, store: SQLStore | Path):
        self.db = SQLiteStore(Path(store)) if isinstance(store, (str, Path)) else store
        self.db.ensure_schema("runs", self.SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        for col, decl in self.EXTRA_COLUMNS.items():
            try:
                self.db.query(f"SELECT {col} FROM runs LIMIT 0")
            except StorageError:
                self.db.execute(f"ALTER TABLE runs ADD COLUMN {col} {decl}")

    # ------------------------------------------------------------------ blobs
    def put_blob(self, key: str, data: bytes | str) -> None:
        raw = data.encode() if isinstance(data, str) else data
        chunks = split_chunks(raw, BLOB_CHUNK)
        self.db.batch([("DELETE FROM blobs WHERE key = ?", (key,))] +
                      [("INSERT INTO blobs (key, idx, data) VALUES (?, ?, ?)", (key, i, c)) for i, c in enumerate(chunks)])

    def get_blob(self, key: str) -> bytes | None:
        import base64

        rows = self.db.query("SELECT data FROM blobs WHERE key = ? ORDER BY idx", (key,))
        if not rows:
            return None
        return b"".join(base64.b64decode(r["data"]) for r in rows)

    # ------------------------------------------------------------------- runs
    def create(self, run_id: str, dataset_id: str, question: str, owner: str = "local", execution: str = "background",
               deadline_at: float | None = None, rerun_of: str | None = None) -> None:
        self.db.execute(
            "INSERT INTO runs (id, dataset_id, question, status, created_at, owner, heartbeat_at, execution, "
            "deadline_at, rerun_of) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?)",
            (run_id, dataset_id, question, _now(), owner, time.time(), execution, deadline_at, rerun_of))

    def append_events(self, run_id: str, events: list[tuple[int, float, dict[str, Any]]]) -> None:
        stmts = [("INSERT OR REPLACE INTO run_events (run_id, seq, ts, payload) VALUES (?, ?, ?, ?)",
                  (run_id, seq, ts, json.dumps(p, default=str))) for seq, ts, p in events]
        stmts.append(("UPDATE runs SET heartbeat_at = ? WHERE id = ? AND status = 'running'", (time.time(), run_id)))
        self.db.batch(stmts)

    def heartbeat(self, run_id: str) -> None:
        self.db.execute("UPDATE runs SET heartbeat_at = ? WHERE id = ? AND status = 'running'", (time.time(), run_id))

    def events(self, run_id: str, after_seq: int = -1) -> list[tuple[int, dict[str, Any]]]:
        rows = self.db.query("SELECT seq, payload FROM run_events WHERE run_id = ? AND seq > ? ORDER BY seq",
                             (run_id, after_seq))
        return [(r["seq"], json.loads(r["payload"])) for r in rows]

    def finish(self, run_id: str, status: str, result: dict[str, Any],
               interruption: dict[str, Any] | None = None) -> None:
        self.put_blob(f"result:{run_id}", json.dumps(result, default=str))
        info = json.dumps({**interruption, "detected_at": _now()}) if interruption else None
        self.db.execute("UPDATE runs SET status = ?, finished_at = ?, result = NULL, heartbeat_at = ?, "
                        "interruption = COALESCE(?, interruption) WHERE id = ?",
                        (status, _now(), time.time(), info, run_id))

    def get(self, run_id: str) -> dict[str, Any] | None:
        rows = self.db.query("SELECT * FROM runs WHERE id = ?", (run_id,))
        if not rows:
            return None
        out = dict(rows[0])
        if out.get("result"):  # rows written before results moved to blobs
            out["result"] = json.loads(out["result"])
        else:
            blob = self.get_blob(f"result:{run_id}")
            out["result"] = json.loads(blob) if blob else None
        out["interruption"] = json.loads(out["interruption"]) if out.get("interruption") else None
        return out

    def list(self, dataset_id: str | None = None, limit: int = 50, owner: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT id, dataset_id, question, status, created_at, finished_at FROM runs"
        conds, args = [], []
        if dataset_id:
            conds.append("dataset_id = ?")
            args.append(dataset_id)
        if owner is not None:
            conds.append("owner = ?")
            args.append(owner)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        return self.db.query(q + " ORDER BY created_at DESC LIMIT ?", (*args, limit))

    def mark_interrupted_if_stale(self, run_id: str, stale_before: float | None, reason: str) -> bool:
        """Atomically flip a running run whose heartbeat is older than ``stale_before`` to interrupted."""
        info = json.dumps({"reason": reason, "detected_at": _now(), "resumable": False})
        return self.db.execute(
            "UPDATE runs SET status = 'interrupted', finished_at = ?, interruption = ? "
            "WHERE id = ? AND status = 'running' AND (? IS NULL OR COALESCE(heartbeat_at, 0) < ?)",
            (_now(), info, run_id, stale_before, stale_before)) == 1

    def purge_older_than(self, hours: float) -> list[str]:
        if hours <= 0:
            return []
        from datetime import timedelta

        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        ids = [r["id"] for r in self.db.query("SELECT id FROM runs WHERE created_at < ? AND status != 'running'",
                                              (cutoff,))]
        for rid in ids:
            self.db.batch([("DELETE FROM run_events WHERE run_id = ?", (rid,)),
                           ("DELETE FROM blobs WHERE key IN (?, ?)", (f"result:{rid}", f"report:{rid}")),
                           ("DELETE FROM runs WHERE id = ?", (rid,))])
        return ids


def _slim_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for e in evidence:
        e = dict(e)
        data = e.get("data")
        if e["tool"] == "create_chart":
            e["data"] = {"chart_id": (data or {}).get("chart_id")}
        elif isinstance(data, dict) and isinstance(data.get("rows"), list) and len(data["rows"]) > 200:
            e["data"] = {**data, "rows": data["rows"][:200], "rows_stored": 200}
        out.append(e)
    return out


def partial_from_events(events: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    """Rebuild the saved progress of a run from its persisted event log."""
    timeline: dict[str, dict[str, Any]] = {}
    out: dict[str, Any] = {"plan": None, "reflections": [], "charts": [], "evidence": [], "alignment": None,
                           "needs_clarification": None}
    for _, m in events:
        kind = m.get("kind")
        if kind == "event":
            timeline[m["event"]["id"]] = m["event"]
        elif kind == "plan":
            out["plan"] = m.get("plan")
            out["spec"] = m.get("spec")
        elif kind == "reflection":
            out["reflections"].append(m["reflection"])
        elif kind == "chart":
            out["charts"].append(m["chart"])
        elif kind == "evidence":
            out["evidence"].append(m["evidence"])
        elif kind == "alignment":
            out["alignment"] = m["alignment"]
        elif kind == "clarification":
            out["needs_clarification"] = m["clarification"]
    out["timeline"] = sorted(timeline.values(), key=lambda e: e.get("started_at", ""))
    return out


class EventPersister:
    """Buffers a run's events and appends them to the store off the event loop, with a heartbeat."""

    def __init__(self, store: RunStore, run_id: str, every_s: float = 0.75, heartbeat_s: float = 15.0):
        self.store, self.run_id = store, run_id
        self.every_s, self.heartbeat_s = every_s, heartbeat_s
        self.buffer: list[tuple[int, float, dict[str, Any]]] = []
        self.seq = 0
        self._task: asyncio.Task | None = None
        self._last_write = time.time()
        self.errors = 0

    def __call__(self, message: dict[str, Any]) -> None:
        self.buffer.append((self.seq, time.time(), message))
        self.seq += 1

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.every_s)
            await self.flush()

    async def flush(self) -> None:
        batch, self.buffer = self.buffer, []
        try:
            if batch:
                await asyncio.to_thread(self.store.append_events, self.run_id, batch)
                self._last_write = time.time()
            elif time.time() - self._last_write > self.heartbeat_s:
                await asyncio.to_thread(self.store.heartbeat, self.run_id)
                self._last_write = time.time()
        except StorageError:
            self.errors += 1
            self.buffer = batch + self.buffer  # keep them for the next attempt
            log.warning("run %s: could not persist events (attempt %d)", self.run_id, self.errors)

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
        await self.flush()


class RunManager:
    def __init__(self, registry: DatasetRegistry, settings: Settings | None = None,
                 llm_factory: Callable[[], BaseChatModel] | None = None, max_concurrent: int = 2,
                 store: SQLStore | None = None):
        from .storage import open_store

        self.registry = registry
        self.s = settings or get_settings()
        self.store = RunStore(store or open_store("runs"))
        self.llm_factory = llm_factory or (lambda: build_chat_model(self.s.temperature, self.s.llm_timeout_s))
        self.recorders: dict[str, RunRecorder] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self._sem = asyncio.Semaphore(max_concurrent)

    # ----------------------------------------------------------------- start
    def _create(self, dataset_id: str, question: str, owner: str | None, execution: str,
                rerun_of: str | None = None) -> tuple[str, Any]:
        ds = self.registry.get(dataset_id, owner)  # raises if missing or not the caller's
        run_id = uuid.uuid4().hex[:12]
        deadline = time.time() + self.s.run_deadline_s if self.s.run_deadline_s > 0 else None
        self.store.create(run_id, dataset_id, question, owner or ds.owner, execution, deadline, rerun_of)
        return run_id, ds

    def start(self, dataset_id: str, question: str, clarification: str | None = None, owner: str | None = None,
              rerun_of: str | None = None) -> str:
        """Background mode: start the agent as a task in this process and return the run id."""
        run_id, ds = self._create(dataset_id, question, owner, "background", rerun_of)
        rec = RunRecorder(run_id)
        self.recorders[run_id] = rec
        self.tasks[run_id] = asyncio.create_task(self._execute(run_id, ds.id, ds.name, question, rec, clarification))
        return run_id

    async def stream(self, dataset_id: str, question: str, clarification: str | None = None,
                     owner: str | None = None, rerun_of: str | None = None) -> AsyncIterator[dict[str, Any]]:
        """Inline mode: run the investigation inside the caller's request, yielding its events."""
        run_id, ds = self._create(dataset_id, question, owner, "inline", rerun_of)
        rec = RunRecorder(run_id)
        self.recorders[run_id] = rec
        queue = rec.subscribe()
        task = asyncio.create_task(self._execute(run_id, ds.id, ds.name, question, rec, clarification))
        self.tasks[run_id] = task
        yield {"kind": "run", "run_id": run_id, "execution": "inline",
               "deadline_s": self.s.run_deadline_s or None}
        try:
            while True:
                getter = asyncio.ensure_future(queue.get())
                await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
                if getter.done():
                    msg = getter.result()
                    yield msg
                    if msg.get("kind") == "done":
                        break
                    continue
                getter.cancel()
                if queue.empty():  # the run ended without a final message: report what was saved
                    row = await asyncio.to_thread(self.store.get, run_id)
                    yield {"kind": "done", "status": (row or {}).get("status", "failed")}
                    break
        finally:
            rec.unsubscribe(queue)
            if not task.done():
                await task  # keep the invocation alive until the run is saved, even if the client left

    async def run_sync(self, dataset_id: str, question: str, clarification: str | None = None) -> dict[str, Any]:
        """Start a run and wait for it (used by the evaluation harness)."""
        run_id = self.start(dataset_id, question, clarification)
        await self.tasks[run_id]
        return self.get(run_id)

    # --------------------------------------------------------------- execute
    async def _execute(self, run_id: str, dataset_id: str, dataset_name: str, question: str, rec: RunRecorder,
                       clarification: str | None = None) -> None:
        status, state = "failed", {}
        persister = EventPersister(self.store, run_id)
        rec.sink = persister
        persister.start()
        async with self._sem:
            rec.publish({"kind": "status", "status": "running"})
            try:
                llm = self.llm_factory()
                async with MCPToolHub(self.registry) as hub:
                    agent = AnalystAgent(llm, hub, rec, self.s)
                    run = agent.run({"run_id": run_id, "dataset_id": dataset_id, "dataset_name": dataset_name,
                                     "question": question, "clarification": clarification,
                                     "deadline_at": time.time() + self.s.run_deadline_s
                                     if self.s.run_deadline_s > 0 else None})
                    if self.s.run_deadline_s > 0:  # hard backstop behind the agent's own deadline handling
                        state = await asyncio.wait_for(run, self.s.run_deadline_s + self.s.run_deadline_grace_s)
                    else:
                        state = await run
                if state.get("needs_clarification"):
                    status = "needs_clarification"
                elif (state.get("alignment") or {}).get("status") == "misaligned":
                    status = "completed_with_issues"  # answered something other than what was asked
                elif state.get("report", {}).get("ok"):
                    status = "completed"
                else:
                    status = "completed_with_errors"
            except Exception as exc:  # noqa: BLE001
                while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
                    exc = exc.exceptions[0]  # surface the real cause, not the task-group wrapper
                if isinstance(exc, TimeoutError) and self.s.run_deadline_s > 0:
                    status = "interrupted"
                    rec.log("error", "deadline_exceeded", status="error",
                            error=f"stopped after {self.s.run_deadline_s:.0f}s (serverless time limit)")
                    state = {"errors": ["investigation exceeded the time limit"], "stop_reason": "deadline exceeded"}
                    await self._finish(run_id, status, state, rec, persister)
                    return
                log.exception("run %s failed", run_id)
                rec.log("error", "run_failed", status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}",
                        # internals (tracebacks, paths) stay in the server log on public deployments
                        **({} if self.s.public_mode else {"output": traceback.format_exc()[-2000:]}))
                state.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}")
            await self._finish(run_id, status, state, rec, persister)

    async def _finish(self, run_id: str, status: str, state: dict[str, Any], rec: RunRecorder,
                      persister: EventPersister) -> None:
        result = self._result(state, rec)
        if status == "interrupted":
            result["interruption"] = {"reason": "the investigation exceeded its time limit", "resumable": False}
        try:
            await asyncio.to_thread(self._save_report, run_id)
            await asyncio.to_thread(self.store.finish, run_id, status, result, result.get("interruption"))
        except Exception:  # noqa: BLE001 - never leave the stream without a final status
            log.exception("run %s: could not save the result", run_id)
        finally:
            rec.publish({"kind": "status", "status": status})
            rec.publish({"kind": "done", "status": status})
            await persister.close()

    def _save_report(self, run_id: str) -> None:
        path = self.s.reports_dir / f"{run_id}.pdf"
        if path.exists():
            self.store.put_blob(f"report:{run_id}", path.read_bytes())

    @staticmethod
    def _result(state: dict[str, Any], rec: RunRecorder) -> dict[str, Any]:
        return {
            "plan": state.get("plan"), "reflections": state.get("reflections", []),
            "synthesis": state.get("synthesis"), "verification": state.get("verification"),
            "charts": state.get("charts", []), "evidence": _slim_evidence(state.get("evidence", [])),
            "timeline": rec.events, "metrics": rec.metrics(), "report": state.get("report"),
            "errors": state.get("errors", []), "stop_reason": state.get("stop_reason"),
            "iterations": state.get("iteration"), "profile": state.get("profile"),
            "temporal": state.get("temporal"), "spec": state.get("spec"), "alignment": state.get("alignment"),
            "needs_clarification": state.get("needs_clarification"),
        }

    # ------------------------------------------------------------------- read
    def _stale_cutoff(self, row: dict[str, Any]) -> tuple[float | None, str]:
        if row.get("deadline_at") and time.time() > row["deadline_at"] + 60:
            return None, "the investigation passed its time limit without finishing"  # None: regardless of heartbeat
        return time.time() - self.s.run_stale_s, (
            f"no progress was saved for over {self.s.run_stale_s:.0f}s: the server instance running it stopped "
            "(time limit, crash, redeploy or the instance was frozen)")

    def get(self, run_id: str) -> dict[str, Any] | None:
        row = self.store.get(run_id)
        if row is None:
            return None
        if row["status"] == "running":
            if run_id in self.recorders and run_id in self.tasks and not self.tasks[run_id].done():
                rec = self.recorders[run_id]  # executing in this process: live view
                row["result"] = {**partial_from_events([(0, m) for m in rec.history]),
                                 "timeline": rec.events, "metrics": rec.metrics()}
                row["connection"] = "live"
                return row
            cutoff, reason = self._stale_cutoff(row)
            if self.store.mark_interrupted_if_stale(run_id, cutoff, reason):
                row = self.store.get(run_id)
            else:
                row["connection"] = "reconnected"  # still running elsewhere: saved progress
        if row["status"] in ("running", "interrupted") and not row.get("result"):
            row["result"] = partial_from_events(self.store.events(run_id))
            row["result"]["partial"] = True
        return row

    def report_bytes(self, run_id: str) -> bytes | None:
        path = self.s.reports_dir / f"{run_id}.pdf"
        if path.exists():
            return path.read_bytes()
        return self.store.get_blob(f"report:{run_id}")
