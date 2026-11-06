"""Analysis run lifecycle: execution, live event fan-out and SQLite persistence."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from .agent.graph import AnalystAgent
from .config import Settings, get_settings
from .datasets import DatasetRegistry
from .llm import build_chat_model
from .mcp_client import MCPToolHub
from .observability import RunRecorder


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, question TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL, finished_at TEXT, result TEXT)""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs(dataset_id, created_at)")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, run_id: str, dataset_id: str, question: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO runs (id, dataset_id, question, status, created_at) VALUES (?,?,?,?,?)",
                      (run_id, dataset_id, question, "running", _now()))

    def finish(self, run_id: str, status: str, result: dict[str, Any]) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE runs SET status=?, finished_at=?, result=? WHERE id=?",
                      (status, _now(), json.dumps(result, default=str), run_id))

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["result"] = json.loads(out["result"]) if out["result"] else None
        return out

    def list(self, dataset_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        q = "SELECT id, dataset_id, question, status, created_at, finished_at FROM runs"
        args: tuple = ()
        if dataset_id:
            q += " WHERE dataset_id=?"
            args = (dataset_id,)
        q += " ORDER BY created_at DESC LIMIT ?"
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args + (limit,)).fetchall()]

    def mark_interrupted(self) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE runs SET status='interrupted', finished_at=? WHERE status='running'", (_now(),))


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


class RunManager:
    def __init__(self, registry: DatasetRegistry, settings: Settings | None = None,
                 llm_factory: Callable[[], BaseChatModel] | None = None, max_concurrent: int = 2):
        self.registry = registry
        self.s = settings or get_settings()
        self.store = RunStore(self.s.runs_db)
        self.store.mark_interrupted()
        self.llm_factory = llm_factory or (lambda: build_chat_model(self.s.temperature, self.s.llm_timeout_s))
        self.recorders: dict[str, RunRecorder] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self._sem = asyncio.Semaphore(max_concurrent)

    def start(self, dataset_id: str, question: str) -> str:
        ds = self.registry.get(dataset_id)  # raises if missing
        run_id = uuid.uuid4().hex[:12]
        self.store.create(run_id, dataset_id, question)
        rec = RunRecorder(run_id)
        self.recorders[run_id] = rec
        self.tasks[run_id] = asyncio.create_task(self._execute(run_id, ds.id, ds.name, question, rec))
        return run_id

    async def run_sync(self, dataset_id: str, question: str) -> dict[str, Any]:
        """Start a run and wait for it (used by the evaluation harness)."""
        run_id = self.start(dataset_id, question)
        await self.tasks[run_id]
        return self.get(run_id)

    async def _execute(self, run_id: str, dataset_id: str, dataset_name: str, question: str, rec: RunRecorder) -> None:
        status, result = "failed", {}
        state: dict[str, Any] = {}
        async with self._sem:
            rec.publish({"kind": "status", "status": "running"})
            try:
                llm = self.llm_factory()
                async with MCPToolHub(self.registry) as hub:
                    agent = AnalystAgent(llm, hub, rec, self.s)
                    state = await agent.run({"run_id": run_id, "dataset_id": dataset_id, "dataset_name": dataset_name,
                                             "question": question})
                status = "completed" if state.get("report", {}).get("ok") else "completed_with_errors"
            except Exception as exc:  # noqa: BLE001
                while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
                    exc = exc.exceptions[0]  # surface the real cause, not the task-group wrapper
                rec.log("error", "run_failed", status="error", error=f"{type(exc).__name__}: {exc}",
                        output=traceback.format_exc()[-2000:])
                state.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}")
            result = self._result(state, rec)
            self.store.finish(run_id, status, result)
            rec.publish({"kind": "status", "status": status})
            rec.publish({"kind": "done", "status": status})

    @staticmethod
    def _result(state: dict[str, Any], rec: RunRecorder) -> dict[str, Any]:
        return {
            "plan": state.get("plan"), "reflections": state.get("reflections", []),
            "synthesis": state.get("synthesis"), "verification": state.get("verification"),
            "charts": state.get("charts", []), "evidence": _slim_evidence(state.get("evidence", [])),
            "timeline": rec.events, "metrics": rec.metrics(), "report": state.get("report"),
            "errors": state.get("errors", []), "stop_reason": state.get("stop_reason"),
            "iterations": state.get("iteration"), "profile": state.get("profile"),
        }

    def get(self, run_id: str) -> dict[str, Any] | None:
        row = self.store.get(run_id)
        if row is None:
            return None
        if row["status"] == "running" and run_id in self.recorders:
            rec = self.recorders[run_id]
            row["result"] = {"timeline": rec.events, "metrics": rec.metrics()}
        return row
