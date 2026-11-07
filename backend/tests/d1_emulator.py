"""In-process emulator of the Cloudflare D1 HTTP query API, for tests.

Implements ``POST /accounts/{account}/d1/database/{db}/query`` with ``{"sql", "params"}`` or
``{"batch": [...]}`` bodies and D1's response envelope. Like a D1 database (a single Durable Object),
it executes one request at a time; a batch runs as one transaction and fails as a whole.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path

import httpx


class D1Emulator:
    def __init__(self, path: Path, max_row_bytes: int = 2_000_000):
        self.path = Path(path)
        self.lock = threading.Lock()  # D1 processes queries one at a time
        self.calls = 0
        self.fail_next: list[int] = []  # HTTP statuses to return for the next requests
        self.max_row_bytes = max_row_bytes
        self.bodies: list[dict] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def _run(self, conn: sqlite3.Connection, stmt: dict) -> dict:
        params = stmt.get("params") or []
        for p in params:
            if isinstance(p, str) and len(p.encode()) > self.max_row_bytes:
                raise sqlite3.DataError("string or blob too big: SQLITE_TOOBIG")
        cur = conn.execute(stmt["sql"], params)
        rows = [dict(r) for r in cur.fetchall()] if cur.description else []
        return {"results": rows, "success": True,
                "meta": {"changes": max(cur.rowcount, 0), "last_row_id": cur.lastrowid, "duration": 0.1}}

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.fail_next:
            return httpx.Response(self.fail_next.pop(0), json={"success": False, "errors": [{"message": "busy"}]})
        if not re.search(r"/accounts/[^/]+/d1/database/[^/]+/query$", request.url.path):
            return httpx.Response(404, json={"success": False, "errors": [{"code": 7003, "message": "no route"}]})
        if not request.headers.get("authorization", "").startswith("Bearer "):
            return httpx.Response(401, json={"success": False, "errors": [{"code": 10000, "message": "Authentication error"}]})
        body = json.loads(request.content)
        self.bodies.append(body)
        stmts = body["batch"] if "batch" in body else [body]
        with self.lock:
            conn = sqlite3.connect(self.path, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("BEGIN IMMEDIATE")
                results = [self._run(conn, s) for s in stmts]
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                return httpx.Response(400, json={"success": False, "result": [],
                                                 "errors": [{"code": 7500, "message": f"D1_ERROR: {exc}"}]})
            finally:
                conn.close()
        return httpx.Response(200, json={"success": True, "errors": [], "messages": [], "result": results})


def d1_store(emulator: D1Emulator, **kw):
    from databeacon.storage.d1 import D1Store

    return D1Store("acct", "db", token="test-token", base_url="https://d1.test/client/v4",
                   transport=emulator.transport(), **kw)
