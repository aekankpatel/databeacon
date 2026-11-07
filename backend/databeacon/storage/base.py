"""SQL store interface and the local SQLite implementation."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

Params = Sequence[Any]


class StorageError(RuntimeError):
    """The store could not be read or written."""


class SQLStore(Protocol):
    kind: str

    def query(self, sql: str, params: Params = ()) -> list[dict[str, Any]]:
        """Run a read statement and return rows as dicts."""

    def execute(self, sql: str, params: Params = ()) -> int:
        """Run one write statement atomically; return the number of rows changed."""

    def batch(self, statements: Sequence[tuple[str, Params]]) -> list[int]:
        """Run several write statements as one atomic unit; return rows changed per statement."""

    def ensure_schema(self, key: str, statements: Sequence[str]) -> None:
        """Run idempotent DDL once per process for ``key``."""


class SQLiteStore:
    kind = "sqlite"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._schemas: set[str] = set()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def query(self, sql: str, params: Params = ()) -> list[dict[str, Any]]:
        try:
            conn = self._conn()
            try:
                return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]
            finally:
                conn.close()
        except sqlite3.Error as exc:
            raise StorageError(f"sqlite query failed: {exc}") from exc

    def execute(self, sql: str, params: Params = ()) -> int:
        return self.batch([(sql, params)])[0]

    def batch(self, statements: Sequence[tuple[str, Params]]) -> list[int]:
        try:
            with self._lock:
                conn = self._conn()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    out = []
                    for sql, params in statements:
                        out.append(conn.execute(sql, tuple(params)).rowcount)
                    conn.execute("COMMIT")
                    return out
                except BaseException:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    raise
                finally:
                    conn.close()
        except sqlite3.Error as exc:
            raise StorageError(f"sqlite write failed: {exc}") from exc

    def ensure_schema(self, key: str, statements: Sequence[str]) -> None:
        if key in self._schemas:
            return
        self.batch([(s, ()) for s in statements])
        self._schemas.add(key)
