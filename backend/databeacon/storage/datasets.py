"""Dataset registry backed by the shared SQL store (Cloudflare D1 on serverless).

The uploaded file is ingested with the regular local registry (same validation and limits), then the
table is stored as compressed Parquet chunks plus a metadata row. Any instance can rebuild the DuckDB
file from the store into its local cache; the cache is only an optimisation.

Write protocol: metadata row with ``ready = 0`` -> chunks -> ``ready = 1``. Readers only see ready
datasets, so a half-written upload is never visible; abandoned partial uploads are purged by retention.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path

from ..datasets import Dataset, DatasetError, DatasetRegistry
from .base import SQLStore, StorageError
from .dataset_codec import (
    DEFAULT_CHUNK_BYTES,
    DatasetCodecError,
    join_chunks,
    parquet_to_duckdb,
    sha256,
    split_chunks,
    table_to_parquet,
)

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL, table_name TEXT NOT NULL,
        filename TEXT NOT NULL, created_at TEXT NOT NULL, row_count INTEGER NOT NULL,
        column_count INTEGER NOT NULL, columns_json TEXT NOT NULL, parquet_sha256 TEXT NOT NULL,
        parquet_size INTEGER NOT NULL, chunk_count INTEGER NOT NULL, ready INTEGER NOT NULL DEFAULT 0)""",
    "CREATE INDEX IF NOT EXISTS idx_datasets_owner ON datasets(owner, created_at)",
    """CREATE TABLE IF NOT EXISTS dataset_chunks (
        dataset_id TEXT NOT NULL, idx INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY (dataset_id, idx))""",
]
CACHE_LIMIT_MB = float(os.environ.get("DATABEACON_DATASET_CACHE_MB", "150"))
_TOO_BIG = ("too big", "toobig", "too large", "exceeds", "size limit")


class SharedDatasetRegistry(DatasetRegistry):
    def __init__(self, store: SQLStore, cache_dir: Path, memory_limit: str | None = None):
        super().__init__(cache_dir)
        self.store = store
        self.memory_limit = memory_limit
        self._build_lock = threading.Lock()
        store.ensure_schema("datasets", SCHEMA)

    # ------------------------------------------------------------------ write
    def ingest_file(self, source: Path, *, name: str | None = None, original_filename: str | None = None,
                    owner: str = "local", max_rows: int | None = None, max_columns: int | None = None,
                    memory_limit: str | None = None) -> Dataset:
        ds = super().ingest_file(source, name=name, original_filename=original_filename, owner=owner,
                                 max_rows=max_rows, max_columns=max_columns, memory_limit=memory_limit)
        try:
            data = table_to_parquet(super().db_path(ds.id), ds.table)
            self._store(ds, data)
        except Exception:
            shutil.rmtree(self._dir(ds.id), ignore_errors=True)
            self._delete_rows(ds.id)
            raise
        return ds

    def _store(self, ds: Dataset, data: bytes) -> None:
        chunk_bytes = DEFAULT_CHUNK_BYTES
        while True:
            chunks = split_chunks(data, chunk_bytes)
            self._delete_rows(ds.id)
            self.store.execute(
                """INSERT INTO datasets (id, owner, name, table_name, filename, created_at, row_count, column_count,
                   columns_json, parquet_sha256, parquet_size, chunk_count, ready)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (ds.id, ds.owner, ds.name, ds.table, ds.filename, ds.created_at, ds.row_count, ds.column_count,
                 json.dumps(ds.columns), sha256(data), len(data), len(chunks)))
            try:
                for i, c in enumerate(chunks):  # one chunk per request keeps every request small
                    self.store.execute("INSERT INTO dataset_chunks (dataset_id, idx, data) VALUES (?, ?, ?)",
                                       (ds.id, i, c))
            except StorageError as exc:
                if any(k in str(exc).lower() for k in _TOO_BIG) and chunk_bytes > 32 * 1024:
                    chunk_bytes //= 2  # the store rejected the row size: retry with smaller chunks
                    continue
                raise
            self.store.execute("UPDATE datasets SET ready = 1 WHERE id = ?", (ds.id,))
            return

    def _delete_rows(self, dataset_id: str) -> None:
        self.store.batch([("DELETE FROM dataset_chunks WHERE dataset_id = ?", (dataset_id,)),
                          ("DELETE FROM datasets WHERE id = ?", (dataset_id,))])

    # ------------------------------------------------------------------- read
    @staticmethod
    def _row_to_ds(r: dict) -> Dataset:
        return Dataset(id=r["id"], name=r["name"], table=r["table_name"], filename=r["filename"],
                       created_at=r["created_at"], row_count=r["row_count"], column_count=r["column_count"],
                       columns=json.loads(r["columns_json"]), owner=r["owner"])

    def _row(self, dataset_id: str) -> dict:
        self._dir(dataset_id)  # validates the id format
        rows = self.store.query("SELECT * FROM datasets WHERE id = ? AND ready = 1", (dataset_id,))
        if not rows:
            raise DatasetError(f"Dataset {dataset_id} not found")
        return rows[0]

    def get(self, dataset_id: str, owner: str | None = None) -> Dataset:
        ds = self._row_to_ds(self._row(dataset_id))
        if owner is not None and ds.owner != owner:
            raise DatasetError(f"Dataset {dataset_id} not found")
        return ds

    def list(self, owner: str | None = None) -> list[Dataset]:
        if owner is None:
            rows = self.store.query("SELECT * FROM datasets WHERE ready = 1 ORDER BY created_at DESC")
        else:
            rows = self.store.query("SELECT * FROM datasets WHERE ready = 1 AND owner = ? ORDER BY created_at DESC",
                                    (owner,))
        return [self._row_to_ds(r) for r in rows]

    def delete(self, dataset_id: str, owner: str | None = None) -> None:
        self.get(dataset_id, owner)  # ownership check
        self._delete_rows(dataset_id)
        shutil.rmtree(self._dir(dataset_id), ignore_errors=True)

    def db_path(self, dataset_id: str) -> Path:
        """Local DuckDB file for the dataset, rebuilt from the shared store when this instance lacks it."""
        row = self._row(dataset_id)
        ddir = self._dir(dataset_id)
        db, marker = ddir / "data.duckdb", ddir / "parquet.sha256"
        if db.exists() and marker.exists() and marker.read_text() == row["parquet_sha256"]:
            os.utime(ddir)  # LRU touch
            return db
        with self._build_lock:
            if not (db.exists() and marker.exists() and marker.read_text() == row["parquet_sha256"]):
                chunks = [c["data"] for c in self.store.query(
                    "SELECT data FROM dataset_chunks WHERE dataset_id = ? ORDER BY idx", (dataset_id,))]
                if len(chunks) != row["chunk_count"]:
                    raise DatasetError(f"Dataset {dataset_id} is incomplete in storage")
                try:
                    data = join_chunks(chunks, row["parquet_sha256"], row["parquet_size"])
                except DatasetCodecError as exc:
                    raise DatasetError(f"Dataset {dataset_id}: {exc}") from exc
                parquet_to_duckdb(data, db, row["table_name"], self.memory_limit)
                (ddir / "meta.json").write_text(json.dumps(self._row_to_ds(row).to_dict(), indent=2))
                marker.write_text(row["parquet_sha256"])
                self._enforce_cache_limit(keep=dataset_id)
        return db

    def _enforce_cache_limit(self, keep: str) -> None:
        dirs = [p for p in self.root.iterdir() if p.is_dir()]
        sizes = {p: sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) for p in dirs}
        total = sum(sizes.values())
        for p in sorted(dirs, key=lambda d: d.stat().st_mtime):  # least recently used first
            if total <= CACHE_LIMIT_MB * 1e6:
                break
            if p.name != keep:
                total -= sizes[p]
                shutil.rmtree(p, ignore_errors=True)

    def purge_older_than(self, hours: float) -> int:
        if hours <= 0:
            return 0
        from datetime import UTC, datetime, timedelta

        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        old = [r["id"] for r in self.store.query("SELECT id FROM datasets WHERE created_at < ?", (cutoff,))]
        if old:
            self.store.batch([("DELETE FROM dataset_chunks WHERE dataset_id IN "
                               "(SELECT id FROM datasets WHERE created_at < ?)", (cutoff,)),
                              ("DELETE FROM datasets WHERE created_at < ?", (cutoff,))])
            for i in old:
                shutil.rmtree(self.root / i, ignore_errors=True)
        return len(old)


def make_registry():
    """The deployment's dataset registry: D1-backed when DATABEACON_STORAGE=d1, else local files."""
    from ..config import get_settings
    from . import open_store, shared_storage_enabled

    s = get_settings()
    if shared_storage_enabled():
        return SharedDatasetRegistry(open_store("app"), s.data_dir / "dataset_cache", s.duckdb_memory_limit)
    return DatasetRegistry(s.datasets_dir)
