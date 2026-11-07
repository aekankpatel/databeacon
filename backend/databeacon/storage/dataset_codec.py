"""Compact, type-preserving dataset serialisation for shared storage.

A dataset table is exported from DuckDB as zstd-compressed Parquet (types, NULLs and nested values
survive the round trip), split into base64 chunks that fit Cloudflare D1's 2 MB row limit, and rebuilt
into a DuckDB file on demand. A SHA-256 of the Parquet bytes guards against truncated or mixed chunks.
"""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile
from pathlib import Path

import duckdb

# raw bytes per chunk: 512 KB -> ~683 KB of base64 per D1 row (limit 2 MB); configurable
DEFAULT_CHUNK_BYTES = int(os.environ.get("DATABEACON_D1_CHUNK_BYTES", 512 * 1024))


class DatasetCodecError(ValueError):
    pass


def table_to_parquet(db_path: Path, table: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="db_export_") as d:
        out = Path(d) / "data.parquet"
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            con.execute(f"COPY \"{table}\" TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, "
                        "ROW_GROUP_SIZE 100000)")
        finally:
            con.close()
        return out.read_bytes()


def parquet_to_duckdb(data: bytes, db_path: Path, table: str, memory_limit: str | None = None) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_db = db_path.with_suffix(".tmp")
    tmp_db.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="db_import_") as d:
        pq = Path(d) / "data.parquet"
        pq.write_bytes(data)
        con = duckdb.connect(str(tmp_db), config={"memory_limit": memory_limit} if memory_limit else {})
        try:
            con.execute(f"CREATE TABLE \"{table}\" AS SELECT * FROM read_parquet('{pq.as_posix()}')")
        finally:
            con.close()
    os.replace(tmp_db, db_path)  # atomic: readers never see a half-written database


def split_chunks(data: bytes, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> list[str]:
    return [base64.b64encode(data[i:i + chunk_bytes]).decode("ascii") for i in range(0, len(data), chunk_bytes)] or [""]


def join_chunks(chunks: list[str], expected_sha256: str, expected_size: int) -> bytes:
    data = b"".join(base64.b64decode(c) for c in chunks)
    if len(data) != expected_size or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise DatasetCodecError("stored dataset is incomplete or corrupted (checksum mismatch)")
    return data


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table_fingerprint(db_path: Path, table: str) -> dict:
    """Schema, row count and an order-independent content hash, to verify a round trip."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        schema = con.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? "
                             "ORDER BY ordinal_position", [table]).fetchall()
        n = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        digest = con.execute(f'SELECT md5(string_agg(CAST(t AS VARCHAR), chr(10) ORDER BY CAST(t AS VARCHAR))) '
                             f'FROM "{table}" AS t').fetchone()[0]
    finally:
        con.close()
    return {"schema": [tuple(x) for x in schema], "rows": int(n), "content_md5": digest}
