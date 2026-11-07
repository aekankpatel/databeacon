"""SQL sandbox: validation and resource-limited, read-only execution on DuckDB.

Defense in depth:
  1. Static validation - exactly one statement, and DuckDB's own parser must
     classify it as a SELECT (``json_serialize_sql`` only accepts SELECT trees).
  2. Deny-list for table functions / pragmas that touch the filesystem or network.
  3. Execution on a connection opened ``read_only=True`` with
     ``enable_external_access = false`` and a memory/thread cap.
  4. Row cap (result wrapped in an outer LIMIT) and a wall-clock timeout that
     interrupts the query.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

_FORBIDDEN_PATTERNS = [
    r"\bread_(csv|csv_auto|parquet|json|json_auto|text|blob|ndjson|xlsx)\b",
    r"\b(glob|parquet_scan|parquet_metadata|sniff_csv|iceberg_scan|delta_scan)\s*\(",
    r"\b(pragma_\w+|duckdb_secrets|duckdb_settings|duckdb_extensions)\s*\(",
    r"\b(getenv|current_setting)\s*\(",
    r"(https?|s3|gcs|az|hf)://",
]
_FORBIDDEN_RE = re.compile("|".join(_FORBIDDEN_PATTERNS), re.IGNORECASE)
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")
_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


class SQLValidationError(ValueError):
    """Raised when a query is rejected by the sandbox."""


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    elapsed_ms: float
    dtypes: list[str] = field(default_factory=list)

    def to_dict(self, max_rows: int | None = None) -> dict[str, Any]:
        rows = self.rows if max_rows is None else self.rows[:max_rows]
        return {
            "sql": self.sql,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "rows": rows,
            "row_count": self.row_count,
            "rows_shown": len(rows),
            "truncated": self.truncated or len(rows) < len(self.rows),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


def _strip(sql: str) -> str:
    sql = _COMMENT_RE.sub(" ", sql)
    return sql.strip().rstrip(";").strip()


def validate_sql(sql: str) -> str:
    """Return a normalised query or raise :class:`SQLValidationError`."""
    if not isinstance(sql, str) or not sql.strip():
        raise SQLValidationError("Query is empty.")
    if len(sql) > 20_000:
        raise SQLValidationError("Query is too long (max 20k characters).")
    cleaned = _strip(sql)
    without_literals = _STRING_LITERAL_RE.sub("''", cleaned)
    if ";" in without_literals:
        raise SQLValidationError("Only a single statement is allowed.")
    first = without_literals.lstrip("( \n\t").split(None, 1)[0].lower() if without_literals else ""
    if first not in {"select", "with", "from", "values", "summarize", "describe", "table"}:
        raise SQLValidationError(f"Only read-only SELECT queries are allowed (got '{first}').")
    if first in {"summarize", "describe", "table"}:
        # DuckDB shorthands that are read-only; parser check below does not cover them.
        if _FORBIDDEN_RE.search(without_literals):
            raise SQLValidationError("Query uses a forbidden function or keyword.")
        return cleaned
    if _FORBIDDEN_RE.search(without_literals):
        raise SQLValidationError("Query uses a forbidden function or keyword.")
    # Authoritative check with DuckDB's parser.
    con = duckdb.connect(":memory:")
    try:
        raw = con.execute("SELECT json_serialize_sql(?)", [cleaned]).fetchone()[0]
    finally:
        con.close()
    parsed = json.loads(raw)
    if parsed.get("error"):
        msg = parsed.get("error_message", "parse error")
        if "Only SELECT" in msg:
            raise SQLValidationError("Only read-only SELECT queries are allowed.")
        raise SQLValidationError(f"SQL parse error: {msg}")
    if len(parsed.get("statements", [])) != 1:
        raise SQLValidationError("Only a single statement is allowed.")
    return cleaned


def _memory_limit() -> str:
    from .config import get_settings

    return get_settings().duckdb_memory_limit


def connect_readonly(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(
        str(db_path),
        read_only=True,
        config={"enable_external_access": False, "threads": 2, "memory_limit": _memory_limit()},
    )
    return con


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    if hasattr(v, "isoformat"):
        return v.isoformat()
    try:
        import decimal

        if isinstance(v, decimal.Decimal):
            return float(v)
    except Exception:  # pragma: no cover
        pass
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def execute_readonly(db_path: Path, sql: str, *, max_rows: int = 5000, timeout_s: float = 15.0) -> QueryResult:
    """Validate and run ``sql`` against the dataset database with limits."""
    cleaned = validate_sql(sql)
    con = connect_readonly(db_path)
    timer = threading.Timer(timeout_s, con.interrupt)
    started = time.perf_counter()
    try:
        timer.start()
        wrapped = f"SELECT * FROM ({cleaned}) AS _q LIMIT {int(max_rows) + 1}"
        try:
            rel = con.execute(wrapped)
        except duckdb.ParserException:
            rel = con.execute(cleaned)  # e.g. SUMMARIZE cannot be wrapped
        columns = [d[0] for d in rel.description]
        dtypes = [str(d[1]) for d in rel.description]
        rows = rel.fetchmany(max_rows + 1)
    except duckdb.InterruptException as exc:
        raise SQLValidationError(f"Query exceeded the {timeout_s:.0f}s time limit.") from exc
    except duckdb.Error as exc:
        raise SQLValidationError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        timer.cancel()
        con.close()
    elapsed = (time.perf_counter() - started) * 1000
    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    return QueryResult(
        sql=cleaned,
        columns=columns,
        rows=[[_jsonable(v) for v in r] for r in rows],
        row_count=len(rows),
        truncated=truncated,
        elapsed_ms=elapsed,
        dtypes=dtypes,
    )


def query_df(db_path: Path, sql: str, *, max_rows: int = 200_000, timeout_s: float = 30.0):
    """Run a validated query and return a pandas DataFrame (used by analysis tools)."""
    cleaned = validate_sql(sql)
    con = connect_readonly(db_path)
    timer = threading.Timer(timeout_s, con.interrupt)
    try:
        timer.start()
        return con.execute(f"SELECT * FROM ({cleaned}) AS _q LIMIT {int(max_rows)}").df()
    except duckdb.InterruptException as exc:
        raise SQLValidationError(f"Query exceeded the {timeout_s:.0f}s time limit.") from exc
    except duckdb.Error as exc:
        raise SQLValidationError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        timer.cancel()
        con.close()
