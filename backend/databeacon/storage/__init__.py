"""Pluggable SQL storage: local SQLite files (default) or Cloudflare D1 (shared, for serverless).

Both speak the SQLite dialect, so every component (usage ledger, rate limits, runs, events, datasets,
reports) uses one set of SQL statements. Atomic operations are written as *single* conditional
statements (``INSERT ... SELECT ... WHERE <limit check>``): D1's HTTP API has no interactive
transactions, but each statement runs atomically and a D1 database executes queries one at a time.
"""

from .base import SQLStore, StorageError
from .factory import open_store, shared_storage_enabled

__all__ = ["SQLStore", "StorageError", "open_store", "shared_storage_enabled"]
