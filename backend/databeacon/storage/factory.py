"""Choose the storage backend from settings."""

from __future__ import annotations

import os
import threading

from .base import SQLiteStore, SQLStore, StorageError

_stores: dict[str, SQLStore] = {}
_lock = threading.Lock()


def shared_storage_enabled() -> bool:
    return os.environ.get("DATABEACON_STORAGE", "local").strip().lower() == "d1"


def open_store(name: str) -> SQLStore:
    """``name`` selects a local SQLite file (``<data dir>/<name>.sqlite``); with D1 every component
    shares the one configured database."""
    from ..config import get_settings

    d1 = shared_storage_enabled()
    key = "d1" if d1 else f"sqlite:{get_settings().data_dir}:{name}"
    with _lock:
        if key not in _stores:
            if d1:
                from .d1 import D1Store

                _stores[key] = D1Store(os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""),
                                       os.environ.get("DATABEACON_D1_DATABASE_ID", ""))
            else:
                _stores[key] = SQLiteStore(get_settings().data_dir / f"{name}.sqlite")
        return _stores[key]


def reset_stores() -> None:
    with _lock:
        _stores.clear()


__all__ = ["open_store", "shared_storage_enabled", "reset_stores", "StorageError"]
