"""Shared helpers for DataBeacon MCP servers."""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Callable
from typing import Any, TypeVar

import anyio

from ..config import get_settings
from ..datasets import DatasetRegistry

T = TypeVar("T")

# Expected tool failures are returned to the agent; don't spam the server log with them.
logging.getLogger("mcp.server.mcpserver.server").setLevel(logging.WARNING)


def default_registry() -> DatasetRegistry:
    return DatasetRegistry(get_settings().datasets_dir)


async def in_thread(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run blocking (DuckDB / pandas / SciPy) work off the event loop."""
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


def quote_ident(name: str) -> str:
    if not isinstance(name, str) or not name or len(name) > 128:
        raise ValueError(f"Invalid column name: {name!r}")
    return '"' + name.replace('"', '""') + '"'


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}(:\d{2})?)?$")


def check_date(value: str, field: str) -> str:
    if not isinstance(value, str) or not _ISO_DATE.match(value.strip()):
        raise ValueError(f"{field} must be an ISO date like 2025-10-01 (got {value!r})")
    return value.strip()


def expected_errors(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Surface anticipated failures (bad SQL, unknown columns, ...) to the client as ToolError.

    MCP servers hide the text of unexpected exceptions; these messages are safe and the
    agent needs them to self-correct (error recovery)."""
    from mcp.server.mcpserver.exceptions import ToolError

    from ..datasets import DatasetError
    from ..sql_guard import SQLValidationError

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except (ValueError, KeyError, TypeError, SQLValidationError, DatasetError) as exc:
            raise ToolError(str(exc)) from exc

    return wrapper
