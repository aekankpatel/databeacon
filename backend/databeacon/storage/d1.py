"""Cloudflare D1 over its HTTP API (``POST /accounts/{account}/d1/database/{db}/query``).

Request: ``{"sql": ..., "params": [...]}`` or ``{"batch": [{"sql", "params"}, ...]}`` (a batch is
executed as one transaction). Response: ``{"success": bool, "errors": [...], "result":
[{"results": [rows], "success": bool, "meta": {"changes": n, ...}}, ...]}``.

Authentication: ``CLOUDFLARE_API_TOKEN`` (needs the *D1 Edit* permission), or a placeholder when an
egress proxy injects the Authorization header (``DATABEACON_INJECTED_AUTH=1``). D1 on the Workers Free
plan has hard daily limits (5M rows read, 100k rows written) and fails rather than charging.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Sequence
from typing import Any

import httpx

from .base import Params, StorageError

API_BASE = "https://api.cloudflare.com/client/v4"
RETRYABLE = {429, 500, 502, 503, 504}


class D1Store:
    kind = "d1"

    def __init__(self, account_id: str, database_id: str, token: str | None = None,
                 base_url: str | None = None, timeout: float = 30.0, attempts: int = 3,
                 transport: httpx.BaseTransport | None = None):
        if not account_id or not database_id:
            raise StorageError("D1 needs CLOUDFLARE_ACCOUNT_ID and DATABEACON_D1_DATABASE_ID")
        injected = os.environ.get("DATABEACON_INJECTED_AUTH", "").lower() in {"1", "true", "yes"}
        token = token or os.environ.get("DATABEACON_D1_API_TOKEN") or os.environ.get("CLOUDFLARE_API_TOKEN")
        if not token and not injected:
            raise StorageError("D1 needs an API token with D1 Edit permission (DATABEACON_D1_API_TOKEN)")
        self.url = f"{(base_url or os.environ.get('DATABEACON_D1_BASE_URL') or API_BASE).rstrip('/')}" \
                   f"/accounts/{account_id}/d1/database/{database_id}/query"
        self._headers = {"Authorization": f"Bearer {token or 'injected-by-egress-proxy'}",
                         "Content-Type": "application/json"}
        self._client = httpx.Client(timeout=timeout, transport=transport)
        self._attempts = attempts
        self._schemas: set[str] = set()
        self._lock = threading.Lock()
        self.requests = 0  # for tests / observability

    def _post(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        delay = 0.5
        last: Exception | None = None
        for attempt in range(self._attempts):
            try:
                self.requests += 1
                resp = self._client.post(self.url, json=body, headers=self._headers)
            except httpx.HTTPError as exc:  # network: retry
                last = exc
            else:
                if resp.status_code in RETRYABLE and attempt < self._attempts - 1:
                    last = StorageError(f"D1 HTTP {resp.status_code}")
                else:
                    try:
                        data = resp.json()
                    except ValueError:
                        raise StorageError(f"D1 HTTP {resp.status_code}: non-JSON response") from None
                    if resp.status_code >= 400 or not data.get("success", False):
                        errs = "; ".join(str(e.get("message", e)) for e in data.get("errors") or []) or resp.text[:300]
                        raise StorageError(f"D1 error (HTTP {resp.status_code}): {errs}")
                    results = data.get("result") or []
                    for r in results:
                        if r.get("success") is False:
                            raise StorageError(f"D1 statement failed: {r.get('error') or r}")
                    return results
            time.sleep(delay)
            delay *= 2
        raise StorageError(f"D1 request failed after {self._attempts} attempts: {last}")

    @staticmethod
    def _stmt(sql: str, params: Params) -> dict[str, Any]:
        return {"sql": sql, "params": [_jsonable(p) for p in params]}

    def query(self, sql: str, params: Params = ()) -> list[dict[str, Any]]:
        res = self._post(self._stmt(sql, params))
        return list(res[0].get("results") or []) if res else []

    def execute(self, sql: str, params: Params = ()) -> int:
        res = self._post(self._stmt(sql, params))
        return int(((res[0].get("meta") or {}).get("changes")) or 0) if res else 0

    def batch(self, statements: Sequence[tuple[str, Params]]) -> list[int]:
        if not statements:
            return []
        if len(statements) == 1:
            return [self.execute(*statements[0])]
        res = self._post({"batch": [self._stmt(s, p) for s, p in statements]})
        return [int(((r.get("meta") or {}).get("changes")) or 0) for r in res]

    def ensure_schema(self, key: str, statements: Sequence[str]) -> None:
        with self._lock:
            if key in self._schemas:
                return
            self.batch([(s, ()) for s in statements])
            self._schemas.add(key)


def _jsonable(v: Any) -> Any:
    if isinstance(v, bytes):
        raise StorageError("binary parameters are not supported over D1 HTTP; base64-encode them")
    if isinstance(v, bool):
        return int(v)
    return v
