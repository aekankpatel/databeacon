"""Request-level protections for public deployments.

* **Visitor isolation** - each browser generates a random session token (sent as the
  ``X-DataBeacon-Session`` header, or ``?session=`` for EventSource/download links that cannot set
  headers). Datasets, runs and reports are bound to a hash of that token; other sessions get 404.
  This is anonymous isolation, not authentication: the token is a bearer secret held by the browser.
* **Rate limits** - in-memory sliding windows per client IP. They are per process: a multi-instance
  deployment gets per-instance limits (the neuron budget guard still caps LLM spend).
* **Security headers** and an optional shared **demo access code** for LLM-spending endpoints.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from ..config import Settings

SESSION_HEADER = "X-DataBeacon-Session"
ACCESS_CODE_HEADER = "X-DataBeacon-Access-Code"
_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{22,128}$")
LOCAL_OWNER = "local"


def session_owner(request: Request, settings: Settings) -> str:
    """Owner id for the caller. Public mode requires a well-formed session token."""
    token = request.headers.get(SESSION_HEADER) or request.query_params.get("session")
    if not token:
        if settings.public_mode:
            raise HTTPException(401, "Missing session. Reload the page to start a new session.")
        return LOCAL_OWNER
    if not _SESSION_RE.fullmatch(token):
        raise HTTPException(400, "Malformed session token.")
    return hashlib.sha256(token.encode()).hexdigest()[:24]


def client_ip(request: Request, settings: Settings) -> str:
    if settings.trust_proxy:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()[:64]
        real = request.headers.get("x-real-ip")
        if real:
            return real.strip()[:64]
    return request.client.host if request.client else "unknown"


def check_access_code(request: Request, settings: Settings) -> None:
    if not settings.demo_access_code:
        return
    given = request.headers.get(ACCESS_CODE_HEADER, "")
    if not hmac.compare_digest(given.encode(), settings.demo_access_code.encode()):
        raise HTTPException(403, "This demo needs an access code to run live AI analyses.")


class RateLimiter:
    """Sliding-window limiter: ``hit(key, limit, window_s)`` raises 429 when the limit is exceeded."""

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window_s: float, what: str) -> None:
        if limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            q = self._events[key]
            while q and q[0] <= now - window_s:
                q.popleft()
            if len(q) >= limit:
                retry = int(q[0] + window_s - now) + 1
                raise HTTPException(429, f"Rate limit: at most {limit} {what} per "
                                         f"{_human(window_s)}. Try again in {_human(retry)}.",
                                    headers={"Retry-After": str(retry)})
            q.append(now)
            if len(self._events) > 50_000:  # bound memory under abuse
                for k in list(self._events)[:10_000]:
                    if not self._events[k] or self._events[k][-1] <= now - 86400:
                        self._events.pop(k, None)

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


def _human(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 86400:
        return f"{seconds // 86400} day" + ("s" if seconds >= 172800 else "")
    if seconds >= 3600:
        return f"{seconds // 3600} hour" + ("s" if seconds >= 7200 else "")
    if seconds >= 60:
        return f"{seconds // 60} min"
    return f"{seconds} s"


# Content Security Policy for the bundled frontend (also in frontend/vercel.json). Plotly needs inline styles.
FRONTEND_CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self' https:; worker-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
}
