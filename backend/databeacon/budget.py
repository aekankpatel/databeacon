"""Free-tier guard for Cloudflare Workers AI.

Workers AI bills in *neurons*; the free allocation is 10,000 neurons per day. Cloudflare documents a
reset at 00:00 UTC, but we observed the allocation still exhausted shortly after midnight by the
previous evening's usage. The guard therefore counts usage over a **rolling 24-hour window** by default
(``DATABEACON_NEURON_WINDOW=rolling24h``), which is conservative under either behaviour;
``utc_day`` restores calendar-day accounting.

Every call is recorded with a timestamp and the neurons Cloudflare reports (or an estimate from the
published per-model rates). A call is only made if its worst-case cost fits under the budget, and
when Cloudflare itself reports the allocation exhausted (error 4006) the provider is blocked for a
cool-down instead of being retried.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

# neurons per 1M tokens (input, output), per developers.cloudflare.com/workers-ai/platform/pricing.
# gpt-oss-120b verified Nov 2026; re-check the others against the pricing page before relying on them.
NEURON_RATES: dict[str, tuple[float, float]] = {
    "@cf/openai/gpt-oss-120b": (31_818, 68_182),
    "@cf/openai/gpt-oss-20b": (18_182, 27_273),
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast": (26_668, 204_805),
    "@cf/meta/llama-4-scout-17b-16e-instruct": (24_545, 77_273),
    "@cf/meta/llama-3.1-8b-instruct-fast": (4_119, 34_868),
}
# unknown models are charged at the most expensive known rate, to stay on the safe side
DEFAULT_RATE = (max(r[0] for r in NEURON_RATES.values()), max(r[1] for r in NEURON_RATES.values()))
FREE_DAILY_NEURONS = 10_000


def estimate_neurons(model: str, input_tokens: int, output_tokens: int) -> float:
    rin, rout = NEURON_RATES.get(model, DEFAULT_RATE)
    return input_tokens * rin / 1e6 + output_tokens * rout / 1e6


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


class BudgetUnavailable(RuntimeError):
    """The usage ledger cannot be read or written; Cloudflare must not be called (fail closed)."""


RESERVATION_TTL_S = 600
WINDOW_S = 24 * 3600
PROVIDER_BLOCK_S = 3600


def worst_case_neurons(model: str, prompt_chars: int, max_output_tokens: int) -> float:
    """Upper bound for one call: the prompt (~4 chars/token, +25% slack for tool schemas/formatting)
    plus the full output allowance."""
    return estimate_neurons(model, int(prompt_chars / 4 * 1.25) + 64, max_output_tokens)


class NeuronBudget:
    def __init__(self, path: Path, daily_limit: float | None = None, window: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        limit = float(daily_limit if daily_limit is not None
                      else os.environ.get("DATABEACON_CLOUDFLARE_DAILY_NEURONS", 9_000))
        allow_paid = os.environ.get("DATABEACON_ALLOW_PAID_INFERENCE", "").lower() in {"1", "true", "yes"}
        if limit > FREE_DAILY_NEURONS and not allow_paid:
            limit = float(FREE_DAILY_NEURONS)  # never above the free allocation without an explicit opt-in
        self.daily_limit = limit
        self.window = (window or os.environ.get("DATABEACON_NEURON_WINDOW", "rolling24h")).lower()
        if self.window not in {"rolling24h", "utc_day"}:
            self.window = "rolling24h"
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS reservations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT NOT NULL, neurons REAL NOT NULL,
                created REAL NOT NULL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS llm_usage (
                day TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
                calls INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0, neurons REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (day, provider, model))""")
            c.execute("""CREATE TABLE IF NOT EXISTS llm_calls (
                ts REAL NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, neurons REAL NOT NULL,
                synthetic INTEGER NOT NULL DEFAULT 0)""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_ts ON llm_calls(ts)")
            c.execute("""CREATE TABLE IF NOT EXISTS provider_blocks (
                provider TEXT PRIMARY KEY, until_ts REAL NOT NULL, reason TEXT NOT NULL)""")
            self._migrate(c)

    @staticmethod
    def _migrate(c: sqlite3.Connection) -> None:
        """Ledgers from before per-call timestamps: count each day's total at the end of that day
        (the latest it could have happened), so the rolling window stays conservative."""
        have = {r[0] for r in c.execute("SELECT DISTINCT date(ts, 'unixepoch') FROM llm_calls")}
        for day, provider, model, neurons in c.execute(
                "SELECT day, provider, model, neurons FROM llm_usage").fetchall():
            if day in have or not neurons:
                continue
            end = datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() + 86399
            c.execute("INSERT INTO llm_calls (ts, provider, model, neurons, synthetic) VALUES (?, ?, ?, ?, 1)",
                      (end, provider, model, neurons))

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def _since(self) -> float:
        if self.window == "utc_day":
            return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        return time.time() - WINDOW_S

    def used(self, provider: str = "cloudflare", c: sqlite3.Connection | None = None) -> float:
        q = "SELECT COALESCE(SUM(neurons), 0) FROM llm_calls WHERE provider=? AND ts > ?"
        if c is not None:
            return float(c.execute(q, (provider, self._since())).fetchone()[0])
        with self._conn() as c2:
            return float(c2.execute(q, (provider, self._since())).fetchone()[0])

    # kept for callers: "today" means the guard's accounting window
    used_today = used

    def remaining(self) -> float:
        return max(0.0, self.daily_limit - self.used() - self._reserved())

    def _reserved(self, c: sqlite3.Connection | None = None) -> float:
        q = "SELECT COALESCE(SUM(neurons), 0) FROM reservations WHERE created > ?"
        args = (time.time() - RESERVATION_TTL_S,)
        if c is not None:
            return float(c.execute(q, args).fetchone()[0])
        with self._conn() as c2:
            return float(c2.execute(q, args).fetchone()[0])

    # --------------------------------------------------------------- provider blocks
    def block_provider(self, provider: str, reason: str, seconds: float = PROVIDER_BLOCK_S) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO provider_blocks (provider, until_ts, reason) VALUES (?, ?, ?)",
                      (provider, time.time() + seconds, reason[:500]))

    def blocked(self, provider: str = "cloudflare") -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT until_ts, reason FROM provider_blocks WHERE provider=?", (provider,)).fetchone()
        if row and row[0] > time.time():
            until = datetime.fromtimestamp(row[0], UTC).strftime("%H:%M UTC")
            return f"{row[1]} (no {provider} calls until {until})"
        return None

    def reserve(self, neurons: float) -> int | None:
        """Atomically reserve a worst-case amount. Returns a reservation id, or None if it would exceed the
        budget. Raises BudgetUnavailable if the ledger cannot be used (callers must then not call Cloudflare)."""
        try:
            with self._lock:
                c = self._conn()
                try:
                    c.execute("BEGIN IMMEDIATE")
                    if self.used(c=c) + self._reserved(c) + neurons > self.daily_limit:
                        c.execute("ROLLBACK")
                        return None
                    cur = c.execute("INSERT INTO reservations (day, neurons, created) VALUES (?, ?, ?)",
                                    (_today(), float(neurons), time.time()))
                    c.execute("COMMIT")
                    return int(cur.lastrowid)
                finally:
                    c.close()
        except sqlite3.Error as exc:
            raise BudgetUnavailable(f"neuron ledger unavailable: {exc}") from exc

    def release(self, reservation_id: int | None) -> None:
        if reservation_id is None:
            return
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM reservations WHERE id=?", (reservation_id,))

    def record(self, provider: str, model: str, input_tokens: int, output_tokens: int,
               reported_neurons: float | None = None) -> float:
        """Record one call. Cloudflare reports the neurons it charged; otherwise we estimate from tokens."""
        if provider != "cloudflare":
            neurons = 0.0
        elif reported_neurons is not None:
            neurons = float(reported_neurons)
        else:
            neurons = estimate_neurons(model, input_tokens, output_tokens)
        with self._lock, self._conn() as c:
            c.execute("""INSERT INTO llm_usage (day, provider, model, calls, input_tokens, output_tokens, neurons)
                         VALUES (?, ?, ?, 1, ?, ?, ?)
                         ON CONFLICT(day, provider, model) DO UPDATE SET calls = calls + 1,
                           input_tokens = input_tokens + excluded.input_tokens,
                           output_tokens = output_tokens + excluded.output_tokens,
                           neurons = neurons + excluded.neurons""",
                      (_today(), provider, model, int(input_tokens), int(output_tokens), neurons))
            c.execute("INSERT INTO llm_calls (ts, provider, model, neurons) VALUES (?, ?, ?, ?)",
                      (time.time(), provider, model, neurons))
        return neurons

    def summary(self) -> dict:
        with self._conn() as c:
            rows = c.execute("SELECT provider, model, calls, input_tokens, output_tokens, neurons FROM llm_usage "
                             "WHERE day=? ORDER BY provider, model", (_today(),)).fetchall()
        return {
            "day_utc": _today(),
            "window": self.window,
            "cloudflare_neurons_used": round(self.used(), 1),  # within the accounting window
            "cloudflare_neurons_used_today_utc": round(sum(r[5] for r in rows if r[0] == "cloudflare"), 1),
            "cloudflare_neuron_budget": self.daily_limit,
            "cloudflare_free_allocation": FREE_DAILY_NEURONS,
            "cloudflare_blocked": self.blocked(),
            "by_model": [{"provider": r[0], "model": r[1], "calls": r[2], "input_tokens": r[3],
                          "output_tokens": r[4], "neurons": round(r[5], 1)} for r in rows],
        }
