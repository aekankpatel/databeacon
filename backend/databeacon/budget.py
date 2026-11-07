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
    """Neuron ledger and guard. Backed by a local SQLite file (``path``) or any :class:`SQLStore`
    (Cloudflare D1 for deployments with several instances)."""

    SCHEMA = [
        """CREATE TABLE IF NOT EXISTS llm_usage (
            day TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
            calls INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0, neurons REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (day, provider, model))""",
        """CREATE TABLE IF NOT EXISTS llm_calls (
            ts REAL NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, neurons REAL NOT NULL,
            synthetic INTEGER NOT NULL DEFAULT 0)""",
        "CREATE INDEX IF NOT EXISTS idx_llm_calls_ts ON llm_calls(provider, ts)",
        """CREATE TABLE IF NOT EXISTS budget_reservations (
            id TEXT PRIMARY KEY, neurons REAL NOT NULL, created REAL NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS idx_budget_res_created ON budget_reservations(created)",
        """CREATE TABLE IF NOT EXISTS provider_blocks (
            provider TEXT PRIMARY KEY, until_ts REAL NOT NULL, reason TEXT NOT NULL)""",
    ]

    def __init__(self, path: Path | None = None, daily_limit: float | None = None, window: str | None = None,
                 store=None):
        from .storage.base import SQLiteStore, StorageError

        if store is None:
            if path is None:
                raise ValueError("NeuronBudget needs a path or a store")
            store = SQLiteStore(Path(path))
        self.store = store
        self._StorageError = StorageError
        limit = float(daily_limit if daily_limit is not None
                      else os.environ.get("DATABEACON_CLOUDFLARE_DAILY_NEURONS", 9_000))
        allow_paid = os.environ.get("DATABEACON_ALLOW_PAID_INFERENCE", "").lower() in {"1", "true", "yes"}
        if limit > FREE_DAILY_NEURONS and not allow_paid:
            limit = float(FREE_DAILY_NEURONS)  # never above the free allocation without an explicit opt-in
        self.daily_limit = limit
        self.window = (window or os.environ.get("DATABEACON_NEURON_WINDOW", "rolling24h")).lower()
        if self.window not in {"rolling24h", "utc_day"}:
            self.window = "rolling24h"
        try:
            store.ensure_schema("budget", self.SCHEMA)
            self._migrate()
        except StorageError as exc:  # fail closed later: reserve()/blocked() will refuse
            self._init_error = exc
        else:
            self._init_error = None

    @classmethod
    def default(cls) -> NeuronBudget:
        """The deployment's ledger: shared D1 when configured, else ``<data dir>/llm_usage.sqlite``."""
        from .storage import open_store

        return cls(store=open_store("llm_usage"))

    def _migrate(self) -> None:
        """Older local ledgers: count each day's total at the end of that day (conservative), and drop
        the old integer-id reservation table if present."""
        s = self.store
        have = {r["d"] for r in s.query("SELECT DISTINCT date(ts, 'unixepoch') AS d FROM llm_calls")}
        rows = s.query("SELECT day, provider, model, neurons FROM llm_usage")
        stmts = []
        for r in rows:
            if r["day"] in have or not r["neurons"]:
                continue
            end = datetime.fromisoformat(r["day"]).replace(tzinfo=UTC).timestamp() + 86399
            stmts.append(("INSERT INTO llm_calls (ts, provider, model, neurons, synthetic) VALUES (?, ?, ?, ?, 1)",
                          (end, r["provider"], r["model"], r["neurons"])))
        if stmts:
            s.batch(stmts)

    def _check(self) -> None:
        if self._init_error is not None:
            raise BudgetUnavailable(f"neuron ledger unavailable: {self._init_error}")

    def _since(self) -> float:
        if self.window == "utc_day":
            return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        return time.time() - WINDOW_S

    def used(self, provider: str = "cloudflare") -> float:
        self._check()
        rows = self.store.query("SELECT COALESCE(SUM(neurons), 0) AS n FROM llm_calls WHERE provider=? AND ts > ?",
                                (provider, self._since()))
        return float(rows[0]["n"]) if rows else 0.0

    # kept for callers: "today" means the guard's accounting window
    used_today = used

    def _reserved(self) -> float:
        rows = self.store.query("SELECT COALESCE(SUM(neurons), 0) AS n FROM budget_reservations WHERE created > ?",
                                (time.time() - RESERVATION_TTL_S,))
        return float(rows[0]["n"]) if rows else 0.0

    def remaining(self) -> float:
        return max(0.0, self.daily_limit - self.used() - self._reserved())

    # --------------------------------------------------------------- provider blocks
    def block_provider(self, provider: str, reason: str, seconds: float = PROVIDER_BLOCK_S) -> None:
        self.store.execute("INSERT OR REPLACE INTO provider_blocks (provider, until_ts, reason) VALUES (?, ?, ?)",
                           (provider, time.time() + seconds, reason[:500]))

    def blocked(self, provider: str = "cloudflare") -> str | None:
        self._check()
        rows = self.store.query("SELECT until_ts, reason FROM provider_blocks WHERE provider=?", (provider,))
        if rows and rows[0]["until_ts"] > time.time():
            until = datetime.fromtimestamp(rows[0]["until_ts"], UTC).strftime("%H:%M UTC")
            return f"{rows[0]['reason']} (no {provider} calls until {until})"
        return None

    def reserve(self, neurons: float) -> str | None:
        """Atomically reserve a worst-case amount: one conditional INSERT, so concurrent callers (threads,
        processes or serverless instances sharing D1) cannot jointly exceed the budget. Returns a
        reservation id, or None if it would exceed the budget. Raises BudgetUnavailable if the ledger
        cannot be used (callers must then not call Cloudflare)."""
        import uuid

        self._check()
        now = time.time()
        rid = uuid.uuid4().hex
        try:
            changed = self.store.execute(
                """INSERT INTO budget_reservations (id, neurons, created)
                   SELECT ?, ?, ?
                   WHERE (SELECT COALESCE(SUM(neurons), 0) FROM llm_calls WHERE provider = 'cloudflare' AND ts > ?)
                       + (SELECT COALESCE(SUM(neurons), 0) FROM budget_reservations WHERE created > ?)
                       + ? <= ?""",
                (rid, float(neurons), now, self._since(), now - RESERVATION_TTL_S, float(neurons), self.daily_limit))
        except self._StorageError as exc:
            raise BudgetUnavailable(f"neuron ledger unavailable: {exc}") from exc
        return rid if changed == 1 else None

    def release(self, reservation_id: str | None) -> None:
        if reservation_id is None:
            return
        try:
            self.store.execute("DELETE FROM budget_reservations WHERE id=?", (reservation_id,))
        except self._StorageError:
            pass  # expires after RESERVATION_TTL_S anyway

    def record(self, provider: str, model: str, input_tokens: int, output_tokens: int,
               reported_neurons: float | None = None, reservation_id: str | None = None) -> float:
        """Record one call (and release its reservation) atomically. Cloudflare reports the neurons it
        charged; otherwise we estimate from tokens."""
        if provider != "cloudflare":
            neurons = 0.0
        elif reported_neurons is not None:
            neurons = float(reported_neurons)
        else:
            neurons = estimate_neurons(model, input_tokens, output_tokens)
        stmts = [
            ("""INSERT INTO llm_usage (day, provider, model, calls, input_tokens, output_tokens, neurons)
                VALUES (?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(day, provider, model) DO UPDATE SET calls = calls + 1,
                  input_tokens = input_tokens + excluded.input_tokens,
                  output_tokens = output_tokens + excluded.output_tokens,
                  neurons = neurons + excluded.neurons""",
             (_today(), provider, model, int(input_tokens), int(output_tokens), neurons)),
            ("INSERT INTO llm_calls (ts, provider, model, neurons) VALUES (?, ?, ?, ?)",
             (time.time(), provider, model, neurons)),
        ]
        if reservation_id:
            stmts.append(("DELETE FROM budget_reservations WHERE id=?", (reservation_id,)))
        self.store.batch(stmts)
        return neurons

    def purge(self) -> None:
        """Drop ledger rows that can no longer affect any window (kept 3 days for reporting)."""
        cutoff = time.time() - 3 * 86400
        self.store.batch([("DELETE FROM llm_calls WHERE ts < ?", (cutoff,)),
                          ("DELETE FROM budget_reservations WHERE created < ?", (time.time() - RESERVATION_TTL_S,))])

    def summary(self) -> dict:
        rows = self.store.query("SELECT provider, model, calls, input_tokens, output_tokens, neurons FROM llm_usage "
                                "WHERE day=? ORDER BY provider, model", (_today(),))
        return {
            "day_utc": _today(),
            "window": self.window,
            "storage": getattr(self.store, "kind", "sqlite"),
            "cloudflare_neurons_used": round(self.used(), 1),  # within the accounting window
            "cloudflare_neurons_used_today_utc": round(sum(r["neurons"] for r in rows if r["provider"] == "cloudflare"), 1),
            "cloudflare_neuron_budget": self.daily_limit,
            "cloudflare_free_allocation": FREE_DAILY_NEURONS,
            "cloudflare_blocked": self.blocked(),
            "by_model": [{"provider": r["provider"], "model": r["model"], "calls": r["calls"],
                          "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                          "neurons": round(r["neurons"], 1)} for r in rows],
        }
