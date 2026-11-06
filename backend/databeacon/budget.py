"""Free-tier guard for Cloudflare Workers AI.

Workers AI bills in *neurons*; the free allocation is 10,000 neurons per day (reset at 00:00 UTC).
We estimate neurons from token usage with Cloudflare's published per-model rates, persist the
running total per UTC day in SQLite, and refuse further Cloudflare calls once the configured budget
is reached (the provider chain then falls back, e.g. to Groq).
"""

from __future__ import annotations

import os
import sqlite3
import threading
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


class NeuronBudget:
    def __init__(self, path: Path, daily_limit: float | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_limit = float(daily_limit if daily_limit is not None
                                 else os.environ.get("DATABEACON_CLOUDFLARE_DAILY_NEURONS", 9_000))
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS llm_usage (
                day TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
                calls INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0, neurons REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (day, provider, model))""")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def used_today(self, provider: str = "cloudflare") -> float:
        with self._conn() as c:
            row = c.execute("SELECT COALESCE(SUM(neurons), 0) FROM llm_usage WHERE day=? AND provider=?",
                            (_today(), provider)).fetchone()
        return float(row[0])

    def remaining(self) -> float:
        return max(0.0, self.daily_limit - self.used_today())

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
        return neurons

    def summary(self) -> dict:
        with self._conn() as c:
            rows = c.execute("SELECT provider, model, calls, input_tokens, output_tokens, neurons FROM llm_usage "
                             "WHERE day=? ORDER BY provider, model", (_today(),)).fetchall()
        used = sum(r[5] for r in rows if r[0] == "cloudflare")
        return {
            "day_utc": _today(),
            "cloudflare_neurons_used": round(used, 1),
            "cloudflare_neuron_budget": self.daily_limit,
            "cloudflare_free_allocation": FREE_DAILY_NEURONS,
            "by_model": [{"provider": r[0], "model": r[1], "calls": r[2], "input_tokens": r[3],
                          "output_tokens": r[4], "neurons": round(r[5], 1)} for r in rows],
        }
