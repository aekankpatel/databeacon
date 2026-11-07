"""Replay mode: recorded investigations visitors can explore without using the LLM.

Replays are real, completed investigations exported with ``replays/export_replays.py`` (only an
allow-list of fields; no local paths or identifiers). Serving them costs no Cloudflare neurons.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

REPLAY_DIR = Path(__file__).resolve().parents[1] / "replays"
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


@lru_cache(maxsize=1)
def _load() -> dict[str, dict[str, Any]]:
    out = {}
    for p in sorted(REPLAY_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        if _ID.fullmatch(data.get("id", "")):
            out[data["id"]] = data
    return out


def summary(r: dict[str, Any]) -> dict[str, Any]:
    from .budget import estimate_neurons

    res = r["result"]
    m = res.get("metrics") or {}
    syn = res.get("synthesis") or {}
    model = r["recorded_with"]["model"]
    return {
        "id": r["id"], "question": r["question"], "title": syn.get("title") or r["question"],
        "answer": (syn.get("answer") or {}).get("primary_entity"), "summary": syn.get("summary"),
        "recorded_at": r["recorded_at"], "recorded_with": r["recorded_with"], "dataset": r["dataset"],
        "metrics": {"latency_s": m.get("latency_s"), "total_tokens": m.get("total_tokens"),
                    "llm_calls": m.get("llm_calls"), "tool_calls": m.get("tool_calls"),
                    "approx_neurons": round(estimate_neurons(model, m.get("input_tokens", 0), m.get("output_tokens", 0)))},
        "alignment": (res.get("alignment") or {}).get("status"),
        "hallucination_rate": (res.get("verification") or {}).get("hallucination_rate"),
        "has_report": (REPLAY_DIR / f"{r['id']}.pdf").exists(),
    }


def list_replays() -> list[dict[str, Any]]:
    return [summary(r) for r in _load().values()]


def get_replay(replay_id: str) -> dict[str, Any] | None:
    if not _ID.fullmatch(replay_id or ""):
        return None
    r = _load().get(replay_id)
    return None if r is None else {**r, "summary": summary(r)}


def replay_pdf(replay_id: str) -> bytes | None:
    if not _ID.fullmatch(replay_id or "") or replay_id not in _load():
        return None
    p = REPLAY_DIR / f"{replay_id}.pdf"
    return p.read_bytes() if p.exists() else None
