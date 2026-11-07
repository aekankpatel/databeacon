"""Export recorded investigations as replay files (developer tool).

    python replays/export_replays.py <runs.sqlite> <reports dir> <slug>=<run_id> [...]

Writes replays/<slug>.json and replays/<slug>.pdf. Only an allow-list of fields is kept; local paths,
owners, session data and account identifiers are dropped, and the export fails if anything that looks
like one remains.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEEP = ("plan", "reflections", "synthesis", "verification", "charts", "evidence", "timeline", "metrics",
        "errors", "stop_reason", "iterations", "profile", "temporal", "spec", "alignment")
FORBIDDEN = [re.compile(p, re.I) for p in (r"/tmp/", r"/home/", r"scratchpad", r"\bclaude\b", r"[0-9a-f]{32}",
                                           r"\.sqlite", r"owner")]


def _strip_paths(obj):
    if isinstance(obj, dict):
        return {k: _strip_paths(v) for k, v in obj.items() if k not in {"path", "owner", "fsPath"}}
    if isinstance(obj, list):
        return [_strip_paths(v) for v in obj]
    if isinstance(obj, str) and re.search(r"(/tmp/|/home/)", obj):
        return re.sub(r"(/tmp|/home)/[^\s'\"]+", "<local path>", obj)
    return obj


def export(db: Path, reports: Path, slug: str, run_id: str) -> dict:
    row = sqlite3.connect(db).execute("SELECT question, status, created_at, result FROM runs WHERE id = ?",
                                      (run_id,)).fetchone()
    if not row:
        raise SystemExit(f"run {run_id} not found")
    question, status, created_at, result = row
    if status != "completed":
        raise SystemExit(f"run {run_id} is {status}; only completed runs are exported")
    res = json.loads(result)
    out = {k: res.get(k) for k in KEEP}
    out["report"] = {"ok": True}
    prof = res.get("profile") or {}
    replay = _strip_paths({
        "id": slug, "question": question, "recorded_at": created_at, "status": status,
        "dataset": {"name": "E-commerce orders (sample)", "table": prof.get("table"),
                    "row_count": prof.get("row_count"), "column_count": prof.get("column_count")},
        "recorded_with": {"provider": "cloudflare", "model": "@cf/openai/gpt-oss-120b",
                          "agent_budgets": {"max_iterations": 2, "max_steps_per_iteration": 6}},
        "result": out,
    })
    text = json.dumps(replay, separators=(",", ":"), default=str)
    for pat in FORBIDDEN:
        m = pat.search(text)
        if m:
            raise SystemExit(f"{slug}: refusing to export, found {m.group(0)!r} near {text[max(0, m.start()-60):m.end()+20]!r}")
    (HERE / f"{slug}.json").write_text(text)
    pdf = reports / f"{run_id}.pdf"
    if pdf.exists():
        (HERE / f"{slug}.pdf").write_bytes(pdf.read_bytes())
    return {"slug": slug, "bytes": len(text), "pdf": pdf.exists()}


if __name__ == "__main__":
    db, reports = Path(sys.argv[1]), Path(sys.argv[2])
    for pair in sys.argv[3:]:
        slug, run_id = pair.split("=", 1)
        print(export(db, reports, slug, run_id))
