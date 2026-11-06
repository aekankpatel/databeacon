"""Run the DataBeacon agent evaluation suite.

    python -m evals.run                    # live LLM (uses configured provider), all cases
    python -m evals.run --mode scripted    # deterministic offline run (CI)
    python -m evals.run --case return_rate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _gold_values(db_path: Path, sql: str) -> list[float]:
    from databeacon.sql_guard import execute_readonly

    res = execute_readonly(db_path, sql)
    out = []
    for row in res.rows:
        out += [float(v) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return out


def _resolve_entities(db_path: Path, case: dict) -> dict:
    """Expected answers derived from the data itself (one entity group per query)."""
    from databeacon.sql_guard import execute_readonly

    groups = [list(g) for g in case.get("expected_entities", [])]
    for sql in case.get("expected_entities_sql", []):
        rows = execute_readonly(db_path, sql).rows
        if rows:
            groups.append([str(rows[0][0])])
    return {**case, "expected_entities": groups}


def _markdown(summary: dict, scores: list[dict], meta: dict) -> str:
    lines = [f"# DataBeacon evaluation - {meta['timestamp']}", "",
             f"Mode: `{meta['mode']}` · Provider: `{meta['provider']}` · Model: `{meta['model']}`", "",
             "## Aggregate", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    lines += ["", "## Per case", "",
              "| Case | Success | Entities | SQL gold | Num. acc. | Tool F1 | Halluc. | Latency (s) | Tokens |",
              "|---|---|---|---|---|---|---|---|---|"]
    for s in scores:
        lines.append(f"| {s['case']} | {'✅' if s['task_success'] else '❌'} | {s['entity_recall']} | "
                     f"{s['sql_gold_match']} | {s['numerical_accuracy']} | {s['tool_selection']['f1']} | "
                     f"{s['hallucination_rate']} | {s['latency_s']} | {s['total_tokens']} |")
    return "\n".join(lines) + "\n"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "scripted"], default="live")
    ap.add_argument("--cases", default=str(ROOT / "evals" / "cases.json"))
    ap.add_argument("--case", action="append", help="only run these case ids")
    ap.add_argument("--out", default=str(ROOT / "evals" / "results"))
    ap.add_argument("--min-success", type=float, default=None, help="exit 1 if task success rate is below this")
    args = ap.parse_args()

    os.environ.setdefault("DATABEACON_DATA_DIR", tempfile.mkdtemp(prefix="databeacon_eval_"))
    from databeacon.config import get_settings
    from databeacon.datasets import DatasetRegistry
    from databeacon.llm import provider_info
    from databeacon.runs import RunManager
    from sample_data.generate_ecommerce import generate

    from .metrics import aggregate, score_case

    settings = get_settings()
    cases = json.loads(Path(args.cases).read_text())
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]
    if args.mode == "scripted":
        cases = [c for c in cases if c.get("scripted")]

    csv = Path(tempfile.mkdtemp()) / "ecommerce_orders.csv"
    generate().to_csv(csv, index=False)
    registry = DatasetRegistry(settings.datasets_dir)
    ds = registry.ingest_file(csv, name="E-commerce orders")

    factory = None
    if args.mode == "scripted":
        from databeacon.agent.testing import scripted_ecommerce_llm

        factory = scripted_ecommerce_llm
    manager = RunManager(registry, settings, llm_factory=factory, max_concurrent=1)

    scores = []
    for case in cases:
        print(f"▶ {case['id']}: {case['question']}", flush=True)
        run = await manager.run_sync(ds.id, case["question"])
        gold = _gold_values(registry.db_path(ds.id), case["gold_sql"])
        case = _resolve_entities(registry.db_path(ds.id), case)
        score = score_case(case, run, gold)
        scores.append(score)
        if score.get("error"):
            print(f"  error: {score['error'][:200]}", flush=True)
        print(f"  expected={case['expected_entities']} status={score['status']} success={score['task_success']} entities={score['entity_recall']} "
              f"gold={score['sql_gold_match']} tool_f1={score['tool_selection']['f1']} "
              f"halluc={score['hallucination_rate']} latency={score['latency_s']}s tokens={score['total_tokens']}",
              flush=True)

    summary = aggregate(scores)
    info = provider_info()
    meta = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": args.mode,
            "provider": "scripted" if args.mode == "scripted" else info["provider"],
            "model": "scripted" if args.mode == "scripted" else info["model"]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "summary": summary, "cases": scores}
    stamp = meta["timestamp"].replace(":", "").replace("-", "")
    (out / f"eval_{args.mode}_{stamp}.json").write_text(json.dumps(payload, indent=2))
    (out / "latest.json").write_text(json.dumps(payload, indent=2))
    (out / "latest.md").write_text(_markdown(summary, scores, meta))
    print(json.dumps(summary, indent=2))
    if args.min_success is not None and summary["task_success_rate"] < args.min_success:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
