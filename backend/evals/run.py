"""Run the DataBeacon agent evaluation suite.

    python -m evals.run --mode scripted                    # deterministic offline run (CI), incl. calibration
    python -m evals.run --case revenue_decline_attribution --max-neurons 2500   # small live retest
    python -m evals.run                                    # live, all benchmark cases (configured provider)

Calibration cases are scripted runs that are wrong on purpose (wrong quarter, wrong conclusion); the
scorer must fail them. They run only in scripted mode.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# first-case estimate for --max-neurons (the most expensive case measured so far used ~3,000 neurons);
# later cases use 1.25 x the most expensive case seen in this run
EST_FIRST_CASE_NEURONS = 3000


def _numbers(db_path: Path, sql: str) -> list[float]:
    from databeacon.sql_guard import execute_readonly

    out: list[float] = []
    for row in execute_readonly(db_path, sql).rows:
        out += [float(v) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return out


def resolve_gold(db_path: Path, case: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Gold answers come from SQL over the dataset wherever possible (not hand-written labels)."""
    from databeacon.sql_guard import execute_readonly

    g = dict(case.get("gold") or {})
    if "ref" in g:
        g = {**by_id[g.pop("ref")].get("gold", {}), **g}
    if g.get("entity_sql"):
        rows = execute_readonly(db_path, g.pop("entity_sql")).rows
        g["entity"] = str(rows[0][0]) if rows else None
    g["numbers"] = _numbers(db_path, g.pop("numbers_sql")) if g.get("numbers_sql") else []
    return g


def _markdown(summary: dict, scores: list[dict], meta: dict) -> str:
    lines = [f"# DataBeacon evaluation - {meta['timestamp']}", "",
             f"Mode: `{meta['mode']}` · Provider: `{meta['provider']}` · Model: `{meta['model']}` · "
             f"Budgets: `{meta.get('budgets')}`", "",
             "A case succeeds only if every check passes: primary conclusion (structured answer), period, metric, "
             "grouping, gold numbers in cited evidence, task alignment and hallucination <= 10%.", "",
             "## Aggregate", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    lines += ["", "## Per case", "",
              "| Case | Expected | Success | Conclusion | Period | Metric | Grouping | Numbers | Aligned | Halluc. | "
              "Tokens |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in scores:
        c = s["checks"]
        exp = "fail" if not s.get("expect_success", True) else "pass"
        lines.append(f"| {s['case']} | {exp} | {'✅' if s['task_success'] else '❌'} | {c['primary_conclusion']} | "
                     f"{c['period']} | {c['metric']} | {c['grouping']} | {c['numerical_evidence']} | {c['alignment']} | "
                     f"{s['hallucination_rate']} | {s['total_tokens']} |")
    return "\n".join(lines) + "\n"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "scripted"], default="live")
    ap.add_argument("--cases", default=str(ROOT / "evals" / "cases.json"))
    ap.add_argument("--case", action="append", help="only run these case ids")
    ap.add_argument("--out", default=str(ROOT / "evals" / "results"))
    ap.add_argument("--min-success", type=float, default=None, help="exit 1 if task success rate is below this")
    ap.add_argument("--min-calibration", type=float, default=None,
                    help="exit 1 if the scorer misjudges any deliberately wrong scripted run")
    ap.add_argument("--max-neurons", type=float, default=None,
                    help="live mode: hard cap on Cloudflare neurons for this run (enforced per call)")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="live mode: allow a configured fallback provider (off by default: the benchmark should "
                         "measure one provider and never switch to possibly paid inference)")
    args = ap.parse_args()

    os.environ.setdefault("DATABEACON_DATA_DIR", tempfile.mkdtemp(prefix="databeacon_eval_"))
    from databeacon.budget import NeuronBudget
    from databeacon.config import get_settings
    from databeacon.datasets import DatasetRegistry
    from databeacon.llm import provider_info
    from databeacon.runs import RunManager
    from sample_data.generate_ecommerce import generate

    from .metrics import aggregate, score_case

    settings = get_settings()
    all_cases = json.loads(Path(args.cases).read_text())
    by_id = {c["id"]: c for c in all_cases}
    cases = all_cases
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]
    if args.mode == "scripted":
        cases = [c for c in cases if "scripted" in c]
    else:
        cases = [c for c in cases if not c.get("calibration")]

    csv = Path(tempfile.mkdtemp()) / "ecommerce_orders.csv"
    generate().to_csv(csv, index=False)
    registry = DatasetRegistry(settings.datasets_dir)
    ds = registry.ingest_file(csv, name="E-commerce orders")
    usage_db = NeuronBudget(settings.data_dir / "llm_usage.sqlite")
    budget_meta: dict[str, Any] = {}
    if args.mode == "live":
        if os.environ.get("DATABEACON_FALLBACK_PROVIDER") and not args.allow_fallback:
            print("Refusing to run: a fallback provider is configured (use --allow-fallback to permit it).")
            return 2
        if os.environ.get("DATABEACON_ALLOW_PAID_INFERENCE", "").lower() in {"1", "true", "yes"}:
            print("Refusing to run: DATABEACON_ALLOW_PAID_INFERENCE is set; benchmarks must stay on the free tier.")
            return 2
        used0 = usage_db.used_today()
        hard_limit = usage_db.daily_limit
        if args.max_neurons is not None:
            # lower the guard's daily limit for this process: no call may push today's total past it
            hard_limit = min(hard_limit, used0 + args.max_neurons)
            os.environ["DATABEACON_CLOUDFLARE_DAILY_NEURONS"] = str(hard_limit)
            usage_db = NeuronBudget(settings.data_dir / "llm_usage.sqlite")
        budget_meta = {"neurons_used_today_before": round(used0, 1), "run_cap": args.max_neurons,
                       "hard_daily_limit_for_run": hard_limit, "free_allocation": 10_000}
        print(f"Budget: {used0:.0f} neurons used today; this run may spend at most {hard_limit - used0:.0f}.",
              flush=True)

    scores = []
    incomplete = None
    spent = 0.0
    max_case = 0.0
    for case in cases:
        if args.mode == "live":
            est = EST_FIRST_CASE_NEURONS if not max_case else max(1500.0, 1.25 * max_case)
            remaining = usage_db.remaining()
            if est > remaining:
                incomplete = (f"stopped before '{case['id']}': estimated {est:.0f} neurons for the case, "
                              f"{remaining:.0f} left under the cap")
                print(f"  {incomplete}", flush=True)
                break
        factory = None
        if args.mode == "scripted":
            from databeacon.agent.testing import scripted_ecommerce_llm

            kw = case.get("scripted") or {}
            factory = lambda kw=kw: scripted_ecommerce_llm(**kw)  # noqa: E731
        manager = RunManager(registry, settings, llm_factory=factory, max_concurrent=1)
        print(f"▶ {case['id']}: {case['question']}", flush=True)
        neurons_before = usage_db.used_today()
        run = await manager.run_sync(ds.id, case["question"])
        neurons = round(usage_db.used_today() - neurons_before, 1)
        spent += neurons
        max_case = max(max_case, neurons)
        gold = resolve_gold(registry.db_path(ds.id), case, by_id)
        score = score_case(case, run, gold)
        score["cloudflare_neurons"] = neurons
        score["served_by"] = ((run.get("result") or {}).get("metrics") or {}).get("served_by")
        if score.get("error") and "QuotaExhaustedError" in score["error"]:
            incomplete = f"stopped at '{case['id']}': LLM provider quota exhausted"
            print(f"  {incomplete}", flush=True)
            break
        scores.append(score)
        if score.get("error"):
            print(f"  error: {score['error'][:200]}", flush=True)
        verdict = "ok" if score["task_success"] == score["expect_success"] else "SCORER MISJUDGED"
        print(f"  success={score['task_success']} (expected {score['expect_success']}: {verdict}) "
              f"checks={score['checks']} tokens={score['total_tokens']} neurons={neurons}", flush=True)
        if score["alignment_issues"]:
            print(f"  alignment: {score['alignment_issues'][:3]}", flush=True)

    summary = aggregate(scores)
    summary["total_cloudflare_neurons"] = round(sum(s["cloudflare_neurons"] for s in scores), 1)
    info = provider_info()
    meta = {"timestamp": datetime.now(UTC).isoformat(timespec="seconds"), "mode": args.mode,
            "provider": "scripted" if args.mode == "scripted" else info["provider"],
            "model": "scripted" if args.mode == "scripted" else info["model"],
            "budgets": {"max_iterations": settings.max_iterations, "max_tool_calls": settings.max_tool_calls,
                        "max_steps_per_iteration": settings.max_steps_per_iteration},
            "incomplete": incomplete, "budget": {**budget_meta, "neurons_spent": round(spent, 1)}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "summary": summary, "cases": scores}
    stamp = meta["timestamp"].replace(":", "").replace("-", "")
    (out / f"eval_{args.mode}_{stamp}.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    if incomplete:
        print(f"Suite incomplete ({incomplete}); latest results left unchanged.")
        return 2
    name = "latest" if args.mode == "live" else "latest_scripted"
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2))
    (out / f"{name}.md").write_text(_markdown(summary, scores, meta))
    if args.min_success is not None and summary["task_success_rate"] < args.min_success:
        return 1
    if args.min_calibration is not None and summary["scorer_calibration"] < args.min_calibration:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
