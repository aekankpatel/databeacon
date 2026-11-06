"""Scoring functions for agent runs.

A run only counts as a success if every independent check passes:
  * the run completed and the agent's own task-alignment check passed;
  * primary conclusion  - the structured answer names the gold entity (and the summary states it);
                          merely mentioning the right entity somewhere does not count;
  * period              - the answer's current/previous periods equal the gold periods;
  * metric / grouping   - the answer is measured on the gold metric and grouped by the gold dimension;
  * numerical evidence  - the gold numbers appear in the evidence the answer cites;
  * hallucination       - at most 10% of numeric claims unsupported.
"""

from __future__ import annotations

import json
import re
from typing import Any

from databeacon.agent.alignment import _entity_match
from databeacon.agent.verifier import collect_numbers, extract_claims

HALLUCINATION_LIMIT = 0.1


def tool_selection(used: list[str], expected: list[str], acceptable: list[str]) -> dict[str, float]:
    used_set = set(used)
    exp = set(expected)
    allowed = exp | set(acceptable)
    recall = len(used_set & exp) / len(exp) if exp else 1.0
    precision = len(used_set & allowed) / len(used_set) if used_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def numbers_in(gold_values: list[float], evidence: list[dict[str, Any]], tol: float = 0.01) -> bool:
    """True if every gold value appears (within tolerance, or as a percentage) in the given evidence."""
    if not gold_values:
        return True
    pool = collect_numbers([e.get("data") for e in evidence if e.get("ok")])
    pool += [v * 100 for v in pool]
    return all(any(abs(p - g) <= max(abs(g) * tol, 0.011) for p in pool) for g in gold_values)


def numbers_in_text(gold_values: list[float], text: str, tol: float = 0.015) -> float:
    vals = [abs(c.value) for c in extract_claims(text)]
    if not gold_values:
        return 1.0
    hit = sum(1 for g in map(abs, gold_values)
              if any(abs(v - g) <= max(g * tol, 0.011) or abs(v - g * 100) <= max(g * 100 * tol, 0.11) for v in vals))
    return hit / len(gold_values)


def trajectory(result: dict[str, Any]) -> dict[str, Any]:
    tl = [e for e in result.get("timeline", []) if e.get("type") == "tool"]
    names = [e["name"] for e in tl]
    seen, dups = set(), 0
    for e in tl:
        key = e["name"] + json.dumps(e.get("input"), sort_keys=True, default=str)
        dups += key in seen
        seen.add(key)
    errors = [i for i, e in enumerate(tl) if e["status"] == "error"]
    recovered = sum(1 for i in errors if i + 1 < len(tl) and tl[i + 1]["status"] == "ok")
    decisions = [e["name"] for e in result.get("timeline", []) if e.get("type") in ("decision", "guardrail")]
    return {
        "profiled_first": bool(names) and names[0] == "profile_dataset",
        "duplicate_calls": dups,
        "duplicates_blocked": decisions.count("duplicate_call_blocked"),
        "tool_errors": len(errors),
        "errors_recovered": recovered,
        "replans": decisions.count("replan"),
        "realignments": decisions.count("realign_to_question"),
        "revisions": decisions.count("revise_answer"),
    }


def _check(ok: bool | None) -> str:
    return "n/a" if ok is None else ("pass" if ok else "fail")


def score_case(case: dict[str, Any], run: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    """``gold`` holds resolved values: entity, dimension, metric_column, current_period, previous_period, numbers."""
    result = run.get("result") or {}
    evidence = result.get("evidence", [])
    syn = result.get("synthesis") or {}
    answer = syn.get("answer") or {}
    ver = result.get("verification") or {}
    metrics = result.get("metrics") or {}
    used = [e["tool"] for e in evidence]
    sql_calls = [e for e in evidence if e["tool"] == "run_sql"]
    sql_valid = sum(1 for e in sql_calls if e["ok"]) / len(sql_calls) if sql_calls else None

    cited_ids = set(answer.get("evidence_ids") or []) | {i for f in syn.get("findings", []) for i in f.get("evidence_ids") or []}
    cited = [e for e in evidence if e["id"] in cited_ids]

    checks: dict[str, str] = {}
    checks["completed"] = _check(run.get("status") in ("completed", "completed_with_issues"))
    checks["alignment"] = _check((result.get("alignment") or {}).get("status") == "aligned")
    if gold.get("entity") is not None:
        named = _entity_match(answer.get("primary_entity"), gold["entity"])
        stated = bool(re.search(re.escape(str(gold["entity"])), syn.get("summary", ""), re.I)) or \
            _entity_match(syn.get("summary"), gold["entity"])
        checks["primary_conclusion"] = _check(named and stated)
    else:
        checks["primary_conclusion"] = "n/a"
    if gold.get("current_period"):
        ok = list(answer.get("current_period") or [])[:2] == gold["current_period"]
        if gold.get("previous_period"):
            ok = ok and list(answer.get("previous_period") or [])[:2] == gold["previous_period"]
        checks["period"] = _check(ok)
    else:
        checks["period"] = "n/a"
    checks["metric"] = _check(None if not gold.get("metric_column")
                              else str(answer.get("metric_column") or "").lower() == gold["metric_column"].lower())
    checks["grouping"] = _check(None if not gold.get("dimension")
                                else str(answer.get("dimension") or "").lower() == gold["dimension"].lower())
    checks["numerical_evidence"] = _check(numbers_in(gold.get("numbers") or [], cited))
    halluc = ver.get("hallucination_rate") if ver else None
    checks["hallucination"] = _check(halluc is not None and halluc <= HALLUCINATION_LIMIT)

    success = all(v in ("pass", "n/a") for v in checks.values()) and checks["completed"] == "pass"
    ts = tool_selection(used, case.get("expected_tools", []), case.get("acceptable_tools", []))
    text = " ".join([syn.get("summary", "")] + [f.get("statement", "") for f in syn.get("findings", [])])
    return {
        "case": case["id"], "status": run.get("status"), "task_success": success, "checks": checks,
        "expected": {k: gold.get(k) for k in ("entity", "dimension", "metric_column", "current_period", "previous_period")},
        "got": {k: answer.get(k) for k in ("primary_entity", "dimension", "metric_column", "current_period",
                                           "previous_period")},
        "error": (result.get("errors") or [None])[-1] if run.get("status") == "failed" else None,
        "alignment_issues": (result.get("alignment") or {}).get("issues", []),
        "sql_validity": None if sql_valid is None else round(sql_valid, 3),
        "numbers_stated": round(numbers_in_text(gold.get("numbers") or [], text), 3),
        "tool_selection": ts, "hallucination_rate": halluc,
        "findings_verified": f"{ver.get('findings_verified', 0)}/{ver.get('findings_total', 0)}",
        "latency_s": metrics.get("latency_s"), "total_tokens": metrics.get("total_tokens"),
        "tokens_by_stage": metrics.get("tokens_by_stage"),
        "llm_calls": metrics.get("llm_calls"), "tool_calls": metrics.get("tool_calls"),
        "iterations": result.get("iterations"), "trajectory": trajectory(result), "tools_used": used,
        "expect_success": case.get("expect_success", True),
    }


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(scores) or 1
    real = [s for s in scores if s.get("expect_success", True)]
    nr = len(real) or 1

    def rate(check: str, pool=real):
        vals = [s["checks"][check] for s in pool if s["checks"][check] != "n/a"]
        return round(sum(v == "pass" for v in vals) / len(vals), 3) if vals else None

    def mean(key, sub=None, pool=real):
        vals = [(s[key][sub] if sub else s[key]) for s in pool]
        vals = [float(v) for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "cases": len(real),
        "task_success_rate": round(sum(s["task_success"] for s in real) / nr, 3),
        "primary_conclusion_rate": rate("primary_conclusion"),
        "period_correct_rate": rate("period"),
        "metric_correct_rate": rate("metric"),
        "grouping_correct_rate": rate("grouping"),
        "numerical_evidence_rate": rate("numerical_evidence"),
        "alignment_rate": rate("alignment"),
        "sql_validity": mean("sql_validity"),
        "tool_selection_f1": mean("tool_selection", "f1"),
        "hallucination_rate": mean("hallucination_rate"),
        "avg_latency_s": mean("latency_s"),
        "avg_total_tokens": mean("total_tokens"),
        "avg_tool_calls": mean("tool_calls"),
        "error_recovery_rate": (
            round(sum(s["trajectory"]["errors_recovered"] for s in real)
                  / max(1, sum(s["trajectory"]["tool_errors"] for s in real)), 3)
        ),
        # calibration: deliberately wrong scripted runs must be scored as failures
        "scorer_calibration": round(sum(s["task_success"] == s.get("expect_success", True) for s in scores) / n, 3),
    }
