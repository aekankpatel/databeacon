"""Scoring functions for agent runs."""

from __future__ import annotations

import json
from typing import Any

from databeacon.agent.verifier import collect_numbers, extract_claims


def _text(result: dict[str, Any]) -> str:
    syn = result.get("synthesis") or {}
    parts = [syn.get("title", ""), syn.get("summary", "")]
    parts += [f"{f.get('title', '')} {f.get('statement', '')}" for f in syn.get("findings", [])]
    return "\n".join(parts)


def tool_selection(used: list[str], expected: list[str], acceptable: list[str]) -> dict[str, float]:
    used_set = set(used)
    exp = set(expected)
    allowed = exp | set(acceptable)
    recall = len(used_set & exp) / len(exp) if exp else 1.0
    precision = len(used_set & allowed) / len(used_set) if used_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def gold_match(gold_values: list[float], evidence: list[dict[str, Any]], tol: float = 0.01) -> bool:
    """True if every gold value appears (within relative tolerance) in some successful tool output."""
    pool = collect_numbers([e.get("data") for e in evidence if e.get("ok") and e.get("tool") != "profile_dataset"])
    pool += [v * 100 for v in pool]
    for g in gold_values:
        if not any(abs(p - g) <= max(abs(g) * tol, 0.011) for p in pool):
            return False
    return True


def numbers_in_text(gold_values: list[float], text: str, tol: float = 0.015) -> float:
    claims = extract_claims(text)
    vals = [abs(c.value) for c in claims]
    if not gold_values:
        return 1.0
    hit = 0
    for g in gold_values:
        g = abs(g)
        if any(abs(v - g) <= max(g * tol, 0.011) or abs(v - g * 100) <= max(g * 100 * tol, 0.11) for v in vals):
            hit += 1
    return hit / len(gold_values)


def entities_found(groups: list[list[str]], text: str) -> float:
    low = text.lower()
    if not groups:
        return 1.0
    return sum(1 for g in groups if any(e.lower() in low for e in g)) / len(groups)


def trajectory(result: dict[str, Any]) -> dict[str, Any]:
    tl = [e for e in result.get("timeline", []) if e.get("type") == "tool"]
    names = [e["name"] for e in tl]
    seen, dups = set(), 0
    for e in tl:
        key = e["name"] + json.dumps(e.get("input"), sort_keys=True, default=str)
        dups += key in seen
        seen.add(key)
    errors = [i for i, e in enumerate(tl) if e["status"] == "error"]
    # recovered: the agent's next tool call after a failure succeeded
    recovered = sum(1 for i in errors if i + 1 < len(tl) and tl[i + 1]["status"] == "ok")
    decisions = [e["name"] for e in result.get("timeline", []) if e.get("type") == "decision"]
    return {
        "profiled_first": bool(names) and names[0] == "profile_dataset",
        "duplicate_calls": dups,
        "tool_errors": len(errors),
        "errors_recovered": recovered,
        "replans": decisions.count("replan"),
        "revisions": decisions.count("revise_unsupported_claims"),
    }


def score_case(case: dict[str, Any], run: dict[str, Any], gold_values: list[float]) -> dict[str, Any]:
    result = run.get("result") or {}
    evidence = result.get("evidence", [])
    used = [e["tool"] for e in evidence]
    sql_calls = [e for e in evidence if e["tool"] == "run_sql"]
    sql_valid = sum(1 for e in sql_calls if e["ok"]) / len(sql_calls) if sql_calls else None
    text = _text(result)
    ver = result.get("verification") or {}
    metrics = result.get("metrics") or {}
    ts = tool_selection(used, case.get("expected_tools", []), case.get("acceptable_tools", []))
    ent = entities_found(case.get("expected_entities", []), text)
    num_acc = numbers_in_text(gold_values, text)
    gm = gold_match(gold_values, evidence)
    halluc = ver.get("hallucination_rate", 1.0 if not ver else 0.0)
    completed = run.get("status") == "completed"
    success = completed and ent == 1.0 and gm and halluc <= 0.1
    return {
        "case": case["id"], "status": run.get("status"), "task_success": success,
        "entity_recall": round(ent, 3), "sql_validity": None if sql_valid is None else round(sql_valid, 3),
        "sql_gold_match": gm, "numerical_accuracy": round(num_acc, 3), "tool_selection": ts,
        "hallucination_rate": halluc, "findings_verified": f"{ver.get('findings_verified', 0)}/{ver.get('findings_total', 0)}",
        "latency_s": metrics.get("latency_s"), "total_tokens": metrics.get("total_tokens"),
        "llm_calls": metrics.get("llm_calls"), "tool_calls": metrics.get("tool_calls"),
        "iterations": result.get("iterations"), "trajectory": trajectory(result), "tools_used": used,
    }


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(scores) or 1

    def mean(key, sub=None):
        vals = [(s[key][sub] if sub else s[key]) for s in scores]
        vals = [float(v) for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "cases": len(scores),
        "task_success_rate": round(sum(s["task_success"] for s in scores) / n, 3),
        "sql_validity": mean("sql_validity"),
        "sql_gold_match_rate": round(sum(s["sql_gold_match"] for s in scores) / n, 3),
        "tool_selection_f1": mean("tool_selection", "f1"),
        "numerical_accuracy": mean("numerical_accuracy"),
        "entity_recall": mean("entity_recall"),
        "hallucination_rate": mean("hallucination_rate"),
        "avg_latency_s": mean("latency_s"),
        "avg_total_tokens": mean("total_tokens"),
        "avg_tool_calls": mean("tool_calls"),
        "error_recovery_rate": (
            round(sum(s["trajectory"]["errors_recovered"] for s in scores)
                  / max(1, sum(s["trajectory"]["tool_errors"] for s in scores)), 3)
        ),
    }
