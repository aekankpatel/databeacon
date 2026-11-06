"""Task specification and task-alignment verification.

The *task spec* fixes what the user asked for: the metric, the time periods (resolved
deterministically from data coverage), the grouping and the kind of comparison. After synthesis the
alignment check verifies, independently of the numeric evidence check, that

  * the answer's periods are the requested periods, the evidence it cites actually queried those
    periods, and the text does not talk about other periods;
  * the answer uses the requested metric and grouping, and so does the cited evidence;
  * the primary conclusion (e.g. "Enterprise contributed most") is what the cited evidence shows.

Numerically correct work on the wrong period, metric or grouping is therefore reported as
misaligned instead of successful.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any

from .temporal import MONTHS, TemporalResolution

CHANGE_DIRECTIONS = {"largest_decline", "largest_increase"}
HIGH = {"highest", "largest_increase", "most", "max"}
LOW = {"lowest", "largest_decline", "least", "min"}


@dataclass
class TaskSpec:
    question: str
    question_type: str = "other"
    metric_column: str | None = None
    aggregation: str | None = None
    group_by: list[str] = field(default_factory=list)
    answer_target: str = ""
    date_column: str | None = None
    current: dict[str, Any] | None = None  # {"start","end","label",...}
    previous: dict[str, Any] | None = None
    comparison: str | None = None
    assumptions: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def window(self) -> tuple[date, date] | None:
        if not self.current:
            return None
        start = date.fromisoformat((self.previous or self.current)["start"])
        return start, date.fromisoformat(self.current["end"])

    def prompt_block(self) -> str:
        lines = ["Task contract (use exactly this scope):"]
        if self.metric_column:
            lines.append(f"- metric: {self.aggregation or 'sum'}({self.metric_column})")
        if self.group_by:
            lines.append(f"- grouping: {', '.join(self.group_by)}")
        if self.current:
            lines.append(f"- current period: {self.current['label']} = [{self.current['start']}, {self.current['end']}) "
                         f"on {self.date_column}")
        if self.previous:
            lines.append(f"- comparison period: {self.previous['label']} = [{self.previous['start']}, "
                         f"{self.previous['end']})")
        if self.answer_target:
            lines.append(f"- answer: {self.answer_target}")
        for a in self.assumptions:
            lines.append(f"- assumption: {a}")
        return "\n".join(lines)


def build_spec(question: str, plan: dict[str, Any], temporal: TemporalResolution | None,
               columns: list[str]) -> TaskSpec:
    cols = {c.lower(): c for c in columns}
    spec = TaskSpec(question=question, question_type=plan.get("question_type") or "other",
                    aggregation=plan.get("aggregation") or None, answer_target=plan.get("answer_target") or "")
    metric = (plan.get("metric_column") or "").strip().strip('"')
    if metric:
        if metric.lower() in cols:
            spec.metric_column = cols[metric.lower()]
        else:
            spec.issues.append(f"plan metric '{metric}' is not a column; metric left unconstrained")
    for g in plan.get("group_by") or []:
        g = str(g).strip().strip('"')
        if g.lower() in cols:
            spec.group_by.append(cols[g.lower()])
        else:
            spec.issues.append(f"plan grouping '{g}' is not a column; ignored")
    if temporal and temporal.status == "resolved" and temporal.current:
        spec.date_column = temporal.column
        spec.current = temporal.current.to_dict()
        spec.previous = temporal.previous.to_dict() if temporal.previous else None
        spec.comparison = temporal.comparison
        spec.assumptions = list(temporal.assumptions)
    return spec


# --------------------------------------------------------------------------- helpers
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_QLABEL_RE = re.compile(r"\b(?:Q([1-4])\s*((?:19|20)\d{2})|((?:19|20)\d{2})\s*[- ]?Q([1-4]))\b", re.I)
_MLABEL_RE = re.compile(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?\s+((?:19|20)\d{2})\b", re.I)


def _dates_in(obj: Any) -> list[date]:
    text = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    out = []
    for y, m, d in _ISO_RE.findall(text):
        try:
            out.append(date(int(y), int(m), int(d)))
        except ValueError:
            continue
    return out


def _mentioned_periods(text: str) -> list[tuple[str, date, date]]:
    out = []
    for m in _QLABEL_RE.finditer(text):
        q, y = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        s = date(int(y), 3 * (int(q) - 1) + 1, 1)
        e = date(s.year + (s.month + 2) // 12, (s.month + 2) % 12 + 1, 1)
        out.append((m.group(0), s, e))
    for m in _MLABEL_RE.finditer(text):
        s = date(int(m.group(2)), MONTHS[m.group(1).lower()], 1)
        e = date(s.year + s.month // 12, s.month % 12 + 1, 1)
        out.append((m.group(0), s, e))
    return out


def _norm(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().strip("'\"").lower()


def _entity_match(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return True
    da, db = _ISO_RE.search(na), _ISO_RE.search(nb)
    if da and db:  # dates: compare at month resolution
        return da.group(1, 2) == db.group(1, 2)
    return False


def _refs(e: dict[str, Any]) -> str:
    return json.dumps({"args": e.get("args"), "sql": e.get("sql"),
                       "dims": (e.get("data") or {}).get("dimensions") if isinstance(e.get("data"), dict) else None},
                      default=str).lower()


def _expected_entity(e: dict[str, Any], dimension: str, direction: str, metric: str | None) -> tuple[Any, str] | None:
    """Recompute the primary entity from one piece of evidence, if its shape allows it."""
    data = e.get("data")
    if not isinstance(data, dict):
        return None
    if e.get("tool") == "period_comparison" and data.get("dimensions") == [dimension]:
        rows = data.get("segments") or []
        if rows:
            pick = (min if direction in LOW else max)(rows, key=lambda r: r.get("delta") or 0)
            return pick.get(dimension), f"{e['id']} ({'largest decline' if direction in LOW else 'largest increase'})"
    if e.get("tool") == "compare_groups" and data.get("group_column") == dimension and data.get("groups"):
        pick = (min if direction in LOW else max)(data["groups"], key=lambda g: g.get("mean") or 0)
        return pick.get("group"), f"{e['id']} ({'lowest' if direction in LOW else 'highest'} mean)"
    cols, rows = data.get("columns"), data.get("rows")
    if isinstance(cols, list) and rows and dimension in cols and not isinstance(cols[0], dict):
        di = cols.index(dimension)
        numeric = [i for i, c in enumerate(cols) if i != di and all(
            isinstance(r[i], (int, float)) and not isinstance(r[i], bool) for r in rows if r[i] is not None)]
        if not numeric or len({r[di] for r in rows}) != len(rows):
            return None  # one row per entity is required to rank unambiguously
        if direction in CHANGE_DIRECTIONS:
            pref = [i for i in numeric if re.search(r"delta|change|diff|growth|pct", str(cols[i]), re.I)]
        else:
            pref = [i for i in numeric if metric and metric.lower() in str(cols[i]).lower()]
        if not pref:
            if len(numeric) != 1:
                return None
            pref = numeric
        vi = pref[0]
        valid = [r for r in rows if r[vi] is not None]
        if not valid:
            return None
        pick = (min if direction in LOW else max)(valid, key=lambda r: r[vi])
        return pick[di], f"{e['id']} ({cols[vi]})"
    return None


# --------------------------------------------------------------------------- the check
def check_alignment(spec: TaskSpec, synthesis: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    answer = synthesis.get("answer") or {}
    by_id = {e["id"]: e for e in evidence if e.get("ok")}
    cited_ids = list(dict.fromkeys((answer.get("evidence_ids") or []) +
                                   [i for f in synthesis.get("findings", []) for i in (f.get("evidence_ids") or [])]))
    cited = [by_id[i] for i in cited_ids if i in by_id and by_id[i]["tool"] not in ("profile_dataset", "create_chart")]
    checks: dict[str, str] = {}
    issues: list[str] = []
    text = " ".join([synthesis.get("summary", "")] + [f.get("statement", "") for f in synthesis.get("findings", [])])

    # ---- period
    p_issues: list[str] = []
    if spec.current:
        cur_s, cur_e = date.fromisoformat(spec.current["start"]), date.fromisoformat(spec.current["end"])
        want = [spec.current["start"], spec.current["end"]]
        got = answer.get("current_period")
        ok = bool(got) and list(got)[:2] == want
        if spec.previous:
            ok = ok and list(answer.get("previous_period") or [])[:2] == [spec.previous["start"], spec.previous["end"]]
        if not ok:
            p_issues.append(f"answer periods {got}/{answer.get('previous_period')} differ from the requested "
                          f"{spec.current['label']} {want}" + (f" vs {spec.previous['label']}" if spec.previous else ""))
        win_s, win_e = spec.window()
        covering = []
        for e in cited:
            ds = _dates_in({"a": e.get("args"), "s": e.get("sql")}) + _dates_in(
                (e.get("preview") or {}).get("rows") or [])
            if not ds:
                continue
            outside = [d for d in ds if not (win_s - timedelta(days=1) <= d <= win_e)]
            if outside and not any(cur_s <= d < cur_e for d in ds):
                p_issues.append(f"{e['id']} analyses {min(ds)}..{max(ds)}, outside the requested window "
                              f"[{win_s}, {win_e})")
            if any(cur_s <= d < cur_e for d in ds) or (e["tool"] == "period_comparison" and
                                                       (e.get("args") or {}).get("current_start") == spec.current["start"]):
                covering.append(e["id"])
        if not covering:
            p_issues.append(f"no cited evidence covers the requested period {spec.current['label']}")
        stray = [lbl for lbl, s, e in _mentioned_periods(text) if not (s >= win_s and e <= win_e)]
        if stray:
            p_issues.append(f"answer text discusses periods outside the request: {', '.join(dict.fromkeys(stray))}")
        checks["period"] = "fail" if p_issues else "pass"
        issues += p_issues
    else:
        checks["period"] = "n/a"

    # ---- metric
    if spec.metric_column:
        m = spec.metric_column.lower()
        bad = []
        if answer.get("metric_column") and _norm(answer["metric_column"]) != m:
            bad.append(f"answer uses metric '{answer['metric_column']}' instead of '{spec.metric_column}'")
        if cited and not any(m in _refs(e) for e in cited):
            bad.append(f"no cited evidence uses the metric '{spec.metric_column}'")
        issues += bad
        checks["metric"] = "fail" if bad else "pass"
    else:
        checks["metric"] = "n/a"

    # ---- grouping
    if spec.group_by:
        bad = []
        dim = answer.get("dimension")
        if dim and dim not in spec.group_by:
            bad.append(f"answer is grouped by '{dim}' but the question asks for {', '.join(spec.group_by)}")
        if cited and not any(all(g.lower() in _refs(e) for g in spec.group_by) for e in cited):
            bad.append(f"no cited evidence is grouped by {', '.join(spec.group_by)}")
        issues += bad
        checks["grouping"] = "fail" if bad else "pass"
    else:
        checks["grouping"] = "n/a"

    # ---- primary conclusion
    entity, dim, direction = answer.get("primary_entity"), answer.get("dimension"), (answer.get("direction") or "")
    if entity in (None, ""):
        if spec.answer_target:
            issues.append("the answer states no primary conclusion (answer.primary_entity is empty)")
            checks["primary_conclusion"] = "fail"
        else:
            checks["primary_conclusion"] = "n/a"
    else:
        recomputed = None
        if dim and direction:
            for e in cited:
                r = _expected_entity(e, dim, direction, spec.metric_column)
                if r is not None:
                    recomputed = r
                    break
        if recomputed is not None and not _entity_match(recomputed[0], entity):
            issues.append(f"primary conclusion '{entity}' contradicts {recomputed[1]}, which points to "
                          f"'{recomputed[0]}'")
            checks["primary_conclusion"] = "fail"
        elif _norm(entity) not in _norm(synthesis.get("summary", "")) and not _entity_match(entity, synthesis.get("summary")):
            issues.append(f"the summary does not state the primary conclusion '{entity}'")
            checks["primary_conclusion"] = "fail"
        else:
            checks["primary_conclusion"] = "pass" if recomputed is not None else "unverified"
    status = "misaligned" if issues else "aligned"
    return {"status": status, "checks": checks, "issues": issues, "cited_evidence": cited_ids}
