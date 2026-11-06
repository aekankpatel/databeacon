"""Deterministic temporal scoping driven by the dataset's actual coverage.

Relative phrases ("last quarter", "this month", "past 6 months", "year over year") are resolved
against the *data*, not the wall clock: the reference point is the latest date present in the chosen
date column. A calendar period counts as complete only if the data reaches its last day, with a
tolerance equal to the column's typical spacing between dates (so weekly or monthly data is handled).

When the intended period cannot be determined reliably, the resolver returns ``needs_clarification``
with concrete options instead of guessing:
  * the question uses a time phrase but the dataset has no date column;
  * several date columns could apply and the question names none of them;
  * the latest period is partial but substantially covered (it is unclear whether the user means the
    partial period-to-date or the last complete one);
  * an explicitly requested period, or the period needed for a comparison, is outside the data.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9

# partial latest period: below this calendar coverage it is clearly "to date" and excluded;
# at or above it the user's intent is ambiguous and we ask.
PARTIAL_AMBIGUITY_THRESHOLD = 1 / 3


@dataclass
class Coverage:
    column: str
    min_date: date
    max_date: date
    distinct_dates: int
    spacing_days: int = 1  # median gap between consecutive distinct dates

    def tolerance(self) -> timedelta:
        """How far the last observed date may sit before a period's end and still count as reaching it.

        Daily data must reach the last day. Coarser data (weekly, or monthly stored as the 1st of the
        month) is complete if the last observation is within one sampling interval of the end."""
        if self.spacing_days <= 1:
            return timedelta(0)
        return timedelta(days=self.spacing_days + 1)


@dataclass
class Period:
    start: date
    end: date  # exclusive
    label: str
    complete: bool = True
    coverage_ratio: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "label": self.label,
                "complete": self.complete, "coverage_ratio": round(self.coverage_ratio, 3)}


@dataclass
class TemporalResolution:
    status: str  # resolved | not_temporal | needs_clarification
    phrase: str | None = None
    column: str | None = None
    granularity: str | None = None
    current: Period | None = None
    previous: Period | None = None
    comparison: str | None = None  # previous_period | same_period_last_year | explicit | None
    assumptions: list[str] = field(default_factory=list)
    clarification: str | None = None
    options: list[str] = field(default_factory=list)
    coverage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["current"] = self.current.to_dict() if self.current else None
        d["previous"] = self.previous.to_dict() if self.previous else None
        return d


# ----------------------------------------------------------------------------- calendar helpers
def _quarter_start(d: date) -> date:
    return date(d.year, 3 * ((d.month - 1) // 3) + 1, 1)


def _add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    return date(d.year + m // 12, m % 12 + 1, 1)


def _period_of(d: date, gran: str) -> tuple[date, date]:
    if gran == "day":
        return d, d + timedelta(days=1)
    if gran == "week":
        s = d - timedelta(days=d.weekday())
        return s, s + timedelta(days=7)
    if gran == "month":
        s = date(d.year, d.month, 1)
        return s, _add_months(s, 1)
    if gran == "quarter":
        s = _quarter_start(d)
        return s, _add_months(s, 3)
    if gran == "half":
        s = date(d.year, 1 if d.month <= 6 else 7, 1)
        return s, _add_months(s, 6)
    s = date(d.year, 1, 1)
    return s, date(d.year + 1, 1, 1)


def _shift(start: date, end: date, gran: str, n: int) -> tuple[date, date]:
    if gran in ("day", "week"):
        step = timedelta(days=1 if gran == "day" else 7) * n
        return start + step, end + step
    months = {"month": 1, "quarter": 3, "half": 6, "year": 12}[gran] * n
    return _add_months(start, months), _add_months(end, months)


def label(start: date, end: date, gran: str | None) -> str:
    if gran == "quarter":
        return f"Q{(start.month - 1) // 3 + 1} {start.year}"
    if gran == "month":
        return start.strftime("%B %Y")
    if gran == "year":
        return str(start.year)
    if gran == "half":
        return f"H{1 if start.month == 1 else 2} {start.year}"
    return f"{start.isoformat()} to {(end - timedelta(days=1)).isoformat()}"


def make_period(start: date, end: date, gran: str | None, cov: Coverage) -> Period:
    last_day = end - timedelta(days=1)
    tol = cov.tolerance()
    complete = cov.max_date + tol >= last_day and cov.min_date - tol <= start
    # each observation of coarse data stands for its whole sampling interval
    extent = cov.max_date + timedelta(days=max(0, cov.spacing_days - 1))
    covered_from, covered_to = max(start, cov.min_date), min(last_day, extent)
    span = (end - start).days
    ratio = max(0, (covered_to - covered_from).days + 1) / span if span else 0.0
    if complete:
        ratio = 1.0
    return Period(start, end, label(start, end, gran), complete, min(1.0, ratio))


def _in_coverage(p: Period, cov: Coverage) -> bool:
    return p.end > cov.min_date and p.start <= cov.max_date


# ----------------------------------------------------------------------------- coverage
def compute_coverage(db_path: Path, table: str, column: str) -> Coverage | None:
    from ..sql_guard import connect_readonly

    con = connect_readonly(db_path)
    try:
        q = '"' + column.replace('"', '""') + '"'
        row = con.execute(f'SELECT MIN({q})::DATE, MAX({q})::DATE, COUNT(DISTINCT {q}::DATE) FROM "{table}"').fetchone()
        if not row or row[0] is None:
            return None
        gap = con.execute(
            f"SELECT MEDIAN(g) FROM (SELECT (d - LAG(d) OVER (ORDER BY d)) AS g FROM "
            f"(SELECT DISTINCT {q}::DATE AS d FROM \"{table}\" WHERE {q} IS NOT NULL)) WHERE g IS NOT NULL"
        ).fetchone()[0]
    finally:
        con.close()
    spacing = int(round(float(gap))) if gap is not None else 1
    return Coverage(column, row[0], row[1], int(row[2]), max(1, spacing))


# ----------------------------------------------------------------------------- phrase parsing
_GRAN = r"(day|week|month|quarter|year|half)"
_REL_RE = re.compile(rf"\b(last|previous|prior|past|most recent|latest|this|current)\s+{_GRAN}\b", re.I)
_LASTN_RE = re.compile(rf"\b(?:last|past|previous)\s+(\d+)\s+{_GRAN}s?\b", re.I)
_Q_RE = re.compile(r"\b(?:Q([1-4])\s*['’]?\s*((?:19|20)\d{2})|((?:19|20)\d{2})\s*[- ]?\s*Q([1-4]))\b", re.I)
_H_RE = re.compile(r"\bH([12])\s*((?:19|20)\d{2})\b", re.I)
_MONTH_RE = re.compile(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?\s+((?:19|20)\d{2})\b", re.I)
_RANGE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\s*(?:to|through|until|-|–)\s*(\d{4}-\d{2}-\d{2})\b", re.I)
_YEAR_RE = re.compile(r"\b(?:in|during|for|of)\s+((?:19|20)\d{2})\b", re.I)
_YOY_RE = re.compile(r"\b(year[- ]over[- ]year|yoy|same (?:period|quarter|month) last year|vs\.? last year)\b", re.I)
_CHANGE_RE = re.compile(r"\b(declin\w*|drop\w*|fell|fall\w*|decreas\w*|increas\w*|grow\w*|grew|rose|rise|"
                        r"chang\w*|compar\w*|vs\.?|versus|differ\w*|why)\b", re.I)
_VAGUE_RE = re.compile(r"\b(recently|lately|recent)\b", re.I)


def _explicit_periods(text: str) -> list[tuple[date, date, str, str]]:
    found: list[tuple[int, date, date, str, str]] = []
    for m in _RANGE_RE.finditer(text):
        s, e = date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2)) + timedelta(days=1)
        found.append((m.start(), s, e, "range", m.group(0)))
    for m in _Q_RE.finditer(text):
        q, y = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        s = date(int(y), 3 * (int(q) - 1) + 1, 1)
        found.append((m.start(), s, _add_months(s, 3), "quarter", m.group(0)))
    for m in _H_RE.finditer(text):
        s = date(int(m.group(2)), 1 if m.group(1) == "1" else 7, 1)
        found.append((m.start(), s, _add_months(s, 6), "half", m.group(0)))
    for m in _MONTH_RE.finditer(text):
        s = date(int(m.group(2)), MONTHS[m.group(1).lower()], 1)
        found.append((m.start(), s, _add_months(s, 1), "month", m.group(0)))
    if not found:
        for m in _YEAR_RE.finditer(text):
            y = int(m.group(1))
            found.append((m.start(), date(y, 1, 1), date(y + 1, 1, 1), "year", m.group(0)))
    found.sort(key=lambda x: x[0])
    return [(s, e, g, t) for _, s, e, g, t in found]


def _pick_column(question: str, temporal_columns: list[str]) -> tuple[str | None, str | None]:
    if not temporal_columns:
        return None, None
    if len(temporal_columns) == 1:
        return temporal_columns[0], None
    q = question.lower()
    named = [c for c in temporal_columns
             if c.lower() in q or c.lower().replace("_", " ") in q
             or any(len(tok) > 3 and tok in q for tok in re.split(r"[_\W]+", c.lower()) if tok not in {"date", "time", "at"})]
    if len(named) == 1:
        return named[0], None
    return None, (f"The dataset has several date columns ({', '.join(temporal_columns)}). "
                  "Which one should define the time period?")


def resolve(question: str, temporal_columns: list[str], coverages: dict[str, Coverage],
            clarification: str | None = None) -> TemporalResolution:
    text = f"{clarification or ''} \n {question}"
    rel = _REL_RE.search(text)
    lastn = _LASTN_RE.search(text)
    explicit = _explicit_periods(clarification or "") or _explicit_periods(question)
    yoy = bool(_YOY_RE.search(text))
    vague = _VAGUE_RE.search(text)
    phrase = (lastn.group(0) if lastn else rel.group(0) if rel else
              ", ".join(t for *_, t in explicit) if explicit else vague.group(0) if vague else None)
    if phrase is None and not yoy:
        return TemporalResolution(status="not_temporal")

    col, ask = _pick_column(text, temporal_columns)
    if col is None and clarification:
        col = next((c for c in temporal_columns if c.lower() in clarification.lower()), None)
        ask = None if col else ask
    if not temporal_columns:
        return TemporalResolution(status="needs_clarification", phrase=phrase,
                                  clarification=f"The question refers to a time period ('{phrase}') but the dataset "
                                                "has no date or timestamp column. Which column holds the dates?")
    if col is None:
        return TemporalResolution(status="needs_clarification", phrase=phrase, clarification=ask,
                                  options=list(temporal_columns))
    cov = coverages.get(col)
    if cov is None:
        return TemporalResolution(status="needs_clarification", phrase=phrase, column=col,
                                  clarification=f"Column '{col}' has no usable dates.")
    res = TemporalResolution(status="resolved", phrase=phrase, column=col, coverage={
        "column": col, "min_date": cov.min_date.isoformat(), "max_date": cov.max_date.isoformat(),
        "distinct_dates": cov.distinct_dates, "spacing_days": cov.spacing_days})
    res.assumptions.append(f"Periods are relative to the data, which covers {cov.min_date} to {cov.max_date} "
                           f"in '{col}', not to today's date.")

    if explicit:
        s, e, gran, _ = explicit[-1] if len(explicit) >= 2 else explicit[0]
        if len(explicit) >= 2:  # "Q4 2025 compared with Q3 2025": later = current, earlier = previous
            periods = sorted(((s_, e_, g_) for s_, e_, g_, _ in explicit[:2]), key=lambda x: x[0])
            (ps, pe, pg), (s, e, gran) = periods
            res.previous = make_period(ps, pe, pg, cov)
            res.comparison = "explicit"
        res.granularity = gran
        res.current = make_period(s, e, gran, cov)
        if not _in_coverage(res.current, cov) or (res.previous and not _in_coverage(res.previous, cov)):
            bad = res.current if not _in_coverage(res.current, cov) else res.previous
            return TemporalResolution(
                status="needs_clarification", phrase=phrase, column=col, coverage=res.coverage,
                clarification=f"{bad.label} is outside the data, which covers {cov.min_date} to {cov.max_date}. "
                              "Which period should be analysed?",
                options=_recent_options(cov, gran if gran != "range" else "quarter"))
        if not res.current.complete:
            res.assumptions.append(f"{res.current.label} is only partially covered "
                                   f"({res.current.coverage_ratio:.0%}); results reflect data to {cov.max_date}.")
    elif lastn:
        n, gran = int(lastn.group(1)), lastn.group(2).lower()
        end = _period_of(cov.max_date, gran)[1]
        latest = make_period(*_period_of(cov.max_date, gran), gran, cov)
        if not latest.complete:
            end = latest.start
            res.assumptions.append(f"The latest {gran} ({latest.label}) is partial and excluded.")
        start = _shift(end, end, gran, -n)[0] if gran not in ("day", "week") else end - timedelta(
            days=n * (1 if gran == "day" else 7))
        res.granularity = gran
        res.current = make_period(start, end, None, cov)
        res.current.label = f"last {n} {gran}s ({res.current.label})"
        if gran in ("day", "week"):
            ps = start - (end - start)
        else:
            ps = _shift(start, start, gran, -n)[0]
        res.previous = make_period(ps, start, None, cov)
        res.comparison = "previous_period"
    else:
        gran = rel.group(2).lower() if rel else "quarter"
        word = rel.group(1).lower() if rel else "recent"
        latest = make_period(*_period_of(cov.max_date, gran), gran, cov)
        res.granularity = gran
        if word in ("this", "current"):
            res.current = latest
            if not latest.complete:
                res.assumptions.append(f"'{phrase}' = {latest.label}, which is partial "
                                       f"({latest.coverage_ratio:.0%} covered, data to {cov.max_date}).")
        elif latest.complete:
            res.current = latest
            res.assumptions.append(f"The data reaches the end of {latest.label} ({cov.max_date}), so it is the "
                                   f"most recent complete {gran} and is used as '{phrase}'.")
        elif latest.coverage_ratio < PARTIAL_AMBIGUITY_THRESHOLD:
            res.current = make_period(*_shift(latest.start, latest.end, gran, -1), gran, cov)
            res.assumptions.append(f"{latest.label} has only {latest.coverage_ratio:.0%} of its days covered "
                                   f"(data to {cov.max_date}), so '{phrase}' = {res.current.label}, the most "
                                   f"recent complete {gran}.")
        else:
            prev = make_period(*_shift(latest.start, latest.end, gran, -1), gran, cov)
            return TemporalResolution(
                status="needs_clarification", phrase=phrase, column=col, granularity=gran, coverage=res.coverage,
                clarification=(f"The data ends on {cov.max_date}, {latest.coverage_ratio:.0%} of the way through "
                               f"{latest.label}. Should '{phrase}' mean the last complete {gran} ({prev.label}) or "
                               f"{latest.label} to date?"),
                options=[prev.label, f"{latest.start.isoformat()} to {cov.max_date.isoformat()}"])
        if vague:
            res.assumptions.append(f"'{vague.group(0)}' is vague; interpreted as {res.current.label}.")

    if res.previous is None and res.current is not None:
        g = res.granularity if res.granularity in ("day", "week", "month", "quarter", "half", "year") else None
        if yoy:
            ps, pe = _add_months(res.current.start, -12), _add_months(res.current.end, -12)
            res.comparison = "same_period_last_year"
        elif g:
            ps, pe = _shift(res.current.start, res.current.end, g, -1)
            res.comparison = "previous_period"
        else:
            span = res.current.end - res.current.start
            ps, pe = res.current.start - span, res.current.start
            res.comparison = "previous_period"
        res.previous = make_period(ps, pe, g, cov)
        if not _in_coverage(res.previous, cov):
            if _CHANGE_RE.search(question):
                return TemporalResolution(
                    status="needs_clarification", phrase=phrase, column=col, coverage=res.coverage,
                    granularity=res.granularity,
                    clarification=(f"The question asks about a change in {res.current.label}, but the data starts on "
                                   f"{cov.min_date}, so there is no earlier period to compare with. Analyse "
                                   f"{res.current.label} on its own, or compare different periods?"),
                    options=[f"{res.current.label} only"] + _recent_options(cov, res.granularity or "quarter")[1:])
            res.previous = None
            res.comparison = None
            res.assumptions.append("No earlier data is available for a comparison period.")
        else:
            res.assumptions.append(f"Comparison period: {res.previous.label} "
                                   f"({'same period last year' if yoy else 'the preceding period'}).")
            if not res.previous.complete:
                res.assumptions.append(f"{res.previous.label} is only partially covered "
                                       f"({res.previous.coverage_ratio:.0%}).")
    return res


def _recent_options(cov: Coverage, gran: str) -> list[str]:
    s, e = _period_of(cov.max_date, gran)
    out = []
    for i in range(3):
        ps, pe = _shift(s, e, gran, -i)
        if pe > cov.min_date:
            out.append(label(ps, pe, gran))
    return out
