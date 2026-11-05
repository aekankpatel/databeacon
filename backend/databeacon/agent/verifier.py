"""Evidence-based verification of generated findings.

Every number stated in a finding must be traceable to the outputs of tools that
were actually executed (directly, or as a simple derived quantity such as a
difference, ratio or percentage change between two evidence values).
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_DATE_RE = re.compile(r"\b\d{4}-\d{2}(-\d{2})?\b")
_QUARTER_RE = re.compile(r"\b(?:Q[1-4]\s*(?:'|20)?\d{0,4}|(?:19|20)\d{2}\s*-?\s*Q[1-4]|H[12]|FY\s?\d{2,4})\b", re.IGNORECASE)
_EVIDENCE_REF_RE = re.compile(r"\[?\bE\d+\b\]?")
_NUM_RE = re.compile(
    r"(?P<sign>[-−+])?\s?(?P<cur>[$€£])?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s?(?P<suf>%|percent\b|pp\b|percentage points\b|[kKmMbB](?![a-zA-Z])|million\b|billion\b|thousand\b)?"
)
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}


@dataclass
class Claim:
    text: str
    value: float
    is_percent: bool
    tolerance: float


def extract_claims(statement: str) -> list[Claim]:
    s = _EVIDENCE_REF_RE.sub(" ", statement)
    s = _DATE_RE.sub(" ", s)
    s = _QUARTER_RE.sub(" ", s)
    claims: list[Claim] = []
    for m in _NUM_RE.finditer(s):
        raw = m.group("num")
        start = m.start("num")
        if start > 0 and (s[start - 1].isalpha() or s[start - 1] == "_"):
            continue  # identifiers like "segment3"
        value = float(raw.replace(",", ""))
        suf = (m.group("suf") or "").lower().strip()
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        is_percent = suf in {"%", "percent", "pp", "percentage points"}
        mult = _MULT.get(suf, 1.0)
        has_unit = bool(m.group("cur") or suf)
        if not has_unit and decimals == 0 and (1900 <= value <= 2100 or value <= 10):
            continue  # years, small counts like "3 segments"
        if m.group("sign") in {"-", "−"} and (m.group("cur") or is_percent or start - m.start() <= 2):
            value = -value
        unit = 10 ** (-decimals) * mult
        tol = max(0.5 * unit, abs(value * mult) * 0.005)
        claims.append(Claim(m.group(0).strip(), value * mult, is_percent, tol))
    return claims


def collect_numbers(obj: Any, out: list[float] | None = None, limit: int = 20000) -> list[float]:
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append(float(obj))
    elif isinstance(obj, str):
        if len(obj) < 40:
            try:
                out.append(float(obj.replace(",", "")))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in {"figure"}:  # plotly figure duplicates chart data
                continue
            collect_numbers(v, out, limit)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_numbers(v, out, limit)
    return out


def _derived(values: list[float], cap: int = 250) -> Iterable[float]:
    vals = [v for v in dict.fromkeys(values) if v != 0][:cap]
    for a, b in itertools.permutations(vals, 2):
        yield a - b
        yield 100.0 * (a - b) / b
        yield 100.0 * a / b
        yield a / b
    for a in vals:
        yield a * 100.0
        yield a / 100.0


def _matches(claim: Claim, candidates: Iterable[float]) -> bool:
    t = abs(claim.value)
    tol = claim.tolerance
    for c in candidates:
        c = abs(c)
        if abs(c - t) <= tol or (claim.is_percent and abs(c * 100 - t) <= tol):
            return True
    return False


def verify_finding(statement: str, evidence_values: list[float], cited_values: list[float]) -> dict[str, Any]:
    """Direct matches may come from any evidence; derived values (differences, ratios, % changes) only from
    the evidence the finding cites - otherwise almost any number could be 'derived' from a large pool."""
    claims = extract_claims(statement)
    supported, unsupported = [], []
    derived_cache: list[float] | None = None
    for cl in claims:
        ok = _matches(cl, cited_values) or _matches(cl, evidence_values)
        if not ok and cited_values:
            if derived_cache is None:
                derived_cache = list(_derived(cited_values))
            ok = _matches(cl, derived_cache)
        (supported if ok else unsupported).append(cl.text)
    n = len(claims)
    if n == 0:
        status = "qualitative"
    elif not unsupported:
        status = "verified"
    elif len(supported) >= len(unsupported):
        status = "partially_verified"
    else:
        status = "unverified"
    return {"status": status, "checked_numbers": n, "supported": len(supported),
            "supported_values": supported, "unsupported_values": unsupported}


def verify_findings(findings: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {e["id"]: e for e in evidence if e.get("ok")}
    all_values = collect_numbers([e.get("data") for e in by_id.values()])
    total_claims = total_supported = 0
    for f in findings:
        cited = [i for i in (f.get("evidence_ids") or []) if i in by_id]
        missing = [i for i in (f.get("evidence_ids") or []) if i not in by_id]
        cited_values = collect_numbers([by_id[i].get("data") for i in cited]) if cited else []
        text = f"{f.get('statement', '')}"
        res = verify_finding(text, all_values, cited_values)
        res["missing_evidence_ids"] = missing
        if not cited and res["status"] != "qualitative":
            res["note"] = "No valid evidence cited; matched against all evidence."
        f["verification"] = res
        total_claims += res["checked_numbers"]
        total_supported += res["supported"]
    unsupported = total_claims - total_supported
    return {
        "findings_total": len(findings),
        "findings_verified": sum(1 for f in findings if f["verification"]["status"] == "verified"),
        "numeric_claims": total_claims,
        "numeric_claims_supported": total_supported,
        "hallucination_rate": round(unsupported / total_claims, 4) if total_claims else 0.0,
        "unsupported": [
            {"finding": f.get("title"), "values": f["verification"]["unsupported_values"]}
            for f in findings if f["verification"]["unsupported_values"]
        ],
    }
