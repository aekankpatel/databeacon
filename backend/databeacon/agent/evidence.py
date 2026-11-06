"""Compact representations of tool outputs for prompts, the UI and the PDF."""

from __future__ import annotations

import json
from typing import Any

DROP_KEYS = {"figure", "series", "contingency", "sample_rows", "summary"}


def _compact(obj: Any, rows: int) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in DROP_KEYS:
                continue
            if k == "rows" and isinstance(v, list):
                out[k] = [_compact(r, rows) for r in v[:rows]]
                if len(v) > rows:
                    out["rows_omitted"] = len(v) - rows
            elif k == "segments" and isinstance(v, list):
                out[k] = [_compact(r, rows) for r in v[:rows]]
            elif k == "columns" and isinstance(v, list) and v and isinstance(v[0], dict) and "role" in v[0]:
                out[k] = [{kk: c.get(kk) for kk in ("name", "type", "role", "missing_pct", "distinct", "stats")}
                          for c in v]
            else:
                out[k] = _compact(v, rows)
        return out
    if isinstance(obj, list):
        return [_compact(v, rows) for v in obj[: max(rows, 10)]]
    if isinstance(obj, float):
        # strip float noise (1082155.9999999993) but keep what an analyst would quote
        if abs(obj) >= 100:
            return round(obj, 2)
        if abs(obj) >= 1:
            return round(obj, 4)
        return float(f"{obj:.4g}")
    return obj


def compact_json(data: Any, rows: int = 25, max_chars: int = 3500) -> str:
    text = json.dumps(_compact(data, rows), default=str, separators=(",", ":"))
    if len(text) > max_chars:
        text = json.dumps(_compact(data, min(rows, 8)), default=str, separators=(",", ":"))
    if len(text) > max_chars:
        text = text[:max_chars] + "...[truncated]"
    return text


def digest_view(tool: str, data: Any) -> Any:
    """Key fields of a tool result, sized for LLM prompts."""
    if not isinstance(data, dict):
        return data
    if tool == "period_comparison":
        keep = ("metric", "dimensions", "previous_period", "current_period", "total_previous", "total_current",
                "total_delta", "total_pct_change")
        out = {k: data.get(k) for k in keep}
        out["top_negative_contributors"] = (data.get("top_negative_contributors") or [])[:3]
        out["top_positive_contributors"] = (data.get("top_positive_contributors") or [])[:3]
        return out
    if tool == "run_sql":
        return {"columns": data.get("columns"), "rows": (data.get("rows") or [])[:15],
                "row_count": data.get("row_count")}
    if tool == "create_chart":
        return {"chart_id": data.get("chart_id"), "title": data.get("title"), "chart_type": data.get("chart_type")}
    if tool == "compare_groups":
        return {k: data.get(k) for k in ("group_column", "value_column", "groups", "tests", "effect_size",
                                         "significant", "interpretation")}
    return {k: v for k, v in data.items() if k not in DROP_KEYS and k != "sql"}


def preview_table(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    if "columns" in data and "rows" in data and data["columns"] and not isinstance(data["columns"][0], dict):
        return {"columns": data["columns"], "rows": data["rows"][:20]}
    if data.get("segments"):
        segs = data["segments"]
        cols = list(segs[0].keys())
        return {"columns": cols, "rows": [[s.get(c) for c in cols] for s in segs[:20]]}
    if isinstance(data.get("data"), dict) and "rows" in data["data"]:
        return {"columns": data["data"]["columns"], "rows": data["data"]["rows"][:20]}
    return None


def summarize(tool: str, data: Any) -> str:
    if not isinstance(data, dict):
        return str(data)[:300]
    if tool == "run_sql":
        return f"{data.get('row_count')} rows x {len(data.get('columns', []))} cols"
    if tool == "period_comparison":
        neg = data.get("top_negative_contributors") or []
        top = neg[0] if neg else {}
        dims = data.get("dimensions") or []
        label = "/".join(str(top.get(d)) for d in dims) if top else "-"
        return (f"{data.get('metric')}: {data.get('total_previous')} -> {data.get('total_current')} "
                f"({data.get('total_pct_change')}%). Largest decline: {label} ({top.get('delta')})")
    if tool == "trend_analysis":
        return (f"{data.get('direction')} trend, slope={data.get('slope_per_period')}, p={data.get('p_value')}, "
                f"change point {((data.get('change_point') or {}).get('period'))}")
    if tool == "compare_groups":
        return data.get("interpretation", "")
    if tool == "detect_anomalies":
        return f"{data.get('anomaly_count')} anomalies ({data.get('anomaly_rate_pct')}%) by {data.get('method')}"
    if tool == "create_chart":
        return f"{data.get('chart_type')} chart '{data.get('title')}'"
    if tool == "profile_dataset":
        return f"{data.get('row_count')} rows, {data.get('column_count')} columns; {len(data.get('warnings', []))} warnings"
    return compact_json(data, rows=3, max_chars=300)
