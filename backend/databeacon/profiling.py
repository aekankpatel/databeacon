"""Automatic dataset understanding: types, missing values, stats, relationships."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .sql_guard import connect_readonly

NUMERIC_PREFIXES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER",
                    "UBIGINT", "FLOAT", "DOUBLE", "REAL", "DECIMAL")
TEMPORAL_PREFIXES = ("DATE", "TIMESTAMP", "TIME")


def column_role(dtype: str, distinct: int, rows: int, name: str) -> str:
    d = dtype.upper()
    lname = name.lower()
    if d.startswith(TEMPORAL_PREFIXES):
        return "temporal"
    if d == "BOOLEAN":
        return "categorical"
    if d.startswith(NUMERIC_PREFIXES):
        if (lname.endswith("_id") or lname == "id") and not d.startswith(("FLOAT", "DOUBLE", "DECIMAL", "REAL")):
            return "identifier"
        if distinct <= 12 and rows > 50 and not d.startswith(("FLOAT", "DOUBLE", "DECIMAL", "REAL")):
            return "categorical"
        return "numeric"
    if lname.endswith("_id") or lname == "id" or (rows > 50 and distinct > 0.9 * rows):
        return "identifier"
    return "categorical" if distinct <= max(50, 0.05 * rows) else "text"


def _f(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if np.isnan(v) or np.isinf(v) else round(v, 6)
    if isinstance(v, np.integer):
        return int(v)
    if hasattr(v, "isoformat"):
        if hasattr(v, "hour") and (v.hour, v.minute, v.second) == (0, 0, 0) and hasattr(v, "date"):
            return v.date().isoformat()
        return v.isoformat()
    return v


def profile_table(db_path: Path, table: str, *, top_k: int = 5) -> dict[str, Any]:
    con = connect_readonly(db_path)
    try:
        rows = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        cols = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
            [table],
        ).fetchall()
        columns: list[dict[str, Any]] = []
        for name, dtype in cols:
            q = f'"{name}"'
            nulls, distinct = con.execute(
                f"SELECT COUNT(*) - COUNT({q}), approx_count_distinct({q}) FROM \"{table}\""
            ).fetchone()
            role = column_role(dtype, int(distinct or 0), rows, name)
            info: dict[str, Any] = {
                "name": name,
                "type": dtype,
                "role": role,
                "missing": int(nulls),
                "missing_pct": round(100.0 * nulls / rows, 2) if rows else 0.0,
                "distinct": int(distinct or 0),
            }
            if dtype.upper().startswith(NUMERIC_PREFIXES):
                stats = con.execute(
                    f"SELECT MIN({q}), MAX({q}), AVG({q}), STDDEV_SAMP({q}), MEDIAN({q}), "
                    f"QUANTILE_CONT({q}, 0.25), QUANTILE_CONT({q}, 0.75), SUM({q}) FROM \"{table}\""
                ).fetchone()
                keys = ["min", "max", "mean", "std", "median", "p25", "p75", "sum"]
                info["stats"] = {k: _f(v) for k, v in zip(keys, stats)}
                p25, p75 = stats[5], stats[6]
                if p25 is not None and p75 is not None:
                    iqr = float(p75) - float(p25)
                    lo, hi = float(p25) - 1.5 * iqr, float(p75) + 1.5 * iqr
                    info["outliers_iqr"] = int(
                        con.execute(f'SELECT COUNT(*) FROM "{table}" WHERE {q} < ? OR {q} > ?', [lo, hi]).fetchone()[0]
                    )
            elif dtype.upper().startswith(TEMPORAL_PREFIXES):
                mn, mx = con.execute(f'SELECT MIN({q}), MAX({q}) FROM "{table}"').fetchone()
                info["stats"] = {"min": _f(mn), "max": _f(mx)}
            if role in ("categorical", "text", "identifier") or dtype.upper() == "BOOLEAN":
                top = con.execute(
                    f'SELECT {q}::VARCHAR AS v, COUNT(*) AS n FROM "{table}" GROUP BY 1 ORDER BY 2 DESC LIMIT {top_k}'
                ).fetchall()
                info["top_values"] = [{"value": v, "count": int(n)} for v, n in top]
            columns.append(info)

        sample = con.execute(f'SELECT * FROM "{table}" LIMIT 5').df()
        numeric_cols = [c["name"] for c in columns if c["role"] == "numeric"][:15]
        correlations: list[dict[str, Any]] = []
        if len(numeric_cols) >= 2:
            cols_sql = ", ".join(f'"{c}"' for c in numeric_cols)
            df = con.execute(f'SELECT {cols_sql} FROM "{table}" USING SAMPLE 50000 ROWS').df()
            corr = df.corr(numeric_only=True)
            for i, a in enumerate(numeric_cols):
                for b in numeric_cols[i + 1:]:
                    v = corr.loc[a, b] if a in corr.index and b in corr.columns else np.nan
                    if pd.notna(v) and abs(v) >= 0.5:
                        correlations.append({"a": a, "b": b, "pearson_r": round(float(v), 3)})
            correlations.sort(key=lambda x: -abs(x["pearson_r"]))
    finally:
        con.close()

    warnings = []
    for c in columns:
        if c["missing_pct"] >= 20:
            warnings.append(f"Column '{c['name']}' is {c['missing_pct']}% missing.")
        if c["distinct"] == 1:
            warnings.append(f"Column '{c['name']}' is constant.")
    candidate_keys = [c["name"] for c in columns if c["distinct"] >= 0.98 * rows and c["missing"] == 0]
    return {
        "table": table,
        "row_count": int(rows),
        "column_count": len(columns),
        "columns": columns,
        "temporal_columns": [c["name"] for c in columns if c["role"] == "temporal"],
        "numeric_columns": [c["name"] for c in columns if c["role"] == "numeric"],
        "categorical_columns": [c["name"] for c in columns if c["role"] == "categorical"],
        "candidate_keys": candidate_keys,
        "strong_correlations": correlations[:10],
        "warnings": warnings,
        "sample_rows": [{k: _f(v) for k, v in r.items()} for r in sample.to_dict(orient="records")],
    }


def schema_summary(profile: dict[str, Any]) -> str:
    """Compact text description of a profile for LLM prompts."""
    lines = [f'Table "{profile["table"]}" - {profile["row_count"]:,} rows, {profile["column_count"]} columns.']
    for c in profile["columns"]:
        extra = ""
        st = c.get("stats") or {}
        if c["role"] == "numeric" and st:
            extra = f" min={st.get('min')} max={st.get('max')} mean={st.get('mean')}"
        elif c["role"] == "temporal" and st:
            extra = f" range={st.get('min')} .. {st.get('max')}"
        elif c.get("top_values"):
            vals = ", ".join(str(t["value"]) for t in c["top_values"][:5])
            extra = f" e.g. [{vals}]"
        miss = f" missing={c['missing_pct']}%" if c["missing"] else ""
        lines.append(f'  - "{c["name"]}" {c["type"]} ({c["role"]}, {c["distinct"]} distinct){miss}{extra}')
    if profile.get("strong_correlations"):
        lines.append("Strong correlations: " + "; ".join(
            f"{x['a']}~{x['b']} r={x['pearson_r']}" for x in profile["strong_correlations"][:5]))
    if profile.get("warnings"):
        lines.append("Data quality: " + " ".join(profile["warnings"][:5]))
    return "\n".join(lines)
