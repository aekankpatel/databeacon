"""Statistical analysis routines (pure functions on pandas DataFrames)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _r(v: Any, nd: int = 4) -> Any:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if np.isnan(f) or np.isinf(f):
        return None
    return round(f, nd)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not in query result (columns: {', '.join(map(str, df.columns))})")
    return pd.to_numeric(df[col], errors="coerce")


def describe(df: pd.DataFrame) -> dict[str, Any]:
    num = df.select_dtypes(include="number")
    out = {}
    for c in num.columns:
        s = num[c].dropna()
        if s.empty:
            continue
        out[c] = {
            "count": int(s.count()), "mean": _r(s.mean()), "std": _r(s.std()), "min": _r(s.min()),
            "p25": _r(s.quantile(0.25)), "median": _r(s.median()), "p75": _r(s.quantile(0.75)),
            "max": _r(s.max()), "sum": _r(s.sum()), "skew": _r(s.skew()), "missing": int(num[c].isna().sum()),
        }
    return {"rows": len(df), "columns": out}


def correlation(df: pd.DataFrame, method: str = "pearson") -> dict[str, Any]:
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be pearson, spearman or kendall")
    num = df.select_dtypes(include="number")
    cols = list(num.columns)
    if len(cols) < 2:
        raise ValueError("Need at least two numeric columns for correlation analysis")
    pairs = []
    fn = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            sub = num[[a, b]].dropna()
            if len(sub) < 3 or sub[a].nunique() < 2 or sub[b].nunique() < 2:
                continue
            r, p = fn(sub[a], sub[b])
            pairs.append({"a": a, "b": b, "coefficient": _r(r), "p_value": _r(p, 6), "n": len(sub),
                          "strength": _strength(abs(float(r)))})
    pairs.sort(key=lambda x: -abs(x["coefficient"] or 0))
    return {"method": method, "pairs": pairs}


def _strength(r: float) -> str:
    return "very strong" if r >= 0.8 else "strong" if r >= 0.6 else "moderate" if r >= 0.4 else "weak" if r >= 0.2 else "negligible"


def compare_groups(df: pd.DataFrame, group_col: str, value_col: str, alpha: float = 0.05) -> dict[str, Any]:
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not in query result")
    vals = _num(df, value_col)
    data = pd.DataFrame({"g": df[group_col].astype(str), "v": vals}).dropna()
    groups = {k: g["v"].to_numpy() for k, g in data.groupby("g") if len(g) >= 2}
    if len(groups) < 2:
        raise ValueError("Need at least two groups with 2+ observations each")
    summary = [{"group": k, "n": len(v), "mean": _r(v.mean()), "median": _r(np.median(v)), "std": _r(v.std(ddof=1)),
                "sum": _r(v.sum())} for k, v in groups.items()]
    summary.sort(key=lambda x: -(x["mean"] or 0))
    arrays = list(groups.values())
    if len(groups) == 2:
        a, b = arrays
        t, p = stats.ttest_ind(a, b, equal_var=False)
        u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0
        tests = [
            {"test": "Welch t-test", "statistic": _r(t), "p_value": _r(p, 6)},
            {"test": "Mann-Whitney U", "statistic": _r(u), "p_value": _r(pu, 6)},
        ]
        effect = {"cohens_d": _r(d)}
        primary_p = p
    else:
        f, p = stats.f_oneway(*arrays)
        h, ph = stats.kruskal(*arrays)
        grand = np.concatenate(arrays)
        ss_between = sum(len(x) * (x.mean() - grand.mean()) ** 2 for x in arrays)
        ss_total = ((grand - grand.mean()) ** 2).sum()
        tests = [
            {"test": "One-way ANOVA", "statistic": _r(f), "p_value": _r(p, 6)},
            {"test": "Kruskal-Wallis H", "statistic": _r(h), "p_value": _r(ph, 6)},
        ]
        effect = {"eta_squared": _r(ss_between / ss_total if ss_total else 0.0)}
        primary_p = p
    return {
        "group_column": group_col, "value_column": value_col, "groups": summary, "tests": tests,
        "effect_size": effect, "alpha": alpha, "significant": bool(primary_p < alpha),
        "interpretation": (
            f"Differences in {value_col} across {group_col} are "
            f"{'statistically significant' if primary_p < alpha else 'not statistically significant'} "
            f"(p={_r(primary_p, 6)})."
        ),
    }


def chi_square(df: pd.DataFrame, col_a: str, col_b: str) -> dict[str, Any]:
    for c in (col_a, col_b):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not in query result")
    table = pd.crosstab(df[col_a], df[col_b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Both columns need at least two categories")
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {"chi2": _r(chi2), "p_value": _r(p, 6), "dof": int(dof), "cramers_v": _r(v), "n": int(n),
            "significant": bool(p < 0.05), "contingency": table.head(20).to_dict()}


def detect_anomalies(df: pd.DataFrame, value_col: str, method: str = "zscore", threshold: float = 3.0,
                     label_col: str | None = None, max_items: int = 25) -> dict[str, Any]:
    vals = _num(df, value_col)
    s = vals.dropna()
    if len(s) < 5:
        raise ValueError("Need at least 5 observations for anomaly detection")
    if method == "zscore":
        z = (vals - s.mean()) / (s.std(ddof=1) or 1)
        mask = z.abs() > threshold
        score = z
        bounds = {"mean": _r(s.mean()), "std": _r(s.std(ddof=1)), "threshold_z": threshold}
    elif method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        k = threshold if threshold < 3.0 else 1.5
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask = (vals < lo) | (vals > hi)
        score = (vals - s.median()) / (iqr or 1)
        bounds = {"lower": _r(lo), "upper": _r(hi), "k": k}
    elif method == "mad":
        med = s.median()
        mad = (s - med).abs().median() * 1.4826 or 1
        score = (vals - med) / mad
        mask = score.abs() > threshold
        bounds = {"median": _r(med), "mad": _r(mad), "threshold": threshold}
    else:
        raise ValueError("method must be zscore, iqr or mad")
    idx = df.index[mask.fillna(False)]
    items = []
    for i in idx[:max_items]:
        row = {"value": _r(vals.loc[i]), "score": _r(score.loc[i], 3)}
        if label_col and label_col in df.columns:
            row["label"] = str(df.loc[i, label_col])
        else:
            row["row"] = {k: (str(v) if not isinstance(v, (int, float)) else _r(v)) for k, v in df.loc[i].items()}
        items.append(row)
    return {"method": method, "value_column": value_col, "n": int(len(s)), "anomaly_count": int(mask.sum()),
            "anomaly_rate_pct": _r(100.0 * mask.sum() / len(s), 2), "bounds": bounds, "anomalies": items}


def trend(df: pd.DataFrame, time_col: str, value_col: str, freq: str | None = None, agg: str = "sum") -> dict[str, Any]:
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not in query result")
    t = pd.to_datetime(df[time_col], errors="coerce")
    v = _num(df, value_col)
    data = pd.DataFrame({"t": t, "v": v}).dropna()
    if freq:
        rule = {"day": "D", "week": "W", "month": "MS", "quarter": "QS", "year": "YS"}.get(freq)
        if not rule:
            raise ValueError("freq must be day, week, month, quarter or year")
        if agg not in {"sum", "mean", "count", "median"}:
            raise ValueError("agg must be sum, mean, count or median")
        data = getattr(data.set_index("t")["v"].resample(rule), agg)().reset_index()
    else:
        data = data.groupby("t", as_index=False)["v"].sum()
    data = data.sort_values("t")
    if len(data) < 3:
        raise ValueError("Need at least 3 time points for trend analysis")
    x = np.arange(len(data))
    lr = stats.linregress(x, data["v"].to_numpy())
    tau, p_mk = stats.kendalltau(x, data["v"].to_numpy())
    first, last = float(data["v"].iloc[0]), float(data["v"].iloc[-1])
    prev = float(data["v"].iloc[-2])
    # simple change-point: split maximising difference in means
    best = None
    arr = data["v"].to_numpy()
    for k in range(2, len(arr) - 1):
        diff = arr[k:].mean() - arr[:k].mean()
        if best is None or abs(diff) > abs(best[1]):
            best = (k, diff)
    series = [{"period": ts.date().isoformat(), "value": _r(val, 2)} for ts, val in zip(data["t"], data["v"])]
    direction = "increasing" if lr.slope > 0 else "decreasing"
    return {
        "time_column": time_col, "value_column": value_col, "freq": freq, "agg": agg, "points": len(data),
        "slope_per_period": _r(lr.slope), "r_squared": _r(lr.rvalue ** 2), "p_value": _r(lr.pvalue, 6),
        "mann_kendall_tau": _r(tau), "mann_kendall_p": _r(p_mk, 6),
        "direction": direction, "significant": bool(lr.pvalue < 0.05),
        "first_value": _r(first, 2), "last_value": _r(last, 2),
        "total_change_pct": _r(100 * (last - first) / first, 2) if first else None,
        "last_period_change_pct": _r(100 * (last - prev) / prev, 2) if prev else None,
        "change_point": {"period": series[best[0]]["period"], "mean_shift": _r(best[1], 2)} if best else None,
        "series": series[-60:],
    }


def contribution(df: pd.DataFrame, dims: list[str], value_col: str = "value", period_col: str = "period") -> dict[str, Any]:
    """df has columns dims + period ('previous'|'current') + value. Returns change decomposition."""
    piv = df.pivot_table(index=dims, columns=period_col, values=value_col, aggfunc="sum", fill_value=0.0)
    for c in ("previous", "current"):
        if c not in piv.columns:
            piv[c] = 0.0
    piv["delta"] = piv["current"] - piv["previous"]
    total_prev, total_cur = float(piv["previous"].sum()), float(piv["current"].sum())
    total_delta = total_cur - total_prev
    piv["pct_change"] = np.where(piv["previous"] != 0, 100 * piv["delta"] / piv["previous"].replace(0, np.nan), np.nan)
    piv["share_of_total_change_pct"] = 100 * piv["delta"] / total_delta if total_delta else np.nan
    piv = piv.sort_values("delta")
    rows = []
    for key, r in piv.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        rows.append({
            **{d: str(k) for d, k in zip(dims, key)},
            "previous": _r(r["previous"], 2), "current": _r(r["current"], 2), "delta": _r(r["delta"], 2),
            "pct_change": _r(r["pct_change"], 2), "share_of_total_change_pct": _r(r["share_of_total_change_pct"], 2),
        })
    return {
        "dimensions": dims,
        "total_previous": _r(total_prev, 2), "total_current": _r(total_cur, 2), "total_delta": _r(total_delta, 2),
        "total_pct_change": _r(100 * total_delta / total_prev, 2) if total_prev else None,
        "segments": rows[:50],
        "top_negative_contributors": [x for x in rows if (x["delta"] or 0) < 0][:5],
        "top_positive_contributors": [x for x in reversed(rows) if (x["delta"] or 0) > 0][:5],
    }
