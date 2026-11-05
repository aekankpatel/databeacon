"""Chart specs: build Plotly figures from query results."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pandas as pd
import plotly.express as px

CHART_TYPES = {"bar", "line", "area", "scatter", "pie", "histogram", "box"}
PALETTE = ["#2563eb", "#f97316", "#10b981", "#a855f7", "#ef4444", "#14b8a6", "#eab308", "#64748b"]


def build_chart(df: pd.DataFrame, chart_type: str, x: str, y: str | list[str] | None = None, color: str | None = None,
                title: str = "", orientation: str = "v") -> dict[str, Any]:
    if chart_type not in CHART_TYPES:
        raise ValueError(f"chart_type must be one of {sorted(CHART_TYPES)}")
    if df.empty:
        raise ValueError("The query returned no rows to chart")
    ys = [y] if isinstance(y, str) else list(y or [])
    for c in [x, *ys, *([color] if color else [])]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not in query result (columns: {', '.join(map(str, df.columns))})")
    df = df.head(500)
    common = {"title": title, "color_discrete_sequence": PALETTE}
    yarg: Any = ys[0] if len(ys) == 1 else (ys or None)
    if chart_type == "bar":
        if orientation == "h":
            fig = px.bar(df, x=yarg, y=x, color=color, orientation="h", barmode="group", **common)
        else:
            fig = px.bar(df, x=x, y=yarg, color=color, barmode="group", **common)
    elif chart_type == "line":
        fig = px.line(df, x=x, y=yarg, color=color, markers=True, **common)
    elif chart_type == "area":
        fig = px.area(df, x=x, y=yarg, color=color, **common)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x, y=yarg, color=color, **common)
    elif chart_type == "pie":
        if len(ys) != 1:
            raise ValueError("pie charts need exactly one y (values) column")
        fig = px.pie(df, names=x, values=ys[0], **common)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x, color=color, **common)
    else:
        fig = px.box(df, x=color or None, y=x, **common)
    fig.update_layout(template="plotly_white", margin={"l": 40, "r": 20, "t": 50, "b": 40},
                      font={"family": "Inter, system-ui, sans-serif"})
    records = json.loads(df.to_json(orient="split", date_format="iso"))
    return {
        "chart_id": uuid.uuid4().hex[:8],
        "title": title,
        "chart_type": chart_type,
        "x": x,
        "y": ys,
        "color": color,
        "orientation": orientation,
        "data": {"columns": records["columns"], "rows": records["data"]},
        "figure": json.loads(fig.to_json()),
    }
