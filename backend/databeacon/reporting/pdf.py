"""PDF report generation with ReportLab (charts drawn natively from chart data)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .charts import PALETTE

PAL = [colors.HexColor(c) for c in PALETTE]
INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#64748b")
ACCENT = colors.HexColor("#2563eb")


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], textColor=INK, fontSize=22, leading=26, alignment=TA_LEFT),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], textColor=INK, fontSize=15, spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], textColor=INK, fontSize=12, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13, textColor=INK),
        "muted": ParagraphStyle("m", parent=ss["BodyText"], fontSize=8.5, leading=11, textColor=MUTED),
        "code": ParagraphStyle("c", parent=ss["Code"], fontSize=7.5, leading=9.5, backColor=colors.HexColor("#f1f5f9"),
                               borderPadding=4, textColor=INK),
    }


def _p(text: Any, style) -> Paragraph:
    return Paragraph(escape(str(text if text is not None else "")).replace("\n", "<br/>"), style)


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(v: Any) -> str:
    f = _num(v)
    if f is None or isinstance(v, bool):
        return str(v)
    if abs(f) >= 1000:
        return f"{f:,.0f}"
    return f"{f:,.2f}".rstrip("0").rstrip(".") if f != int(f) else f"{int(f):,}"


def chart_drawing(chart: dict[str, Any], width: float = 16 * cm, height: float = 7 * cm) -> Drawing | None:
    data = chart.get("data") or {}
    cols, rows = data.get("columns") or [], data.get("rows") or []
    if not cols or not rows:
        return None
    x, ys, color = chart.get("x"), chart.get("y") or [], chart.get("color")
    ctype = chart.get("chart_type", "bar")
    try:
        xi = cols.index(x)
        yis = [cols.index(y) for y in ys]
    except ValueError:
        return None
    d = Drawing(width, height)
    d.add(String(0, height - 12, str(chart.get("title") or ""), fontName="Helvetica-Bold", fontSize=10, fillColor=INK))
    rows = rows[:60]
    categories = [str(r[xi])[:18] for r in rows]

    series: list[tuple[str, list[float]]] = []
    if color and color in cols and len(yis) == 1:
        ci = cols.index(color)
        groups: dict[str, dict[str, float]] = {}
        cats: list[str] = []
        for r in rows:
            c = str(r[xi])[:18]
            if c not in cats:
                cats.append(c)
            groups.setdefault(str(r[ci]), {})[c] = _num(r[yis[0]]) or 0.0
        categories = cats
        series = [(g, [vals.get(c, 0.0) for c in cats]) for g, vals in list(groups.items())[:8]]
    else:
        series = [(cols[i], [_num(r[i]) or 0.0 for r in rows]) for i in yis[:6]]
    if not series and ctype not in {"histogram"}:
        return None

    plot_w, plot_h = width - 2.6 * cm, height - 2.2 * cm
    if ctype == "pie" and series:
        pie = Pie()
        pie.x, pie.y, pie.width, pie.height = 1 * cm, 0.3 * cm, plot_h, plot_h
        pie.data = [max(v, 0) for v in series[0][1][:10]]
        pie.labels = None
        for i in range(len(pie.data)):
            pie.slices[i].fillColor = PAL[i % len(PAL)]
            pie.slices[i].strokeColor = colors.white
        d.add(pie)
        lg = Legend()
        lg.x, lg.y = plot_h + 2 * cm, plot_h
        lg.fontSize = 7
        lg.colorNamePairs = [(PAL[i % len(PAL)], f"{categories[i]} ({_fmt(v)})") for i, v in enumerate(pie.data)]
        d.add(lg)
        return d
    if ctype in {"line", "area", "scatter"}:
        lp = LinePlot()
        lp.x, lp.y, lp.width, lp.height = 1.8 * cm, 1.0 * cm, plot_w - 1 * cm, plot_h - 0.6 * cm
        lp.data = [[(i, v) for i, v in enumerate(vals)] for _, vals in series]
        for i in range(len(series)):
            lp.lines[i].strokeColor = PAL[i % len(PAL)]
            lp.lines[i].strokeWidth = 1.6
        lp.xValueAxis.valueSteps = list(range(0, len(categories), max(1, len(categories) // 8)))
        lp.xValueAxis.labelTextFormat = lambda v: categories[int(v)] if 0 <= int(v) < len(categories) else ""
        lp.xValueAxis.labels.fontSize = 6
        lp.xValueAxis.labels.angle = 30
        lp.xValueAxis.labels.dy = -6
        lp.yValueAxis.labels.fontSize = 7
        lp.yValueAxis.labelTextFormat = _fmt
        d.add(lp)
    else:
        horizontal = chart.get("orientation") == "h"
        bc = HorizontalBarChart() if horizontal else VerticalBarChart()
        bc.x, bc.y, bc.width, bc.height = (3 * cm if horizontal else 1.8 * cm), 1.0 * cm, plot_w - 1.5 * cm, plot_h - 0.6 * cm
        bc.data = [vals for _, vals in series]
        bc.categoryAxis.categoryNames = categories
        bc.categoryAxis.labels.fontSize = 6
        if not horizontal:
            bc.categoryAxis.labels.angle = 30 if len(categories) > 6 else 0
            bc.categoryAxis.labels.dy = -6 if len(categories) > 6 else -2
        bc.valueAxis.labels.fontSize = 7
        bc.valueAxis.labelTextFormat = _fmt
        vmin = min(min(v) for _, v in series)
        if vmin >= 0:
            bc.valueAxis.valueMin = 0
        for i in range(len(series)):
            bc.bars[i].fillColor = PAL[i % len(PAL)]
            bc.bars[i].strokeColor = None
        d.add(bc)
    if len(series) > 1:
        lg = Legend()
        lg.x, lg.y = width - 2.4 * cm, height - 0.8 * cm
        lg.fontSize = 6.5
        lg.colorNamePairs = [(PAL[i % len(PAL)], name[:16]) for i, (name, _) in enumerate(series)]
        d.add(lg)
    return d


def _table(rows: list[list[Any]], header: list[str], col_widths=None, font=7.5) -> Table:
    data = [header] + [[_fmt(c) if not isinstance(c, Paragraph) else c for c in r] for r in rows]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def build_pdf(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    """report keys: title, question, dataset, summary, findings, charts, evidence, limitations, timeline, metrics."""
    st = _styles()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm, title=report.get("title", "DataBeacon report"),
                            author="DataBeacon")
    story: list[Any] = []
    story.append(_p(report.get("title") or "DataBeacon Analysis Report", st["title"]))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(_p(f"Dataset: {report.get('dataset', '-')}   ·   Generated: {generated}", st["muted"]))
    story.append(Spacer(1, 8))
    story.append(_p("Question", st["h2"]))
    story.append(_p(report.get("question", ""), st["body"]))
    story.append(_p("Executive summary", st["h1"]))
    story.append(_p(report.get("summary", ""), st["body"]))

    metrics = report.get("metrics") or {}
    if metrics:
        mrows = [[k.replace("_", " "), v] for k, v in metrics.items()]
        story.append(Spacer(1, 6))
        story.append(_table(mrows, ["Run metric", "Value"], col_widths=[6 * cm, 5 * cm]))

    findings = report.get("findings") or []
    if findings:
        story.append(_p("Findings", st["h1"]))
        for i, f in enumerate(findings, 1):
            status = f.get("verification", {}).get("status", "unverified")
            badge = {"verified": "VERIFIED", "partially_verified": "PARTIALLY VERIFIED"}.get(status, "UNVERIFIED")
            block = [
                _p(f"{i}. {f.get('title', '')}  [{badge}]", st["h2"]),
                _p(f.get("statement", ""), st["body"]),
            ]
            ver = f.get("verification") or {}
            if ver.get("checked_numbers"):
                block.append(_p(f"Numbers checked: {ver.get('supported', 0)}/{ver.get('checked_numbers', 0)} matched "
                                f"tool outputs. Evidence: {', '.join(f.get('evidence_ids') or []) or '-'}", st["muted"]))
            story.append(KeepTogether(block))

    charts = report.get("charts") or []
    if charts:
        story.append(_p("Visualizations", st["h1"]))
        for ch in charts[:8]:
            dr = chart_drawing(ch)
            if dr is not None:
                story.append(KeepTogether([dr, Spacer(1, 10)]))

    limitations = report.get("limitations") or []
    if limitations:
        story.append(_p("Limitations & caveats", st["h1"]))
        for lim in limitations:
            story.append(_p(f"• {lim}", st["body"]))

    evidence = report.get("evidence") or []
    if evidence:
        story.append(PageBreak())
        story.append(_p("Supporting evidence", st["h1"]))
        for ev in evidence[:25]:
            block = [_p(f"[{ev.get('id')}] {ev.get('tool')} - {ev.get('purpose') or ''}", st["h2"])]
            if ev.get("sql"):
                block.append(_p(ev["sql"][:1500], st["code"]))
            preview = ev.get("preview") or {}
            if preview.get("columns") and preview.get("rows"):
                cols = [str(c)[:18] for c in preview["columns"][:7]]
                rows = [r[:7] for r in preview["rows"][:10]]
                block.append(Spacer(1, 3))
                block.append(_table(rows, cols))
            elif ev.get("summary"):
                block.append(_p(ev["summary"][:800], st["muted"]))
            block.append(Spacer(1, 6))
            story.append(KeepTogether(block))

    timeline = report.get("timeline") or []
    if timeline:
        story.append(_p("Agent execution timeline", st["h1"]))
        trows = [[str(i + 1), e.get("type", ""), _p(str(e.get("name", ""))[:80], st["muted"]),
                  f"{e.get('duration_ms') or 0:.0f} ms", e.get("status", "")] for i, e in enumerate(timeline[:80])]
        story.append(_table(trows, ["#", "Type", "Step", "Duration", "Status"],
                            col_widths=[0.8 * cm, 2 * cm, 9.5 * cm, 2 * cm, 1.6 * cm]))

    def _footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(1.8 * cm, 1 * cm, "DataBeacon - autonomous data analysis report")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1 * cm, f"Page {_doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return {"path": str(path), "bytes": path.stat().st_size}
