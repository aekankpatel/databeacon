"""Report MCP server: interactive Plotly charts from SQL results and PDF reports via ReportLab."""

from __future__ import annotations

import json
import re
from typing import Any

from mcp.server.mcpserver import MCPServer

from ..config import get_settings
from ..datasets import DatasetRegistry
from ..reporting.charts import CHART_TYPES, build_chart
from ..reporting.pdf import build_pdf
from ..sql_guard import query_df
from .common import default_registry, expected_errors, in_thread

INSTRUCTIONS = "DataBeacon report server: create_chart renders evidence-backed charts; generate_pdf_report writes the final report."


def build_report_server(registry: DatasetRegistry | None = None) -> MCPServer:
    reg = registry or default_registry()
    settings = get_settings()
    mcp = MCPServer("databeacon-report", instructions=INSTRUCTIONS, version="0.1.0")

    @mcp.tool(
        description=(
            "Create an interactive chart from a read-only SQL query. chart_type: "
            f"{', '.join(sorted(CHART_TYPES))}. x is the category/time column, y one or more value columns, "
            "color an optional series column. orientation 'h' for horizontal bars. Use aggregated queries "
            "(<= 500 rows). Chart data is kept as evidence."
        )
    )
    @expected_errors
    async def create_chart(dataset_id: str, sql: str, chart_type: str, x: str, y: list[str] | str | None = None,
                           title: str = "", color: str | None = None, orientation: str = "v") -> dict[str, Any]:
        def _do():
            df = query_df(reg.db_path(dataset_id), sql, max_rows=500, timeout_s=settings.sql_timeout_s)
            chart = build_chart(df, chart_type, x, y, color, title, orientation)
            chart["sql"] = sql
            return chart

        return await in_thread(_do)

    @mcp.tool(
        description=(
            "Render the final PDF report. Arguments: run_id, title, question, dataset, summary, findings "
            "(list of {title, statement, verification, evidence_ids}), charts (chart objects from create_chart), "
            "evidence, limitations, timeline, metrics. Returns the file path."
        )
    )
    @expected_errors
    async def generate_pdf_report(run_id: str, title: str, question: str, dataset: str, summary: str,
                                  findings: list[dict[str, Any]] | None = None,
                                  charts: list[dict[str, Any]] | None = None,
                                  evidence: list[dict[str, Any]] | None = None,
                                  limitations: list[str] | None = None,
                                  timeline: list[dict[str, Any]] | None = None,
                                  metrics: dict[str, Any] | None = None,
                                  scope: dict[str, Any] | None = None) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-zA-Z_-]{1,64}", run_id):
            raise ValueError("invalid run_id")
        path = settings.reports_dir / f"{run_id}.pdf"
        payload = {"title": title, "question": question, "dataset": dataset, "summary": summary,
                   "findings": findings or [], "charts": charts or [], "evidence": evidence or [],
                   "limitations": limitations or [], "timeline": timeline or [], "metrics": metrics or {},
                   "scope": scope or {}}
        return await in_thread(build_pdf, path, payload)

    @mcp.resource("reports://list", description="Generated PDF reports", mime_type="application/json")
    def list_reports() -> str:
        return json.dumps(sorted(p.name for p in settings.reports_dir.glob("*.pdf")))

    @mcp.prompt(description="Structure of a DataBeacon report")
    def report_outline() -> str:
        return ("1. Executive summary answering the question directly. 2. Findings, each with numbers traceable to "
                "evidence. 3. Visualizations. 4. Limitations (data quality, causality, sample size). 5. Evidence appendix.")

    return mcp


if __name__ == "__main__":
    build_report_server().run("stdio")
