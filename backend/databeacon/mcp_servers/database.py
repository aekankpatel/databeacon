"""Database MCP server: schema discovery, profiling and sandboxed read-only SQL on DuckDB."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver import MCPServer

from ..config import get_settings
from ..datasets import DatasetRegistry
from ..profiling import profile_table, schema_summary
from ..sql_guard import execute_readonly
from .common import default_registry, expected_errors, in_thread, quote_ident

INSTRUCTIONS = (
    "DataBeacon database server. Every tool takes a dataset_id. Use profile_dataset first to understand the "
    "data, then run_sql for read-only DuckDB SELECT queries. Writes, file and network access are blocked."
)


def build_database_server(registry: DatasetRegistry | None = None) -> MCPServer:
    reg = registry or default_registry()
    settings = get_settings()
    mcp = MCPServer("databeacon-database", instructions=INSTRUCTIONS, version="0.1.0")

    def _table_columns(dataset_id: str) -> tuple[str, list[str]]:
        ds = reg.get(dataset_id)
        return ds.table, [c["name"] for c in ds.columns]

    @mcp.tool(description="List the tables in the dataset and their columns with DuckDB types.")
    @expected_errors
    async def list_tables(dataset_id: str) -> dict[str, Any]:
        ds = reg.get(dataset_id)
        return {"tables": [{"name": ds.table, "row_count": ds.row_count, "columns": ds.columns}]}

    @mcp.tool(
        description=(
            "Profile the dataset: column types and semantic roles (numeric/categorical/temporal/identifier), "
            "missing values, distinct counts, summary statistics, IQR outliers, top values, strong correlations "
            "and data-quality warnings. Call this first."
        )
    )
    @expected_errors
    async def profile_dataset(dataset_id: str) -> dict[str, Any]:
        ds = reg.get(dataset_id)
        prof = await in_thread(profile_table, reg.db_path(dataset_id), ds.table)
        prof["summary"] = schema_summary(prof)
        return prof

    @mcp.tool(
        description=(
            "Execute ONE read-only DuckDB SQL SELECT (CTEs allowed) and return columns and rows. "
            "Use double quotes for identifiers, ISO dates ('2025-10-01'), date_trunc('quarter', col), "
            "year(col), quarter(col). Aggregate in SQL rather than fetching raw rows. "
            "'purpose' briefly states what the query tests."
        )
    )
    @expected_errors
    async def run_sql(dataset_id: str, sql: str, purpose: str = "") -> dict[str, Any]:
        res = await in_thread(
            execute_readonly, reg.db_path(dataset_id), sql,
            max_rows=settings.sql_max_rows, timeout_s=settings.sql_timeout_s,
        )
        out = res.to_dict()
        out["purpose"] = purpose
        return out

    @mcp.tool(description="Return up to n (max 50) example rows from the dataset table.")
    @expected_errors
    async def sample_rows(dataset_id: str, n: int = 10) -> dict[str, Any]:
        table, _ = _table_columns(dataset_id)
        n = max(1, min(int(n), 50))
        res = await in_thread(execute_readonly, reg.db_path(dataset_id), f'SELECT * FROM "{table}" USING SAMPLE {n} ROWS')
        return res.to_dict()

    @mcp.tool(description="Distinct values of a column with their row counts (most frequent first).")
    @expected_errors
    async def column_values(dataset_id: str, column: str, limit: int = 25) -> dict[str, Any]:
        table, cols = _table_columns(dataset_id)
        if column not in cols:
            raise ValueError(f"Unknown column '{column}'. Available: {', '.join(cols)}")
        limit = max(1, min(int(limit), 200))
        q = quote_ident(column)
        sql = f'SELECT {q} AS value, COUNT(*) AS n FROM "{table}" GROUP BY 1 ORDER BY 2 DESC LIMIT {limit}'
        return (await in_thread(execute_readonly, reg.db_path(dataset_id), sql)).to_dict()

    # ------------------------------------------------------------- resources
    @mcp.resource("datasets://catalog", description="Catalog of uploaded datasets", mime_type="application/json")
    def catalog() -> str:
        return json.dumps([d.to_dict() for d in reg.list()], indent=2)

    @mcp.resource("dataset://{dataset_id}/schema", description="Schema of one dataset", mime_type="application/json")
    def dataset_schema(dataset_id: str) -> str:
        return json.dumps(reg.get(dataset_id).to_dict(), indent=2)

    # --------------------------------------------------------------- prompts
    @mcp.prompt(description="DuckDB SQL conventions used by the analyst agent")
    def sql_style_guide() -> str:
        return (
            "Write a single DuckDB SELECT. Quote identifiers with double quotes. Prefer aggregates and GROUP BY. "
            "Compare periods with FILTER (WHERE ...) clauses, e.g. SUM(revenue) FILTER (WHERE order_date >= "
            "'2025-10-01'). Use date_trunc('month', col) for time series. Round outputs to 2 decimals. "
            "Never SELECT * without LIMIT."
        )

    @mcp.prompt(description="Turn a business question into testable hypotheses")
    def hypothesis_prompt(question: str) -> str:
        return (
            f"Question: {question}\nList 3-5 falsifiable hypotheses, the metric each one tests, the dimensions to "
            "slice by and the SQL/statistical evidence that would confirm or reject it."
        )

    return mcp


if __name__ == "__main__":  # stdio entrypoint: python -m databeacon.mcp_servers.database
    build_database_server().run("stdio")
