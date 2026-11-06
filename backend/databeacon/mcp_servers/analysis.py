"""Analysis MCP server: statistics, hypothesis tests, anomalies, trends and change attribution."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from .. import analysis as A
from ..config import get_settings
from ..datasets import DatasetRegistry
from ..sql_guard import query_df
from .common import check_date, default_registry, expected_errors, in_thread, quote_ident

INSTRUCTIONS = (
    "DataBeacon analysis server. Most tools take a read-only SQL query whose result is analysed with "
    "pandas/SciPy. period_comparison attributes a change between two periods to dimension values."
)


def build_analysis_server(registry: DatasetRegistry | None = None) -> MCPServer:
    reg = registry or default_registry()
    settings = get_settings()
    mcp = MCPServer("databeacon-analysis", instructions=INSTRUCTIONS, version="0.1.0")

    def _df(dataset_id: str, sql: str):
        return query_df(reg.db_path(dataset_id), sql, timeout_s=settings.sql_timeout_s * 2)

    @mcp.tool(description="Descriptive statistics (count, mean, std, quartiles, sum, skew) for numeric columns of a SQL result.")
    @expected_errors
    async def descriptive_statistics(dataset_id: str, sql: str) -> dict[str, Any]:
        return await in_thread(lambda: A.describe(_df(dataset_id, sql)))

    @mcp.tool(description="Pairwise correlation (pearson|spearman|kendall) with p-values between numeric columns of a SQL result.")
    @expected_errors
    async def correlation_analysis(dataset_id: str, sql: str, method: str = "pearson") -> dict[str, Any]:
        return await in_thread(lambda: A.correlation(_df(dataset_id, sql), method))

    @mcp.tool(
        description=(
            "Hypothesis test of whether value_column differs across groups of group_column in a SQL result "
            "(row-level data). 2 groups: Welch t-test + Mann-Whitney + Cohen's d; 3+: ANOVA + Kruskal-Wallis + eta²."
        )
    )
    @expected_errors
    async def compare_groups(dataset_id: str, sql: str, group_column: str, value_column: str) -> dict[str, Any]:
        return await in_thread(lambda: A.compare_groups(_df(dataset_id, sql), group_column, value_column))

    @mcp.tool(description="Chi-square test of independence between two categorical columns of a SQL result (with Cramér's V).")
    @expected_errors
    async def chi_square_test(dataset_id: str, sql: str, column_a: str, column_b: str) -> dict[str, Any]:
        return await in_thread(lambda: A.chi_square(_df(dataset_id, sql), column_a, column_b))

    @mcp.tool(
        description=(
            "Detect anomalies in value_column of a SQL result using zscore, iqr or mad. Use aggregated series "
            "(e.g. daily revenue) to find unusual periods, or row-level data for outlier records."
        )
    )
    @expected_errors
    async def detect_anomalies(dataset_id: str, sql: str, value_column: str, method: str = "zscore",
                               threshold: float = 3.0, label_column: str | None = None) -> dict[str, Any]:
        return await in_thread(lambda: A.detect_anomalies(_df(dataset_id, sql), value_column, method, threshold, label_column))

    @mcp.tool(
        description=(
            "Trend analysis of value_column over time_column from a SQL result: optional resampling "
            "(freq=day|week|month|quarter|year, agg=sum|mean|count|median), linear slope, R², p-value, "
            "Mann-Kendall tau, % change and the largest mean-shift change point."
        )
    )
    @expected_errors
    async def trend_analysis(dataset_id: str, sql: str, time_column: str, value_column: str,
                             freq: str | None = None, agg: str = "sum") -> dict[str, Any]:
        return await in_thread(lambda: A.trend(_df(dataset_id, sql), time_column, value_column, freq, agg))

    @mcp.tool(
        description=(
            "Attribute the change in SUM(value_column) between two periods to the values of the grouping columns. "
            "dimensions is REQUIRED: 1-3 column names, e.g. [\"customer_segment\"]. Dates ISO, ends exclusive. "
            "Returns totals, % change and each group's delta and share of the change."
        )
    )
    @expected_errors
    async def period_comparison(dataset_id: str, date_column: str, value_column: str,
                                dimensions: Annotated[list[str], Field(
                                    min_length=1, max_length=3,
                                    description="Grouping columns to attribute the change to (1-3)")],
                                previous_start: str, previous_end: str, current_start: str, current_end: str,
                                agg: str = "sum", filter_sql: str | None = None) -> dict[str, Any]:
        ds = reg.get(dataset_id)
        cols = {c["name"] for c in ds.columns}
        for c in [date_column, value_column, *dimensions]:
            if c not in cols:
                raise ValueError(f"Unknown column '{c}'. Available: {', '.join(sorted(cols))}")
        if not dimensions:
            raise ValueError("dimensions must name at least one grouping column, e.g. [\"customer_segment\"]")
        ps, pe = check_date(previous_start, "previous_start"), check_date(previous_end, "previous_end")
        cs, ce = check_date(current_start, "current_start"), check_date(current_end, "current_end")
        aggs = {"sum": f"SUM({quote_ident(value_column)})", "count": "COUNT(*)", "avg": f"AVG({quote_ident(value_column)})"}
        if agg not in aggs:
            raise ValueError("agg must be sum, count or avg")
        d = quote_ident(date_column)
        dims_sql = ", ".join(quote_ident(x) + "::VARCHAR AS " + quote_ident(x) for x in dimensions)
        where = f" AND ({filter_sql})" if filter_sql else ""
        sql = (
            f"SELECT {dims_sql}, CASE WHEN {d} >= '{cs}' AND {d} < '{ce}' THEN 'current' ELSE 'previous' END AS period, "
            f"{aggs[agg]} AS value FROM \"{ds.table}\" "
            f"WHERE (({d} >= '{ps}' AND {d} < '{pe}') OR ({d} >= '{cs}' AND {d} < '{ce}')){where} "
            f"GROUP BY ALL"
        )
        result = await in_thread(lambda: A.contribution(_df(dataset_id, sql), list(dimensions)))
        result.update({"metric": f"{agg}({value_column})", "previous_period": [ps, pe], "current_period": [cs, ce],
                       "sql": sql})
        return result

    @mcp.prompt(description="Guide for choosing a statistical test")
    def choose_test() -> str:
        return (
            "Numeric vs categorical (2 groups): compare_groups (t-test). 3+ groups: compare_groups (ANOVA). "
            "Two categoricals: chi_square_test. Two numerics: correlation_analysis. Over time: trend_analysis. "
            "Why did X change between periods: period_comparison with candidate dimensions. "
            "Unusual values: detect_anomalies."
        )

    return mcp


if __name__ == "__main__":
    build_analysis_server().run("stdio")
