import pytest

from databeacon.mcp_client import MCPToolHub

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_discovery_across_servers(registry):
    async with MCPToolHub(registry) as hub:
        servers = {t.server for t in hub.tools.values()}
        assert servers == {"database", "analysis", "report"}
        assert {"run_sql", "period_comparison", "create_chart", "generate_pdf_report"} <= set(hub.tools)
        assert any(r["uri"] == "datasets://catalog" for r in hub.resources)
        assert any(p["name"] == "sql_style_guide" for p in hub.prompts)
        # dataset_id is injected by the client and hidden from the LLM
        schema = next(t for t in hub.llm_tools() if t["function"]["name"] == "run_sql")
        assert "dataset_id" not in schema["function"]["parameters"]["properties"]
        assert "generate_pdf_report" not in {t["function"]["name"] for t in hub.llm_tools()}


async def test_calls_and_error_messages(registry, dataset):
    ctx = {"dataset_id": dataset.id}
    async with MCPToolHub(registry) as hub:
        ok = await hub.call("run_sql", {"sql": "SELECT COUNT(*) AS n FROM ecommerce_orders"}, ctx)
        assert ok.ok and ok.data["rows"][0][0] == dataset.row_count
        bad = await hub.call("run_sql", {"sql": "DELETE FROM ecommerce_orders"}, ctx)
        assert not bad.ok and "read-only" in bad.error
        unknown = await hub.call("no_such_tool", {}, ctx)
        assert not unknown.ok and "Unknown tool" in unknown.error
        chart = await hub.call("create_chart", {"sql": "SELECT region, SUM(revenue) AS revenue FROM ecommerce_orders GROUP BY 1",
                                                "chart_type": "bar", "x": "region", "y": "revenue",
                                                "title": "Revenue by region"}, ctx)
        assert chart.ok and chart.data["figure"]["data"]


async def test_stdio_transport(registry, dataset):
    async with MCPToolHub(registry, transport="stdio", servers=("database",)) as hub:
        res = await hub.call("list_tables", {}, {"dataset_id": dataset.id})
        assert res.ok and res.data["tables"][0]["name"] == "ecommerce_orders"
