from pathlib import Path

import pytest

from databeacon.agent.testing import scripted_ecommerce_llm
from databeacon.runs import RunManager

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_full_agent_run(registry, dataset):
    manager = RunManager(registry, llm_factory=scripted_ecommerce_llm)
    run = await manager.run_sync(dataset.id, "Why did revenue decline last quarter and which segments contributed?")
    assert run["status"] == "completed"
    res = run["result"]
    names = [e["name"] for e in res["timeline"]]
    types = [e["type"] for e in res["timeline"]]
    # dataset understanding happens first, followed by tool discovery
    assert names[:3] == ["understand_dataset", "profile_dataset", "mcp_tool_discovery"]
    # dynamic replanning happened once, then the agent stopped on sufficient evidence
    assert "replan" in names and res["stop_reason"] == "evidence sufficient" and res["iterations"] == 2
    # error recovery: the injected bad SQL failed and the run still completed
    assert any(e["type"] == "tool" and e["status"] == "error" for e in res["timeline"])
    # the fabricated claim was caught and revised away
    assert "revise_answer" in names
    # scope was resolved from data coverage, and the answer matches the task contract
    assert "time_scope_resolved" in names and res["spec"]["current"]["label"] == "Q4 2025"
    assert res["spec"]["metric_column"] == "revenue" and res["spec"]["group_by"] == ["customer_segment"]
    assert res["alignment"]["status"] == "aligned", res["alignment"]
    assert res["synthesis"]["answer"]["primary_entity"] == "Enterprise"
    assert res["verification"]["hallucination_rate"] == 0
    assert all(f["verification"]["status"] == "verified" for f in res["synthesis"]["findings"])
    summary = res["synthesis"]["summary"]
    assert "Enterprise" in summary  # largest decline by segment (data-derived truth)
    assert "SMB" in summary and "West" in summary  # sharpest segment x region cell
    assert res["charts"] and res["charts"][0]["figure"]
    assert "llm" in types and res["metrics"]["total_tokens"] > 0
    pdf = Path(res["report"]["path"])
    assert pdf.exists() and pdf.stat().st_size > 2000
    from pypdf import PdfReader

    text = "".join(p.extract_text() for p in PdfReader(pdf).pages)
    assert "Executive summary" in text and "VERIFIED" in text and "Agent execution timeline" in text
    # persisted to SQLite
    assert manager.store.get(run["id"])["status"] == "completed"


async def test_budget_stops_agent(registry, dataset, monkeypatch):
    from dataclasses import replace

    from databeacon.config import get_settings

    settings = replace(get_settings(), max_tool_calls=2, max_iterations=1)
    manager = RunManager(registry, settings, llm_factory=scripted_ecommerce_llm)
    run = await manager.run_sync(dataset.id, "Why did revenue decline last quarter?")
    res = run["result"]
    # two tool calls cannot attribute the decline: the run must say so rather than claim success
    assert run["status"] == "completed_with_issues"
    assert res["alignment"]["status"] == "misaligned"
    assert res["metrics"]["tool_calls"] <= 4  # profile + 2 agent calls + optional auto chart
    assert any(e["name"] == "budget_reached" for e in res["timeline"])


async def test_plan_failure_falls_back(registry, dataset):
    from langchain_core.messages import AIMessage

    from databeacon.agent.testing import EcommerceAnalystPolicy, ScriptedChatModel

    policy = EcommerceAnalystPolicy(hallucinate_first=False)

    def responder(messages, tools_bound):
        task = str(messages[-2].content) if len(messages) >= 2 else ""
        if not tools_bound and "Plan the investigation" in task:
            return AIMessage('{"id": "H1", "statement": "not a plan"}')  # wrong shape every time
        return policy(messages, tools_bound)

    manager = RunManager(registry, llm_factory=lambda: ScriptedChatModel(responder=responder))
    run = await manager.run_sync(dataset.id, "Why did revenue decline last quarter?")
    assert run["status"] == "completed"
    names = [e["name"] for e in run["result"]["timeline"]]
    assert "plan_fallback" in names
    assert run["result"]["synthesis"]["findings"]
