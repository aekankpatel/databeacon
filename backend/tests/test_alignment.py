"""Regression tests: wrong-quarter answers, wrong primary conclusions, unsupported claims,
temporal resolution, clarification, scorer strictness and context reduction."""

import json
from datetime import date

import pandas as pd
import pytest

from databeacon.agent.alignment import TaskSpec, check_alignment
from databeacon.agent.temporal import Coverage, resolve
from databeacon.agent.testing import scripted_ecommerce_llm
from databeacon.runs import RunManager

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


Q4 = {"start": "2025-10-01", "end": "2026-01-01", "label": "Q4 2025"}
Q3 = {"start": "2025-07-01", "end": "2025-10-01", "label": "Q3 2025"}
SPEC = TaskSpec(question="Why did revenue decline last quarter by segment?", question_type="change_attribution",
                metric_column="revenue", aggregation="sum", group_by=["customer_segment"],
                answer_target="segment with the largest decline", date_column="order_date", current=Q4, previous=Q3)


def _pc(eid, cur, prev, rows):
    return {"id": eid, "tool": "period_comparison", "ok": True,
            "args": {"date_column": "order_date", "value_column": "revenue", "dimensions": ["customer_segment"],
                     "current_start": cur["start"], "current_end": cur["end"],
                     "previous_start": prev["start"], "previous_end": prev["end"]},
            "data": {"dimensions": ["customer_segment"], "segments": rows,
                     "current_period": [cur["start"], cur["end"]], "previous_period": [prev["start"], prev["end"]]},
            "sql": None, "preview": None}


ROWS = [{"customer_segment": "Enterprise", "delta": -87116.11}, {"customer_segment": "Consumer", "delta": -31803.0},
        {"customer_segment": "SMB", "delta": -27521.0}]


def _syn(entity, cur, prev, eid="E1", summary=None):
    return {"summary": summary or f"Revenue fell; {entity} contributed most.",
            "findings": [{"title": "t", "statement": "s", "evidence_ids": [eid]}],
            "answer": {"primary_entity": entity, "dimension": "customer_segment", "metric_column": "revenue",
                       "direction": "largest_decline", "current_period": [cur["start"], cur["end"]],
                       "previous_period": [prev["start"], prev["end"]], "evidence_ids": [eid]}}


# ----------------------------------------------------------------- alignment unit tests
def test_aligned_answer_passes():
    out = check_alignment(SPEC, _syn("Enterprise", Q4, Q3), [_pc("E1", Q4, Q3, ROWS)])
    assert out["status"] == "aligned", out["issues"]
    assert out["checks"]["primary_conclusion"] == "pass"


def test_wrong_quarter_is_misaligned_even_with_correct_numbers():
    q2 = {"start": "2025-04-01", "end": "2025-07-01", "label": "Q2 2025"}
    syn = _syn("SMB", Q3, q2, summary="Revenue fell 12.5% in Q3 2025 compared with Q2 2025; SMB contributed most.")
    out = check_alignment(SPEC, syn, [_pc("E1", Q3, q2, [{"customer_segment": "SMB", "delta": -69738.5}])])
    assert out["status"] == "misaligned"
    assert out["checks"]["period"] == "fail"
    text = " ".join(out["issues"])
    assert "Q4 2025" in text and "Q2 2025" in text


def test_wrong_primary_conclusion_contradicting_evidence():
    out = check_alignment(SPEC, _syn("Consumer", Q4, Q3, summary="Enterprise and Consumer fell; Consumer led."),
                          [_pc("E1", Q4, Q3, ROWS)])
    assert out["checks"]["primary_conclusion"] == "fail"
    assert "Enterprise" in " ".join(out["issues"])


def test_wrong_grouping_and_metric():
    syn = _syn("West", Q4, Q3)
    syn["answer"].update({"dimension": "region", "metric_column": "quantity"})
    out = check_alignment(SPEC, syn, [_pc("E1", Q4, Q3, ROWS)])
    assert out["checks"]["grouping"] == "fail" and out["checks"]["metric"] == "fail"


def test_missing_conclusion_is_flagged():
    syn = _syn(None, Q4, Q3)
    out = check_alignment(SPEC, syn, [_pc("E1", Q4, Q3, ROWS)])
    assert out["checks"]["primary_conclusion"] == "fail"


def test_conclusion_recomputed_from_sql_result():
    ev = {"id": "E2", "tool": "run_sql", "ok": True, "args": {"sql": "SELECT customer_segment, SUM(revenue) FILTER "
                                                                  "(WHERE order_date >= '2025-10-01') - ... AS delta"},
          "sql": "SELECT customer_segment, ... revenue ... '2025-10-01' '2025-07-01'",
          "data": {"columns": ["customer_segment", "delta"], "rows": [["Enterprise", -87116.1], ["SMB", -27521.0]]},
          "preview": None}
    assert check_alignment(SPEC, _syn("SMB", Q4, Q3, eid="E2"), [ev])["checks"]["primary_conclusion"] == "fail"
    assert check_alignment(SPEC, _syn("Enterprise", Q4, Q3, eid="E2"), [ev])["checks"]["primary_conclusion"] == "pass"


# ----------------------------------------------------------------- end-to-end agent behaviour
async def test_wrong_quarter_run_is_not_reported_as_success(registry, dataset):
    m = RunManager(registry, llm_factory=lambda: scripted_ecommerce_llm(wrong_quarter=True, stubborn=True))
    run = await m.run_sync(dataset.id, "Why did our revenue decline last quarter, and which segments contributed?")
    assert run["status"] == "completed_with_issues"
    res = run["result"]
    assert res["alignment"]["status"] == "misaligned" and res["alignment"]["checks"]["period"] == "fail"
    names = [e["name"] for e in res["timeline"]]
    assert "realign_to_question" in names  # it was sent back to collect the right evidence


async def test_wrong_quarter_self_corrects_after_realignment(registry, dataset):
    m = RunManager(registry, llm_factory=lambda: scripted_ecommerce_llm(wrong_quarter=True))
    run = await m.run_sync(dataset.id, "Why did our revenue decline last quarter, and which segments contributed?")
    res = run["result"]
    assert run["status"] == "completed", res["alignment"]
    assert res["synthesis"]["answer"]["current_period"] == ["2025-10-01", "2026-01-01"]
    assert res["synthesis"]["answer"]["primary_entity"] == "Enterprise"


async def test_wrong_conclusion_run_is_flagged(registry, dataset):
    m = RunManager(registry, llm_factory=lambda: scripted_ecommerce_llm(wrong_conclusion=True))
    run = await m.run_sync(dataset.id, "Why did our revenue decline last quarter, and which segments contributed?")
    assert run["status"] == "completed_with_issues"
    assert run["result"]["alignment"]["checks"]["primary_conclusion"] == "fail"


async def test_unsupported_claims_are_revised(registry, dataset):
    m = RunManager(registry, llm_factory=lambda: scripted_ecommerce_llm(hallucinate_first=True))
    run = await m.run_sync(dataset.id, "Why did our revenue decline last quarter, and which segments contributed?")
    res = run["result"]
    assert "revise_answer" in [e["name"] for e in res["timeline"]]
    assert res["verification"]["hallucination_rate"] == 0
    assert not any("123,457" in f["statement"] for f in res["synthesis"]["findings"])


async def test_ambiguous_period_asks_for_clarification(registry, sample_csv, tmp_path):
    df = pd.read_csv(sample_csv)
    partial = tmp_path / "orders_to_nov.csv"  # data ends 2025-11-20: Q4 is 56% covered
    df[df["order_date"] <= "2025-11-20"].to_csv(partial, index=False)
    ds = registry.ingest_file(partial, name="partial")
    m = RunManager(registry, llm_factory=scripted_ecommerce_llm)
    run = await m.run_sync(ds.id, "Why did revenue decline last quarter?")
    assert run["status"] == "needs_clarification"
    need = run["result"]["needs_clarification"]
    assert "Q3 2025" in need["options"] and need["question"]
    assert run["result"]["metrics"]["llm_calls"] == 0  # no tokens spent on an ambiguous question
    run2 = await m.run_sync(ds.id, "Why did revenue decline last quarter?", clarification="Q3 2025")
    assert run2["status"] == "completed"
    assert run2["result"]["spec"]["current"]["label"] == "Q3 2025"


# ----------------------------------------------------------------- temporal resolution
@pytest.mark.parametrize("max_d,spacing,status,current", [
    (date(2025, 12, 31), 1, "resolved", "Q4 2025"),            # latest quarter complete
    (date(2026, 1, 10), 1, "resolved", "Q4 2025"),             # barely started Q1: excluded, documented
    (date(2026, 3, 10), 1, "needs_clarification", None),       # mostly-covered partial quarter: ask
    (date(2025, 12, 29), 7, "resolved", "Q4 2025"),            # weekly data reaches quarter end
    (date(2025, 12, 1), 30, "resolved", "Q4 2025"),            # monthly data stored as month starts
])
def test_last_quarter_uses_coverage(max_d, spacing, status, current):
    r = resolve("Why did revenue decline last quarter?", ["d"],
                {"d": Coverage("d", date(2024, 1, 1), max_d, 500, spacing)})
    assert r.status == status
    assert (r.current.label if r.current else None) == current
    if status == "resolved":
        assert r.assumptions and r.previous is not None


def test_temporal_clarification_cases():
    cov = Coverage("d", date(2024, 1, 1), date(2025, 12, 31), 700, 1)
    assert resolve("What happened in Q2 2023?", ["d"], {"d": cov}).status == "needs_clarification"
    assert resolve("Revenue last quarter?", [], {}).status == "needs_clarification"
    two = resolve("Revenue last quarter?", ["order_date", "ship_date"], {"order_date": cov, "ship_date": cov})
    assert two.status == "needs_clarification" and two.options == ["order_date", "ship_date"]
    first = resolve("Why did revenue fall last quarter?", ["d"],
                    {"d": Coverage("d", date(2025, 10, 1), date(2025, 12, 31), 92, 1)})
    assert first.status == "needs_clarification"  # nothing to compare against
    assert resolve("Which region ships slowest?", ["d"], {"d": cov}).status == "not_temporal"
    pair = resolve("Compare Q3 2025 with Q2 2025", ["d"], {"d": cov})
    assert (pair.current.label, pair.previous.label) == ("Q3 2025", "Q2 2025")
    yoy = resolve("Revenue last month year over year", ["d"], {"d": cov})
    assert (yoy.current.label, yoy.previous.label) == ("December 2025", "December 2024")


# ----------------------------------------------------------------- scorer strictness
def test_scorer_requires_structured_conclusion_not_a_mention():
    from evals.metrics import score_case

    gold = {"entity": "Enterprise", "dimension": "customer_segment", "metric_column": "revenue",
            "current_period": ["2025-10-01", "2026-01-01"], "previous_period": ["2025-07-01", "2025-10-01"],
            "numbers": [-87116.11]}
    ev = [_pc("E1", Q4, Q3, ROWS)]
    base = {"status": "completed", "result": {"evidence": ev, "alignment": {"status": "aligned"},
                                              "verification": {"hallucination_rate": 0.0}, "metrics": {}}}
    good = json.loads(json.dumps(base))
    good["result"]["synthesis"] = _syn("Enterprise", Q4, Q3)
    assert score_case({"id": "x"}, good, gold)["task_success"]
    mention = json.loads(json.dumps(base))
    mention["result"]["synthesis"] = _syn("SMB", Q4, Q3, summary="SMB drove it; Enterprise also fell.")
    s = score_case({"id": "x"}, mention, gold)
    assert not s["task_success"] and s["checks"]["primary_conclusion"] == "fail"
    wrong_period = json.loads(json.dumps(base))
    wrong_period["result"]["synthesis"] = _syn("Enterprise", Q3, {"start": "2025-04-01", "end": "2025-07-01"})
    assert score_case({"id": "x"}, wrong_period, gold)["checks"]["period"] == "fail"


# ----------------------------------------------------------------- context reduction
async def test_llm_tool_schemas_are_compact_but_complete(registry):
    from databeacon.mcp_client import MCPToolHub

    async with MCPToolHub(registry) as hub:
        tools = {t["function"]["name"]: t["function"] for t in hub.llm_tools()}
    assert not {"profile_dataset", "list_tables", "generate_pdf_report"} & set(tools)
    blob = json.dumps(tools)
    assert '"title": "' not in blob.replace('"title": {"type"', "")  # metadata titles gone...
    assert "title" in tools["create_chart"]["parameters"]["properties"]  # ...but a parameter named title is kept
    assert '"type": "null"' not in blob
    assert tools["run_sql"]["parameters"]["required"] == ["sql"]
    assert set(tools["period_comparison"]["parameters"]["required"]) >= {"date_column", "value_column", "dimensions",
                                                                         "current_start", "current_end"}
    assert all(len(t["description"]) <= 241 for t in tools.values())


def test_duplicate_detection_normalises_sql():
    from databeacon.agent.graph import normalize_call

    a = normalize_call("run_sql", {"sql": "SELECT region, SUM(revenue)\n  FROM t GROUP BY 1;", "purpose": "a"})
    b = normalize_call("run_sql", {"sql": "select region, sum(revenue) from t group by 1", "purpose": "b"})
    assert a == b
    assert a != normalize_call("run_sql", {"sql": "select region, avg(revenue) from t group by 1"})


def test_history_compaction_keeps_ids_and_latest_results():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    from databeacon.agent.graph import AnalystAgent

    msgs = [SystemMessage("s"), HumanMessage("q")]
    for i in range(6):
        msgs.append(AIMessage("", tool_calls=[{"name": "run_sql", "args": {"sql": f"q{i}"}, "id": f"c{i}"}]))
        msgs.append(ToolMessage(json.dumps({"evidence_id": f"E{i}", "result": {"rows": [[j] * 20 for j in range(60)]}}),
                                tool_call_id=f"c{i}", name="run_sql"))
    out = AnalystAgent._compact_history(msgs, budget_chars=9000)
    tools = [m for m in out if isinstance(m, ToolMessage)]
    assert [json.loads(m.content)["evidence_id"] for m in tools] == [f"E{i}" for i in range(6)]
    assert tools[-1].content == msgs[-1].content and tools[-2].content == msgs[-3].content  # latest kept verbatim
    assert sum(len(m.content) for m in tools) < sum(len(m.content) for m in msgs if isinstance(m, ToolMessage)) / 2
    assert [m.tool_call_id for m in tools] == [f"c{i}" for i in range(6)]  # call/result pairing intact


def test_compact_rounding_still_verifies():
    from databeacon.agent.evidence import compact_json
    from databeacon.agent.verifier import verify_findings

    data = {"rows": [[1082155.9999999993, 0.22113456, 946659.2700000012]]}
    shown = json.loads(compact_json(data))
    assert shown["rows"][0] == [1082156.0, 0.2211, 946659.27]
    f = [{"title": "t", "statement": "Revenue fell from 1,082,156 to 946,659.27; discount 22.1%.", "evidence_ids": ["E1"]}]
    assert verify_findings(f, [{"id": "E1", "tool": "run_sql", "ok": True, "data": data}])["hallucination_rate"] == 0
