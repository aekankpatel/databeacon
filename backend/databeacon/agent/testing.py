"""Deterministic chat models for tests and offline evaluation (no API key needed)."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, PrivateAttr

Responder = Callable[[list[BaseMessage], bool], AIMessage]


class ScriptedChatModel(BaseChatModel):
    """Chat model whose replies come from a python callable ``responder(messages, tools_bound)``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    responder: Any
    tools_bound: bool = False
    model_name: str = "scripted"
    _calls: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: D401
        return self.model_copy(update={"tools_bound": True})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.responder(messages, self.tools_bound)
        chars = sum(len(str(m.content)) for m in messages)
        msg.usage_metadata = {"input_tokens": chars // 4, "output_tokens": len(str(msg.content)) // 4 + 20,
                              "total_tokens": chars // 4 + len(str(msg.content)) // 4 + 20}
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _call(name: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}",
                                              "type": "tool_call"}])


def _last_human(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def _evidence_blocks(text: str) -> dict[str, tuple[str, Any]]:
    out = {}
    for m in re.finditer(r"\[(E\d+)\] (\w+)[^\n]*\n(\{.*?\})(?=\n\n|\Z)", text, re.DOTALL):
        try:
            out[m.group(1)] = (m.group(2), json.loads(m.group(3)))
        except json.JSONDecodeError:
            continue
    return out


class EcommerceAnalystPolicy:
    """Scripted trajectory for the sample e-commerce dataset (exercises every graph branch)."""

    def __init__(self, hallucinate_first: bool = True, inject_tool_error: bool = True):
        self.hallucinate_first = hallucinate_first
        self.inject_tool_error = inject_tool_error

    def __call__(self, messages: list[BaseMessage], tools_bound: bool) -> AIMessage:
        if tools_bound:
            return self._act(messages)
        prompt = _last_human(messages)
        if "JSON schema" in prompt:
            # the task prompt precedes the schema instruction
            task = str(messages[-2].content) if len(messages) >= 2 else ""
        else:
            task = prompt
        if "Create an investigation plan" in task:
            return AIMessage(json.dumps({
                "objective": "Explain the Q4 2025 revenue decline vs Q3 2025 and attribute it to segments",
                "key_metric": "revenue", "time_scope": "previous 2025-07-01..2025-10-01, current 2025-10-01..2026-01-01",
                "hypotheses": [
                    {"id": "H1", "statement": "Revenue fell in Q4 2025 vs Q3 2025", "approach": "run_sql by quarter"},
                    {"id": "H2", "statement": "A specific segment/region drives most of the decline",
                     "approach": "period_comparison"},
                    {"id": "H3", "statement": "Discounting increased in some category", "approach": "compare_groups"},
                ],
                "steps": ["quarterly revenue", "period comparison by segment and region", "discount test", "chart"],
            }))
        if "Evaluate critically" in task:
            done = "compare_groups" in task
            return AIMessage(json.dumps({
                "sufficient": done, "confidence": 0.8 if done else 0.5,
                "hypothesis_status": {"H1": "supported", "H2": "supported", "H3": "supported" if done else "untested"},
                "gaps": [] if done else ["Discount changes by product category not tested"],
                "next_actions": [] if done else ["compare_groups discount_pct by product_category in Q4 2025"],
            }))
        if "Write the final analysis" in task:
            return AIMessage(json.dumps(self._synthesis(task)))
        return AIMessage("{}")

    def _act(self, messages: list[BaseMessage]) -> AIMessage:
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        n = len(tool_msgs)
        replanned = any(isinstance(m, HumanMessage) and "Reflection on the evidence" in str(m.content) for m in messages)
        script = [
            lambda: _call("run_sql", sql=(
                "SELECT quarter(order_date) AS quarter, ROUND(SUM(revenue), 2) AS revenue, COUNT(*) AS orders "
                "FROM ecommerce_orders WHERE year(order_date) = 2025 GROUP BY 1 ORDER BY 1"), purpose="Quarterly revenue 2025"),
            lambda: _call("period_comparison", date_column="order_date", value_column="revenue",
                          dimensions=["customer_segment", "region"], previous_start="2025-07-01",
                          previous_end="2025-10-01", current_start="2025-10-01", current_end="2026-01-01"),
        ]
        if self.inject_tool_error:
            script.insert(1, lambda: _call("run_sql", sql="SELECT SUM(revenu) FROM ecommerce_orders",
                                           purpose="typo to exercise error recovery"))
        script.append(lambda: _call("create_chart", sql=(
            "SELECT date_trunc('month', order_date) AS month, SUM(revenue) AS revenue FROM ecommerce_orders "
            "GROUP BY 1 ORDER BY 1"), chart_type="line", x="month", y=["revenue"], title="Monthly revenue"))
        if not replanned:
            if n < len(script):
                return script[n]()
            return AIMessage("Initial evidence collected: revenue fell and SMB/West is the largest contributor.")
        follow = [lambda: _call("compare_groups", sql=(
            "SELECT product_category, discount_pct FROM ecommerce_orders WHERE order_date >= '2025-10-01'"),
            group_column="product_category", value_column="discount_pct")]
        k = n - len(script)
        if k < len(follow):
            return follow[k]()
        return AIMessage("Discount hypothesis tested.")

    def _synthesis(self, task: str) -> dict[str, Any]:
        blocks = _evidence_blocks(task)
        pc = next((d for t, d in blocks.values() if t == "period_comparison"), None)
        pc_id = next((i for i, (t, _) in blocks.items() if t == "period_comparison"), None)
        cg_id = next((i for i, (t, _) in blocks.items() if t == "compare_groups"), None)
        cg = blocks[cg_id][1] if cg_id else None
        revise = "verification pass found" in task
        findings = []
        summary = "Revenue declined in Q4 2025 versus Q3 2025."
        if pc:
            worst = (pc.get("top_negative_contributors") or [{}])[0]
            summary = (f"Revenue fell from {pc['total_previous']:,.2f} to {pc['total_current']:,.2f} "
                       f"({pc['total_pct_change']}%). The largest contributor was the {worst.get('customer_segment')} "
                       f"segment in the {worst.get('region')} region.")
            findings.append({
                "title": "Revenue declined quarter over quarter",
                "statement": f"Revenue moved from {pc['total_previous']:,.2f} to {pc['total_current']:,.2f}, "
                             f"a change of {pc['total_delta']:,.2f} ({pc['total_pct_change']}%).",
                "evidence_ids": [pc_id], "confidence": "high"})
            findings.append({
                "title": f"{worst.get('customer_segment')} / {worst.get('region')} drove the decline",
                "statement": f"{worst.get('customer_segment')} customers in {worst.get('region')} fell by "
                             f"{worst.get('delta'):,.2f} ({worst.get('pct_change')}%), "
                             f"{worst.get('share_of_total_change_pct')}% of the total change.",
                "evidence_ids": [pc_id], "confidence": "high"})
        if cg:
            top = cg["groups"][0]
            findings.append({
                "title": f"{top['group']} discounts were highest",
                "statement": f"Mean discount for {top['group']} was {top['mean']} in Q4 2025; {cg['interpretation']}",
                "evidence_ids": [cg_id], "confidence": "medium"})
        if self.hallucinate_first and not revise:
            findings.append({"title": "Unsupported claim", "statement": "Customer churn rose to 37.9% (123,457 customers).",
                             "evidence_ids": [], "confidence": "low"})
        return {"title": "Q4 2025 revenue decline", "summary": summary, "findings": findings,
                "limitations": ["Observational data: attribution is not causal.",
                                "days_to_ship has missing values."],
                "recommendations": ["Investigate SMB accounts in the West region."]}


def scripted_ecommerce_llm(**kw: Any) -> ScriptedChatModel:
    return ScriptedChatModel(responder=EcommerceAnalystPolicy(**kw))
