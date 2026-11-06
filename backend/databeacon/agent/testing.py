"""Deterministic chat models for tests and offline evaluation (no API key needed)."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
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


_CONTRACT_RE = re.compile(r"current period: (?P<cl>[^=]+?) = \[(?P<cs>[\d-]+), (?P<ce>[\d-]+)\)")
_PREV_RE = re.compile(r"comparison period: (?P<pl>[^=]+?) = \[(?P<ps>[\d-]+), (?P<pe>[\d-]+)\)")


def _contract(messages: list[BaseMessage]) -> dict[str, str] | None:
    for m in messages:
        if isinstance(m, SystemMessage):
            c, p = _CONTRACT_RE.search(str(m.content)), _PREV_RE.search(str(m.content))
            if c and p:
                return {**c.groupdict(), **p.groupdict()}
    return None


class EcommerceAnalystPolicy:
    """Scripted trajectory for the sample e-commerce dataset (exercises every graph branch).

    Reads the task contract from the system prompt, like a real model. Failure modes for tests:
      * hallucinate_first   - first synthesis contains fabricated numbers
      * inject_tool_error   - one bad SQL query, to exercise error recovery
      * wrong_quarter       - investigates the quarter before the contract's (the real failure we saw);
                              ``stubborn=True`` keeps doing so even after a realignment request
      * wrong_conclusion    - names the wrong primary entity in `answer` while mentioning the right one
    """

    def __init__(self, hallucinate_first: bool = True, inject_tool_error: bool = True, wrong_quarter: bool = False,
                 stubborn: bool = False, wrong_conclusion: bool = False):
        self.hallucinate_first = hallucinate_first
        self.inject_tool_error = inject_tool_error
        self.wrong_quarter = wrong_quarter
        self.stubborn = stubborn
        self.wrong_conclusion = wrong_conclusion

    # ------------------------------------------------------------------ periods
    def _periods(self, messages: list[BaseMessage]) -> dict[str, str]:
        c = _contract(messages) or {"cl": "Q4 2025", "cs": "2025-10-01", "ce": "2026-01-01",
                                    "pl": "Q3 2025", "ps": "2025-07-01", "pe": "2025-10-01"}
        realigned = any(isinstance(m, HumanMessage) and "does not match the question" in str(m.content)
                        for m in messages)
        if self.wrong_quarter and (self.stubborn or not realigned):
            # shift both periods one quarter back: "Q3 vs Q2" instead of "Q4 vs Q3"
            return {"cl": c["pl"], "cs": c["ps"], "ce": c["pe"], "pl": "Q2 2025", "ps": "2025-04-01", "pe": c["ps"]}
        return c

    def __call__(self, messages: list[BaseMessage], tools_bound: bool) -> AIMessage:
        if tools_bound:
            return self._act(messages)
        prompt = _last_human(messages)
        task = str(messages[-2].content) if "JSON schema" in prompt and len(messages) >= 2 else prompt
        if "Plan the investigation" in task:
            return AIMessage(json.dumps({
                "objective": "Explain the revenue decline between the contract's periods and attribute it to segments",
                "question_type": "change_attribution", "metric_column": "revenue", "aggregation": "sum",
                "group_by": ["customer_segment"],
                "answer_target": "customer_segment with the largest revenue decline",
                "hypotheses": [
                    {"id": "H1", "statement": "Revenue fell between the periods", "approach": "run_sql"},
                    {"id": "H2", "statement": "One customer segment drives most of the decline",
                     "approach": "period_comparison"},
                    {"id": "H3", "statement": "Discounting increased in some category", "approach": "compare_groups"},
                ],
                "steps": ["period totals", "attribution by segment", "segment x region", "discount test", "chart"],
            }))
        if "Does the evidence answer the question" in task:
            done = "compare_groups" in task
            return AIMessage(json.dumps({
                "sufficient": done, "confidence": 0.8 if done else 0.5,
                "hypothesis_status": {"H1": "supported", "H2": "supported", "H3": "supported" if done else "untested"},
                "gaps": [] if done else ["Discount changes by product category not tested"],
                "next_actions": [] if done else ["compare_groups discount_pct by product_category in the current period"],
            }))
        if "Write the final analysis" in task:
            return AIMessage(json.dumps(self._synthesis(task, messages)))
        return AIMessage("{}")

    def _act(self, messages: list[BaseMessage]) -> AIMessage:
        p = self._periods(messages)
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        n = len(tool_msgs)
        phase2 = any(isinstance(m, HumanMessage) and ("Reflection (iteration" in str(m.content)
                                                      or "does not match the question" in str(m.content))
                     for m in messages)
        pc = {"date_column": "order_date", "value_column": "revenue", "previous_start": p["ps"],
              "previous_end": p["pe"], "current_start": p["cs"], "current_end": p["ce"]}
        script = [
            lambda: _call("run_sql", sql=(
                f"SELECT CASE WHEN order_date >= '{p['cs']}' THEN 'current' ELSE 'previous' END AS period, "
                f"ROUND(SUM(revenue), 2) AS revenue, COUNT(*) AS orders FROM ecommerce_orders "
                f"WHERE order_date >= '{p['ps']}' AND order_date < '{p['ce']}' GROUP BY 1 ORDER BY 1"),
                purpose="Revenue in the two periods"),
            lambda: _call("period_comparison", dimensions=["customer_segment"], **pc),
            lambda: _call("period_comparison", dimensions=["customer_segment", "region"], **pc),
            lambda: _call("create_chart", sql=(
                "SELECT date_trunc('month', order_date) AS month, SUM(revenue) AS revenue FROM ecommerce_orders "
                "GROUP BY 1 ORDER BY 1"), chart_type="line", x="month", y=["revenue"], title="Monthly revenue"),
        ]
        if self.inject_tool_error:
            script.insert(1, lambda: _call("run_sql", sql="SELECT SUM(revenu) FROM ecommerce_orders",
                                           purpose="typo to exercise error recovery"))
        follow = [lambda: _call("compare_groups", sql=(
            f"SELECT product_category, discount_pct FROM ecommerce_orders WHERE order_date >= '{p['cs']}' "
            f"AND order_date < '{p['ce']}'"), group_column="product_category", value_column="discount_pct")]
        realigned = any(isinstance(m, HumanMessage) and "does not match the question" in str(m.content)
                        for m in messages)
        if realigned:
            # after a realignment request, redo the period-dependent queries with the corrected periods
            base = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
                       and "does not match the question" in str(m.content))
            k = sum(1 for m in messages[base:] if isinstance(m, ToolMessage))
            redo = [script[0], lambda: _call("period_comparison", dimensions=["customer_segment"], **pc),
                    lambda: _call("period_comparison", dimensions=["customer_segment", "region"], **pc)]
            return redo[k]() if k < len(redo) else AIMessage("Re-collected evidence for the requested periods.")
        if not phase2:
            return script[n]() if n < len(script) else AIMessage("Initial evidence collected.")
        k = n - len(script)
        return follow[k]() if k < len(follow) else AIMessage("Discount hypothesis tested.")

    def _synthesis(self, task: str, messages: list[BaseMessage]) -> dict[str, Any]:
        p = self._periods(messages)
        blocks = _evidence_blocks(task)
        pcs = [(i, d) for i, (t, d) in blocks.items() if t == "period_comparison"]
        # use the most recent evidence (after a realignment the latest blocks are the corrected ones)
        seg = next(((i, d) for i, d in reversed(pcs) if d.get("dimensions") == ["customer_segment"]), None)
        combo = next(((i, d) for i, d in reversed(pcs) if len(d.get("dimensions") or []) > 1), None)
        cg_id = next((i for i, (t, _) in blocks.items() if t == "compare_groups"), None)
        cg = blocks[cg_id][1] if cg_id else None
        if seg and seg[1].get("current_period") and seg[1].get("previous_period") and not self.stubborn:
            # write from the latest evidence, as a real model would (matters after a realignment)
            (ps, pe), (cs, ce) = seg[1]["previous_period"], seg[1]["current_period"]
            if cs != p["cs"]:
                q = lambda d: f"Q{(int(d[5:7]) - 1) // 3 + 1} {d[:4]}"  # noqa: E731
                p = {"cs": cs, "ce": ce, "ps": ps, "pe": pe, "cl": q(cs), "pl": q(ps)}
        revise = "verification found problems" in task
        findings: list[dict[str, Any]] = []
        answer: dict[str, Any] = {"metric_column": "revenue", "aggregation": "sum", "dimension": "customer_segment",
                                  "direction": "largest_decline", "current_period": [p["cs"], p["ce"]],
                                  "previous_period": [p["ps"], p["pe"]]}
        summary = f"Revenue declined in {p['cl']} versus {p['pl']}."
        if seg:
            sid, sd = seg
            top = (sd.get("top_negative_contributors") or [{}])[0]
            entity = top.get("customer_segment")
            if self.wrong_conclusion:
                entity = next((r["customer_segment"] for r in sd.get("top_negative_contributors", [])[1:]), "Consumer")
            answer.update({"primary_entity": entity, "key_value": top.get("delta"), "evidence_ids": [sid]})
            summary = (f"Revenue fell from {sd['total_previous']:,.2f} in {p['pl']} to {sd['total_current']:,.2f} in "
                       f"{p['cl']} ({sd['total_pct_change']}%). By customer segment, {entity} contributed most; "
                       f"{top.get('customer_segment')} fell by {top.get('delta'):,.2f} "
                       f"({top.get('share_of_total_change_pct')}% of the decline).")
            findings.append({
                "title": "Revenue declined period over period",
                "statement": f"Revenue moved from {sd['total_previous']:,.2f} to {sd['total_current']:,.2f}, "
                             f"a change of {sd['total_delta']:,.2f} ({sd['total_pct_change']}%).",
                "evidence_ids": [sid], "confidence": "high"})
            findings.append({
                "title": f"{top.get('customer_segment')} is the largest contributing segment",
                "statement": f"{top.get('customer_segment')} revenue fell by {top.get('delta'):,.2f} "
                             f"({top.get('pct_change')}%), {top.get('share_of_total_change_pct')}% of the total change.",
                "evidence_ids": [sid], "confidence": "high"})
        if combo:
            cid, cd = combo
            worst = (cd.get("top_negative_contributors") or [{}])[0]
            summary += (f" At segment-by-region level the sharpest drop is {worst.get('customer_segment')} in "
                        f"{worst.get('region')}.")
            findings.append({
                "title": f"{worst.get('customer_segment')} / {worst.get('region')} is the sharpest local decline",
                "statement": f"{worst.get('customer_segment')} customers in {worst.get('region')} fell by "
                             f"{worst.get('delta'):,.2f} ({worst.get('pct_change')}%).",
                "evidence_ids": [cid], "confidence": "high"})
        if cg:
            top = cg["groups"][0]
            findings.append({
                "title": f"{top['group']} discounts were highest",
                "statement": f"Mean discount for {top['group']} was {top['mean']} in {p['cl']}; {cg['interpretation']}",
                "evidence_ids": [cg_id], "confidence": "medium"})
        if self.hallucinate_first and not revise:
            findings.append({"title": "Unsupported claim", "statement": "Customer churn rose to 37.9% (123,457 customers).",
                             "evidence_ids": [], "confidence": "low"})
        return {"title": f"{p['cl']} revenue decline", "summary": summary, "answer": answer, "findings": findings,
                "limitations": ["Observational data: attribution is not causal.", "days_to_ship has missing values."],
                "recommendations": ["Investigate the largest contributing segment."]}


def scripted_ecommerce_llm(**kw: Any) -> ScriptedChatModel:
    return ScriptedChatModel(responder=EcommerceAnalystPolicy(**kw))
