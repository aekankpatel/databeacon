"""LangGraph orchestration of the autonomous analyst.

    understand -> plan -> act <-> tools
                           |
                        reflect --(gaps & budget left)--> replan -> act
                           |
                       synthesize -> verify --(hallucinations, 1 retry)--> synthesize
                                        |
                                      report -> END
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from ..config import Settings, get_settings
from ..llm import ainvoke_with_retry, invoke_structured, recover_text_tool_calls, text_of, usage_of
from ..mcp_client import MCPToolHub
from ..observability import RunRecorder
from . import prompts as P
from .evidence import compact_json, digest_view, preview_table, summarize
from .schemas import Plan, Reflection, Synthesis
from .verifier import verify_findings


class AgentState(TypedDict, total=False):
    run_id: str
    dataset_id: str
    dataset_name: str
    question: str
    profile: dict[str, Any]
    plan: dict[str, Any]
    messages: list[BaseMessage]
    evidence: list[dict[str, Any]]
    charts: list[dict[str, Any]]
    iteration: int
    steps_in_iteration: int
    tool_calls: int
    reflections: list[dict[str, Any]]
    synthesis: dict[str, Any]
    verification: dict[str, Any]
    verify_attempts: int
    report: dict[str, Any]
    errors: list[str]
    stop_reason: str


class AnalystAgent:
    def __init__(self, llm: BaseChatModel, hub: MCPToolHub, recorder: RunRecorder, settings: Settings | None = None):
        self.llm = llm
        self.hub = hub
        self.rec = recorder
        self.s = settings or get_settings()
        self.tool_schemas = hub.llm_tools()
        self.tool_names = {t["function"]["name"] for t in self.tool_schemas}
        self.llm_tools = llm.bind_tools(self.tool_schemas)
        self._seen_calls: dict[str, str] = {}
        self.graph = self._build()

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("understand", self.understand)
        g.add_node("plan", self.plan)
        g.add_node("act", self.act)
        g.add_node("tools", self.tools)
        g.add_node("reflect", self.reflect)
        g.add_node("replan", self.replan)
        g.add_node("synthesize", self.synthesize)
        g.add_node("verify", self.verify)
        g.add_node("report", self.report)
        g.set_entry_point("understand")
        g.add_edge("understand", "plan")
        g.add_edge("plan", "act")
        g.add_conditional_edges("act", self._after_act, {"tools": "tools", "reflect": "reflect"})
        g.add_edge("tools", "act")
        g.add_conditional_edges("reflect", self._after_reflect, {"replan": "replan", "synthesize": "synthesize"})
        g.add_edge("replan", "act")
        g.add_edge("synthesize", "verify")
        g.add_conditional_edges("verify", self._after_verify, {"synthesize": "synthesize", "report": "report"})
        g.add_edge("report", END)
        return g.compile()

    async def run(self, initial: AgentState) -> AgentState:
        state: AgentState = {"evidence": [], "charts": [], "iteration": 1, "steps_in_iteration": 0, "tool_calls": 0,
                             "reflections": [], "errors": [], "verify_attempts": 0, "messages": [], **initial}
        return await self.graph.ainvoke(state, {"recursion_limit": 200})

    # ---------------------------------------------------------------- helpers
    def _ctx(self, state: AgentState) -> dict[str, Any]:
        return {"dataset_id": state["dataset_id"]}

    async def _llm(self, name: str, messages: list[BaseMessage], tools: bool = False) -> AIMessage:
        async with self.rec.span("llm", name, model=getattr(self.llm, "model_name", None)) as ev:
            resp = await ainvoke_with_retry(self.llm_tools if tools else self.llm, messages)
            usage = usage_of(resp)
            self.rec.add_tokens(usage)
            ev["tokens"] = usage
            if tools:
                resp = recover_text_tool_calls(resp, self.tool_names)
                ev["output"] = {"tool_calls": [{"name": c["name"], "args": c["args"]} for c in resp.tool_calls],
                                "text": text_of(resp)[:1500]}
            else:
                ev["output"] = text_of(resp)[:1500]
            return resp

    async def _structured(self, name: str, messages: list[BaseMessage], schema):
        async with self.rec.span("llm", name, model=getattr(self.llm, "model_name", None)) as ev:
            obj, usage = await invoke_structured(self.llm, messages, schema)
            self.rec.add_tokens(usage)
            ev["tokens"] = usage
            ev["output"] = obj.model_dump()
            return obj

    @staticmethod
    def _compact_history(msgs: list[BaseMessage], budget_chars: int = 14000) -> list[BaseMessage]:
        """Keep the prompt small: older tool results are replaced by their one-line evidence summary."""
        def size(ms):
            return sum(len(text_of(m)) + len(json.dumps(getattr(m, "tool_calls", None) or [], default=str)) for m in ms)

        if size(msgs) <= budget_chars:
            return msgs
        out = list(msgs)
        tool_idx = [i for i, m in enumerate(out) if isinstance(m, ToolMessage)]
        for i in tool_idx[:-2]:  # keep the two most recent results verbatim
            m = out[i]
            try:
                payload = json.loads(m.content)
                short = {"evidence_id": payload.get("evidence_id"),
                         "note": "older result compacted; full data kept as evidence",
                         "preview": compact_json(payload.get("result") or payload.get("error"), rows=3, max_chars=400)}
                out[i] = ToolMessage(json.dumps(short), tool_call_id=m.tool_call_id, name=m.name)
            except (json.JSONDecodeError, TypeError, AttributeError):
                out[i] = ToolMessage(text_of(m)[:300], tool_call_id=m.tool_call_id, name=m.name)
            if size(out) <= budget_chars:
                break
        return out

    def _system(self, state: AgentState, remaining_steps: int | None = None) -> SystemMessage:
        prof = state["profile"]
        latest = "unknown"
        for c in prof.get("columns", []):
            if c.get("role") == "temporal" and (c.get("stats") or {}).get("max"):
                latest = c["stats"]["max"]
                break
        return SystemMessage(P.SYSTEM.format(
            schema=prof.get("summary", ""), table=prof.get("table"), latest_date=latest,
            max_steps=remaining_steps or self.s.max_steps_per_iteration,
            remaining_calls=self.s.max_tool_calls - state.get("tool_calls", 0)))

    def _evidence_digest(self, state: AgentState, max_chars: int = 9000) -> str:
        parts = []
        for e in state.get("evidence", []):
            if e["tool"] == "profile_dataset":
                parts.append(f"[{e['id']}] profile_dataset: {e['summary']}")
                continue
            body = (compact_json(digest_view(e["tool"], e["data"]), rows=15, max_chars=2500)
                    if e["ok"] else f"ERROR: {e['error']}")
            parts.append(f"[{e['id']}] {e['tool']} - {e.get('purpose') or ''}\n{body}")
        text = "\n\n".join(parts)
        return text[-max_chars:] if len(text) > max_chars else text

    def _add_evidence(self, state: AgentState, tool: str, args: dict[str, Any], result, purpose: str = "") -> dict[str, Any]:
        eid = f"E{len(state['evidence'])}"
        data = result.data
        entry = {
            "id": eid, "tool": tool, "server": result.server, "args": args, "ok": result.ok, "error": result.error,
            "data": data, "elapsed_ms": round(result.elapsed_ms, 2), "iteration": state.get("iteration", 1),
            "purpose": purpose or args.get("purpose") or args.get("title") or "",
            "sql": args.get("sql") or (data.get("sql") if isinstance(data, dict) else None),
            "summary": summarize(tool, data) if result.ok else result.error,
            "preview": preview_table(data) if result.ok else None,
        }
        state["evidence"].append(entry)
        if result.ok and tool == "create_chart" and isinstance(data, dict):
            data["evidence_id"] = eid
            state["charts"].append(data)
        return entry

    # ------------------------------------------------------------------ nodes
    async def understand(self, state: AgentState) -> AgentState:
        async with self.rec.span("node", "understand_dataset") as ev:
            async with self.rec.span("tool", "profile_dataset", server="database", input={}) as tev:
                res = await self.hub.call("profile_dataset", {}, self._ctx(state))
                tev["status"] = "ok" if res.ok else "error"
                tev["output"] = summarize("profile_dataset", res.data) if res.ok else res.error
            if not res.ok:
                raise RuntimeError(f"Could not profile dataset: {res.error}")
            entry = self._add_evidence(state, "profile_dataset", {}, res, "Dataset understanding")
            tev["evidence_id"] = entry["id"]
            ev["output"] = res.data.get("summary")
            self.rec.publish({"kind": "profile", "profile": res.data})
            self.rec.log("discovery", "mcp_tool_discovery", output={
                "tools": [{"name": t.name, "server": t.server} for t in self.hub.tools.values()],
                "resources": len(self.hub.resources), "prompts": len(self.hub.prompts)})
        return {"profile": res.data, "evidence": state["evidence"], "charts": state["charts"]}

    async def plan(self, state: AgentState) -> AgentState:
        async with self.rec.span("node", "plan") as ev:
            msgs = [self._system(state), HumanMessage(P.PLAN.format(question=state["question"]))]
            try:
                plan = await self._structured("plan", msgs, Plan)
            except Exception as exc:  # degrade gracefully: investigate without a detailed plan
                if type(exc).__name__ == "QuotaExhaustedError":
                    raise
                state["errors"].append(f"plan: {exc}")
                self.rec.log("guardrail", "plan_fallback", status="warning", error=str(exc)[:500])
                plan = Plan(objective=state["question"],
                            steps=["Profile relevant columns", "Query the metrics the question asks about",
                                   "Test differences statistically", "Chart the key result"])
            ev["output"] = plan.model_dump()
        plan_text = json.dumps(plan.model_dump(), indent=1)
        self.rec.publish({"kind": "plan", "plan": plan.model_dump()})
        messages = [self._system(state), HumanMessage(P.EXECUTE.format(question=state["question"], plan=plan_text))]
        return {"plan": plan.model_dump(), "messages": messages, "errors": state["errors"]}

    async def act(self, state: AgentState) -> AgentState:
        msgs = list(state["messages"])
        msgs[0] = self._system(state, self.s.max_steps_per_iteration - state.get("steps_in_iteration", 0))
        budget_left = state["tool_calls"] < self.s.max_tool_calls
        step_left = state.get("steps_in_iteration", 0) < self.s.max_steps_per_iteration
        if not (budget_left and step_left):
            self.rec.log("decision", "budget_reached", output={"tool_calls": state["tool_calls"],
                                                               "steps_in_iteration": state.get("steps_in_iteration")})
            return {"messages": msgs + [AIMessage("Tool budget for this phase reached.")]}
        try:
            resp = await self._llm(f"act (iteration {state['iteration']})", self._compact_history(msgs), tools=True)
        except Exception as exc:  # LLM/provider failure -> move on with the evidence we have
            state["errors"].append(f"act: {exc}")
            return {"messages": msgs + [AIMessage(f"LLM error: {exc}")], "errors": state["errors"]}
        return {"messages": msgs + [resp], "steps_in_iteration": state.get("steps_in_iteration", 0) + 1}

    def _after_act(self, state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "reflect"

    async def tools(self, state: AgentState) -> AgentState:
        last: AIMessage = state["messages"][-1]
        new_msgs: list[BaseMessage] = []
        for call in last.tool_calls:
            name, args = call["name"], dict(call.get("args") or {})
            key = name + json.dumps(args, sort_keys=True, default=str)
            if state["tool_calls"] >= self.s.max_tool_calls:
                new_msgs.append(ToolMessage(json.dumps({"error": "Tool-call budget exhausted. Stop calling tools."}),
                                            tool_call_id=call["id"], name=name))
                continue
            if key in self._seen_calls:
                self.rec.log("guardrail", "duplicate_call_blocked", input={"tool": name, "args": args})
                new_msgs.append(ToolMessage(json.dumps({"note": f"Identical call already made; see evidence "
                                                                f"{self._seen_calls[key]}."}),
                                            tool_call_id=call["id"], name=name))
                continue
            info = self.hub.tools.get(name)
            async with self.rec.span("tool", name, server=info.server if info else "?", input=args) as tev:
                res = await self.hub.call(name, args, self._ctx(state))
                state["tool_calls"] += 1
                entry = self._add_evidence(state, name, args, res)
                tev["evidence_id"] = entry["id"]
                tev["status"] = "ok" if res.ok else "error"
                tev["output"] = entry["summary"]
                if entry.get("sql"):
                    tev["sql"] = entry["sql"]
                if not res.ok:
                    tev["error"] = res.error
                    state["errors"].append(f"{name}: {res.error}")
            self._seen_calls[key] = entry["id"]
            if res.ok:
                payload = {"evidence_id": entry["id"], "result": json.loads(compact_json(res.data, rows=self.s.llm_result_rows))
                           if name != "create_chart" else {"chart_id": res.data.get("chart_id"), "created": True}}
                content = json.dumps(payload, default=str)
            else:
                content = json.dumps({"evidence_id": entry["id"], "error": res.error,
                                      "hint": "Fix the arguments (check column names / SQL) and retry."})
            new_msgs.append(ToolMessage(content[:6000], tool_call_id=call["id"], name=name))
            self.rec.publish({"kind": "evidence", "evidence": {k: v for k, v in entry.items() if k != "data"}})
            if res.ok and name == "create_chart":
                self.rec.publish({"kind": "chart", "chart": res.data})
        return {"messages": state["messages"] + new_msgs, "tool_calls": state["tool_calls"],
                "evidence": state["evidence"], "charts": state["charts"], "errors": state["errors"]}

    async def reflect(self, state: AgentState) -> AgentState:
        async with self.rec.span("node", f"reflect (iteration {state['iteration']})") as ev:
            hyps = "\n".join(f"{h['id']}: {h['statement']}" for h in state["plan"].get("hypotheses", [])) or "-"
            msgs = [self._system(state), HumanMessage(P.REFLECT.format(
                question=state["question"], hypotheses=hyps, evidence=self._evidence_digest(state)))]
            try:
                refl = await self._structured("reflect", msgs, Reflection)
            except Exception as exc:
                state["errors"].append(f"reflect: {exc}")
                refl = Reflection(sufficient=True, confidence=0.3, gaps=[f"reflection failed: {exc}"])
            ev["output"] = refl.model_dump()
        reflections = state["reflections"] + [{"iteration": state["iteration"], **refl.model_dump()}]
        self.rec.publish({"kind": "reflection", "reflection": reflections[-1]})
        can_continue = state["iteration"] < self.s.max_iterations and state["tool_calls"] < self.s.max_tool_calls
        stop_reason = ""
        if refl.sufficient or not can_continue:
            stop_reason = "evidence sufficient" if refl.sufficient else (
                "max iterations reached" if state["iteration"] >= self.s.max_iterations else "tool budget exhausted")
            self.rec.log("decision", "stop_investigation", output={"reason": stop_reason, "iteration": state["iteration"],
                                                                   "confidence": refl.confidence})
        return {"reflections": reflections, "stop_reason": stop_reason}

    def _after_reflect(self, state: AgentState) -> str:
        return "synthesize" if state.get("stop_reason") else "replan"

    async def replan(self, state: AgentState) -> AgentState:
        refl = state["reflections"][-1]
        self.rec.log("decision", "replan", output={"gaps": refl["gaps"], "next_actions": refl["next_actions"]})
        msg = HumanMessage(P.REPLAN.format(iteration=state["iteration"], gaps="; ".join(refl["gaps"]) or "-",
                                           next_actions="; ".join(refl["next_actions"]) or "-"))
        return {"messages": state["messages"] + [msg], "iteration": state["iteration"] + 1, "steps_in_iteration": 0}

    async def synthesize(self, state: AgentState) -> AgentState:
        feedback = ""
        ver = state.get("verification")
        if ver and ver.get("unsupported"):
            issues = "\n".join(f"- {u['finding']}: {', '.join(u['values'])}" for u in ver["unsupported"])
            feedback = P.VERIFY_FEEDBACK.format(issues=issues)
        async with self.rec.span("node", "synthesize" + (" (revision)" if feedback else "")) as ev:
            charts = ", ".join(f"{c['evidence_id']}: {c.get('title')}" for c in state["charts"]) or "none"
            msgs = [self._system(state), HumanMessage(P.SYNTHESIZE.format(
                question=state["question"], evidence=self._evidence_digest(state, 11000), charts=charts,
                feedback=feedback))]
            syn = await self._structured("synthesize", msgs, Synthesis)
            ev["output"] = {"summary": syn.summary, "findings": len(syn.findings)}
        return {"synthesis": syn.model_dump()}

    async def verify(self, state: AgentState) -> AgentState:
        async with self.rec.span("node", "verify_findings") as ev:
            findings = [dict(f) for f in state["synthesis"]["findings"]]
            # the summary is verified too, as a pseudo-finding citing all evidence
            all_ids = [e["id"] for e in state["evidence"] if e["ok"] and e["tool"] not in ("profile_dataset", "create_chart")]
            summary_check = {"title": "Executive summary", "statement": state["synthesis"]["summary"],
                             "evidence_ids": all_ids}
            result = verify_findings(findings + [summary_check], state["evidence"])
            state["synthesis"]["findings"] = findings
            state["synthesis"]["summary_verification"] = summary_check["verification"]
            ev["output"] = {k: v for k, v in result.items() if k != "unsupported"} | {"unsupported": result["unsupported"]}
            ev["status"] = "ok" if result["hallucination_rate"] == 0 else "warning"
        return {"verification": result, "synthesis": state["synthesis"], "verify_attempts": state["verify_attempts"] + 1}

    def _after_verify(self, state: AgentState) -> str:
        if state["verification"]["hallucination_rate"] > 0 and state["verify_attempts"] < 2:
            self.rec.log("decision", "revise_unsupported_claims", output=state["verification"]["unsupported"])
            return "synthesize"
        return "report"

    async def _ensure_chart(self, state: AgentState) -> None:
        if state["charts"]:
            return
        for e in reversed(state["evidence"]):
            d = e.get("data")
            if not (e["ok"] and e["tool"] == "run_sql" and isinstance(d, dict)):
                continue
            cols, dtypes = d.get("columns", []), d.get("dtypes", [])
            if not (2 <= len(cols) and 2 <= d.get("row_count", 0) <= 60):
                continue
            num = [c for c, t in zip(cols[1:], dtypes[1:]) if any(k in t.upper() for k in ("INT", "DOUBLE", "DECIMAL", "FLOAT"))]
            if not num:
                continue
            ctype = "line" if any(k in (dtypes[0] or "").upper() for k in ("DATE", "TIME")) else "bar"
            args = {"sql": e["sql"], "chart_type": ctype, "x": cols[0], "y": [num[0]],
                    "title": e.get("purpose") or f"{num[0]} by {cols[0]}"}
            async with self.rec.span("tool", "create_chart", server="report", input=args) as tev:
                res = await self.hub.call("create_chart", args, self._ctx(state))
                tev["status"] = "ok" if res.ok else "error"
            if res.ok:
                entry = self._add_evidence(state, "create_chart", args, res, "Auto-generated chart")
                tev["evidence_id"] = entry["id"]
                self.rec.publish({"kind": "chart", "chart": res.data})
                return

    async def report(self, state: AgentState) -> AgentState:
        async with self.rec.span("node", "generate_report") as ev:
            await self._ensure_chart(state)
            syn = state["synthesis"]
            ver = state["verification"]
            metrics = self.rec.metrics()
            run_metrics = {
                "iterations": state["iteration"], "tool_calls": state["tool_calls"],
                "sql_queries": metrics["sql_queries"], "findings_verified": f"{ver['findings_verified']}/{ver['findings_total']}",
                "numeric_claims_supported": f"{ver['numeric_claims_supported']}/{ver['numeric_claims']}",
                "hallucination_rate": ver["hallucination_rate"], "total_tokens": metrics["total_tokens"],
                "latency_s": metrics["latency_s"],
            }
            args = {
                "run_id": state["run_id"], "title": syn.get("title") or "DataBeacon analysis",
                "question": state["question"], "dataset": state.get("dataset_name", state["dataset_id"]),
                "summary": syn["summary"], "findings": syn["findings"],
                "charts": [{k: v for k, v in c.items() if k != "figure"} for c in state["charts"]],
                "evidence": [{k: e.get(k) for k in ("id", "tool", "purpose", "sql", "summary", "preview")}
                             for e in state["evidence"] if e["ok"] and e["tool"] != "create_chart"],
                "limitations": syn.get("limitations", []),
                "timeline": [{k: e.get(k) for k in ("type", "name", "duration_ms", "status")} for e in self.rec.events],
                "metrics": run_metrics,
            }
            res = await self.hub.call("generate_pdf_report", args, self._ctx(state))
            ev["status"] = "ok" if res.ok else "error"
            ev["output"] = res.data if res.ok else res.error
            report = {"ok": res.ok, "path": (res.data or {}).get("path") if res.ok else None, "error": res.error,
                      "metrics": run_metrics}
        return {"report": report, "charts": state["charts"], "evidence": state["evidence"]}
