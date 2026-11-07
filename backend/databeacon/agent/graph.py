"""LangGraph orchestration of the autonomous analyst.

    understand -> scope --(period ambiguous)--> END (needs_clarification)
                    |
                   plan -> act <-> tools
                           |
                        reflect --(gaps & budget left)--> replan -> act
                           |
                       synthesize -> verify --(misaligned, budget left)--> realign -> act
                                        |  \\--(unsupported numbers / misaligned)--> synthesize (revision)
                                      report -> END
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from ..config import Settings, get_settings
from ..llm import ainvoke_with_retry, invoke_structured, recover_text_tool_calls, text_of, usage_of
from ..mcp_client import MCPToolHub
from ..observability import RunRecorder
from . import prompts as P
from .alignment import TaskSpec, build_spec, check_alignment
from .evidence import compact_json, digest_view, preview_table, summarize
from .schemas import Plan, Reflection, Synthesis
from .temporal import Coverage, TemporalResolution, resolve
from .verifier import verify_findings

STAGES = ("plan", "act", "reflect", "synthesize")


class AgentState(TypedDict, total=False):
    run_id: str
    dataset_id: str
    dataset_name: str
    question: str
    clarification: str | None
    profile: dict[str, Any]
    temporal: dict[str, Any]
    spec: dict[str, Any]
    needs_clarification: dict[str, Any] | None
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
    alignment: dict[str, Any]
    verify_attempts: int
    realign_attempts: int
    report: dict[str, Any]
    errors: list[str]
    stop_reason: str
    deadline_at: float | None


def normalize_call(name: str, args: dict[str, Any]) -> str:
    """Key for duplicate detection: SQL compared modulo whitespace, case and a trailing semicolon."""
    norm = {}
    for k, v in sorted(args.items()):
        if k == "purpose" or k == "title":
            continue
        if isinstance(v, str) and k == "sql":
            v = re.sub(r"\s+", " ", v).strip().rstrip(";").strip().lower()
        norm[k] = v
    return name + json.dumps(norm, sort_keys=True, default=str)


class AnalystAgent:
    def __init__(self, llm: BaseChatModel, hub: MCPToolHub, recorder: RunRecorder, settings: Settings | None = None):
        self.llm = llm
        self.hub = hub
        self.rec = recorder
        self.s = settings or get_settings()
        self.tool_schemas = hub.llm_tools()
        self.tool_names = {t["function"]["name"] for t in self.tool_schemas}
        self.tool_schema_chars = len(json.dumps(self.tool_schemas, separators=(",", ":")))
        self.llm_tools = llm.bind_tools(self.tool_schemas)
        self._seen_calls: dict[str, str] = {}
        self.graph = self._build()

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(AgentState)
        for name in ("understand", "scope", "plan", "act", "tools", "reflect", "replan", "synthesize", "verify",
                     "realign", "report"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("understand")
        g.add_edge("understand", "scope")
        g.add_conditional_edges("scope", lambda s: END if s.get("needs_clarification") else "plan",
                                {END: END, "plan": "plan"})
        g.add_edge("plan", "act")
        g.add_conditional_edges("act", self._after_act, {"tools": "tools", "reflect": "reflect"})
        g.add_edge("tools", "act")
        g.add_conditional_edges("reflect", self._after_reflect, {"replan": "replan", "synthesize": "synthesize"})
        g.add_edge("replan", "act")
        g.add_edge("realign", "act")
        g.add_edge("synthesize", "verify")
        g.add_conditional_edges("verify", self._after_verify,
                                {"synthesize": "synthesize", "realign": "realign", "report": "report"})
        g.add_edge("report", END)
        return g.compile()

    async def run(self, initial: AgentState) -> AgentState:
        state: AgentState = {"evidence": [], "charts": [], "iteration": 1, "steps_in_iteration": 0, "tool_calls": 0,
                             "reflections": [], "errors": [], "verify_attempts": 0, "realign_attempts": 0,
                             "messages": [], "clarification": None, **initial}
        return await self.graph.ainvoke(state, {"recursion_limit": 200})

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _time_left(state: AgentState) -> float:
        """Seconds left before the run's deadline (inf when there is none)."""
        import time as _t

        d = state.get("deadline_at")
        return float("inf") if not d else d - _t.time()

    # keep enough time to synthesize, verify and write the report before a serverless time limit
    DEADLINE_RESERVE_S = 75

    def _ctx(self, state: AgentState) -> dict[str, Any]:
        return {"dataset_id": state["dataset_id"]}

    def _spec(self, state: AgentState) -> TaskSpec | None:
        return TaskSpec(**state["spec"]) if state.get("spec") else None

    def _contract(self, state: AgentState) -> str:
        spec = self._spec(state)
        if spec:
            return spec.prompt_block()
        t = state.get("temporal") or {}
        if t.get("current"):
            return (f"Time scope: {t['current']['label']} = [{t['current']['start']}, {t['current']['end']})"
                    + (f" vs {t['previous']['label']} = [{t['previous']['start']}, {t['previous']['end']})"
                       if t.get("previous") else "")
                    + "".join(f"\n- assumption: {a}" for a in t.get("assumptions", [])))
        return "Time scope: none specified."

    def _system_act(self, state: AgentState, remaining_steps: int) -> SystemMessage:
        prof = state["profile"]
        return SystemMessage(P.SYSTEM_ACT.format(
            schema=prof.get("summary", ""), contract=self._contract(state), table=prof.get("table"),
            max_steps=remaining_steps, remaining_calls=self.s.max_tool_calls - state.get("tool_calls", 0)))

    def _system_brief(self, state: AgentState) -> SystemMessage:
        return SystemMessage(P.SYSTEM_BRIEF.format(schema=state["profile"].get("summary", ""),
                                                   contract=self._contract(state)))

    async def _llm(self, name: str, stage: str, messages: list[BaseMessage], tools: bool = False) -> AIMessage:
        async with self.rec.span("llm", name, model=getattr(self.llm, "model_name", None), stage=stage) as ev:
            ev["prompt_chars"] = self._prompt_chars(messages, tools)
            resp = await ainvoke_with_retry(self.llm_tools if tools else self.llm, messages)
            usage = usage_of(resp)
            self.rec.add_tokens(usage, getattr(self.llm, "last_served", None), stage=stage)
            ev["tokens"] = usage
            if getattr(self.llm, "last_served", None):
                ev["served_by"] = self.llm.last_served
            if tools:
                resp = recover_text_tool_calls(resp, self.tool_names)
                ev["output"] = {"tool_calls": [{"name": c["name"], "args": c["args"]} for c in resp.tool_calls],
                                "text": text_of(resp)[:1500]}
            else:
                ev["output"] = text_of(resp)[:1500]
            return resp

    async def _structured(self, name: str, stage: str, messages: list[BaseMessage], schema):
        async with self.rec.span("llm", name, model=getattr(self.llm, "model_name", None), stage=stage) as ev:
            ev["prompt_chars"] = self._prompt_chars(messages, False)
            obj, usage = await invoke_structured(self.llm, messages, schema)
            self.rec.add_tokens(usage, getattr(self.llm, "last_served", None), stage=stage)
            ev["tokens"] = usage
            if getattr(self.llm, "last_served", None):
                ev["served_by"] = self.llm.last_served
            ev["output"] = obj.model_dump()
            return obj

    def _prompt_chars(self, messages: list[BaseMessage], tools: bool) -> dict[str, int]:
        """Composition of a prompt, for token profiling (characters; ~4 chars per token)."""
        sys_chars = sum(len(text_of(m)) for m in messages if isinstance(m, SystemMessage))
        tool_results = sum(len(text_of(m)) for m in messages if isinstance(m, ToolMessage))
        calls = sum(len(json.dumps(m.tool_calls, default=str)) for m in messages
                    if isinstance(m, AIMessage) and m.tool_calls)
        other = sum(len(text_of(m)) for m in messages if not isinstance(m, (SystemMessage, ToolMessage)))
        return {"system": sys_chars, "tool_schemas": self.tool_schema_chars if tools else 0,
                "tool_results": tool_results, "tool_calls": calls, "other": other}

    @staticmethod
    def _compact_history(msgs: list[BaseMessage], budget_chars: int = 9000) -> list[BaseMessage]:
        """Keep the prompt small: older tool results are replaced by a short preview of their evidence."""
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
                short = {"evidence_id": payload.get("evidence_id"), "note": "older result compacted",
                         "preview": compact_json(payload.get("result") or payload.get("error"), rows=3, max_chars=300)}
                out[i] = ToolMessage(json.dumps(short), tool_call_id=m.tool_call_id, name=m.name)
            except (json.JSONDecodeError, TypeError, AttributeError):
                out[i] = ToolMessage(text_of(m)[:300], tool_call_id=m.tool_call_id, name=m.name)
            if size(out) <= budget_chars:
                break
        return out

    def _evidence_digest(self, state: AgentState, max_chars: int = 7000, per_item: int = 1500) -> str:
        parts = []
        for e in state.get("evidence", []):
            if e["tool"] == "profile_dataset":
                continue  # already in the system prompt
            if e["tool"] == "create_chart":
                parts.append(f"[{e['id']}] create_chart - {e.get('purpose') or ''}")
                continue
            body = (compact_json(digest_view(e["tool"], e["data"]), rows=12, max_chars=per_item)
                    if e["ok"] else f"ERROR: {str(e['error'])[:200]}")
            parts.append(f"[{e['id']}] {e['tool']} - {e.get('purpose') or ''}\n{body}")
        text = "\n\n".join(parts)
        return text[-max_chars:] if len(text) > max_chars else text

    def _ledger(self, state: AgentState) -> str:
        rows = [f"{e['id']} {e['tool']}: {e.get('purpose') or ''} -> {e['summary']}"[:200]
                for e in state["evidence"] if e["tool"] != "profile_dataset"]
        return "\n".join(rows) or "-"

    def _add_evidence(self, state: AgentState, tool: str, args: dict[str, Any], result, purpose: str = "",
                      origin: str = "agent") -> dict[str, Any]:
        """origin: 'agent' (chosen by the model), 'system' (run by the pipeline, e.g. profiling) or
        'fallback' (added automatically because the agent did not do it)."""
        eid = f"E{len(state['evidence'])}"
        data = result.data
        entry = {"origin": origin,
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
            data["origin"] = origin
            state["charts"].append(data)
        return entry

    # ------------------------------------------------------------------ nodes
    async def understand(self, state: AgentState) -> AgentState:
        async with self.rec.span("node", "understand_dataset") as ev:
            async with self.rec.span("tool", "profile_dataset", server="database", input={}, origin="system") as tev:
                res = await self.hub.call("profile_dataset", {}, self._ctx(state))
                tev["status"] = "ok" if res.ok else "error"
                tev["output"] = summarize("profile_dataset", res.data) if res.ok else res.error
            if not res.ok:
                raise RuntimeError(f"Could not profile dataset: {res.error}")
            entry = self._add_evidence(state, "profile_dataset", {}, res, "Dataset understanding", origin="system")
            tev["evidence_id"] = entry["id"]
            ev["output"] = res.data.get("summary")
            self.rec.publish({"kind": "profile", "profile": res.data})
            self.rec.log("discovery", "mcp_tool_discovery", output={
                "tools": [{"name": t.name, "server": t.server} for t in self.hub.tools.values()],
                "llm_tools": sorted(self.tool_names), "resources": len(self.hub.resources),
                "prompts": len(self.hub.prompts)})
        return {"profile": res.data, "evidence": state["evidence"], "charts": state["charts"]}

    async def _coverage(self, state: AgentState, column: str) -> Coverage | None:
        q = '"' + column.replace('"', '""') + '"'
        table = state["profile"]["table"]
        sql = (f"SELECT MIN({q})::DATE AS min_d, MAX({q})::DATE AS max_d, COUNT(DISTINCT {q}::DATE) AS n, "
               f"(SELECT MEDIAN(g) FROM (SELECT d - LAG(d) OVER (ORDER BY d) AS g FROM (SELECT DISTINCT {q}::DATE AS d "
               f"FROM \"{table}\" WHERE {q} IS NOT NULL)) WHERE g IS NOT NULL) AS spacing FROM \"{table}\"")
        res = await self.hub.call("run_sql", {"sql": sql, "purpose": f"coverage of {column}"}, self._ctx(state))
        if not res.ok or not res.data["rows"] or res.data["rows"][0][0] is None:
            return None
        mn, mx, n, spacing = res.data["rows"][0]
        return Coverage(column, date.fromisoformat(str(mn)[:10]), date.fromisoformat(str(mx)[:10]), int(n),
                        max(1, int(round(float(spacing)))) if spacing is not None else 1)

    async def scope(self, state: AgentState) -> AgentState:
        """Resolve the question's time scope deterministically from data coverage (no LLM tokens)."""
        async with self.rec.span("node", "resolve_time_scope") as ev:
            cols = state["profile"].get("temporal_columns", [])
            coverages = {}
            for c in cols:
                cov = await self._coverage(state, c)
                if cov:
                    coverages[c] = cov
            tr: TemporalResolution = resolve(state["question"], cols, coverages, state.get("clarification"))
            ev["output"] = tr.to_dict()
            if tr.status == "needs_clarification":
                ev["status"] = "warning"
                need = {"question": tr.clarification, "options": tr.options, "phrase": tr.phrase}
                self.rec.log("decision", "clarification_needed", status="warning", output=need)
                self.rec.publish({"kind": "clarification", "clarification": need})
                return {"temporal": tr.to_dict(), "needs_clarification": need, "stop_reason": "needs clarification"}
            if tr.status == "resolved":
                self.rec.log("decision", "time_scope_resolved", output={
                    "current": tr.current.label if tr.current else None,
                    "previous": tr.previous.label if tr.previous else None, "assumptions": tr.assumptions})
        return {"temporal": tr.to_dict(), "needs_clarification": None}

    async def plan(self, state: AgentState) -> AgentState:
        async with self.rec.span("node", "plan") as ev:
            msgs = [self._system_brief(state), HumanMessage(P.PLAN.format(question=state["question"]))]
            try:
                plan = await self._structured("plan", "plan", msgs, Plan)
            except Exception as exc:  # degrade gracefully: investigate without a detailed plan
                if type(exc).__name__ == "QuotaExhaustedError":
                    raise
                state["errors"].append(f"plan: {exc}")
                self.rec.log("guardrail", "plan_fallback", status="warning", error=str(exc)[:500])
                plan = Plan(objective=state["question"],
                            steps=["Query the metrics the question asks about", "Test differences statistically",
                                   "Chart the key result"])
            ev["output"] = plan.model_dump()
        tr = state.get("temporal") or {}
        tr_obj = None
        if tr.get("status") == "resolved" and tr.get("current"):
            from .temporal import Period

            tr_obj = TemporalResolution(**{**tr, "current": Period(**_period_args(tr["current"])),
                                           "previous": Period(**_period_args(tr["previous"])) if tr.get("previous") else None})
        columns = [c["name"] for c in state["profile"].get("columns", [])]
        spec = build_spec(state["question"], plan.model_dump(), tr_obj, columns)
        self.rec.log("decision", "task_contract", output=spec.to_dict())
        self.rec.publish({"kind": "plan", "plan": plan.model_dump(), "spec": spec.to_dict()})
        compact_plan = json.dumps({k: v for k, v in plan.model_dump().items() if v not in ("", [], None)},
                                  separators=(",", ":"))
        state["spec"] = spec.to_dict()
        messages = [self._system_act(state, self.s.max_steps_per_iteration),
                    HumanMessage(P.EXECUTE.format(question=state["question"], plan=compact_plan))]
        return {"plan": plan.model_dump(), "spec": spec.to_dict(), "messages": messages, "errors": state["errors"]}

    async def act(self, state: AgentState) -> AgentState:
        msgs = list(state["messages"])
        msgs[0] = self._system_act(state, self.s.max_steps_per_iteration - state.get("steps_in_iteration", 0))
        budget_left = state["tool_calls"] < self.s.max_tool_calls
        step_left = state.get("steps_in_iteration", 0) < self.s.max_steps_per_iteration
        if self._time_left(state) < self.DEADLINE_RESERVE_S:
            self.rec.log("decision", "deadline_reached", output={
                "seconds_left": round(self._time_left(state)), "action": "stop investigating; write the answer"})
            return {"messages": msgs + [AIMessage("Time limit approaching: stop and answer with the evidence collected.")],
                    "stop_reason": "time limit approaching"}
        if not (budget_left and step_left):
            self.rec.log("decision", "budget_reached", output={"tool_calls": state["tool_calls"],
                                                               "steps_in_iteration": state.get("steps_in_iteration")})
            return {"messages": msgs + [AIMessage("Tool budget for this phase reached.")]}
        try:
            resp = await self._llm(f"act (iteration {state['iteration']})", "act", self._compact_history(msgs),
                                   tools=True)
        except Exception as exc:  # LLM/provider failure -> move on with the evidence we have
            if type(exc).__name__ == "QuotaExhaustedError":
                raise
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
            key = normalize_call(name, args)
            if state["tool_calls"] >= self.s.max_tool_calls:
                new_msgs.append(ToolMessage(json.dumps({"error": "Tool-call budget exhausted. Stop calling tools."}),
                                            tool_call_id=call["id"], name=name))
                continue
            if name not in self.tool_names:
                self.rec.log("guardrail", "tool_not_offered", input={"tool": name})
                new_msgs.append(ToolMessage(json.dumps({"error": f"'{name}' is not available here. "
                                                                 f"Available: {', '.join(sorted(self.tool_names))}"}),
                                            tool_call_id=call["id"], name=name))
                continue
            if key in self._seen_calls:
                self.rec.log("guardrail", "duplicate_call_blocked", input={"tool": name, "args": args})
                new_msgs.append(ToolMessage(json.dumps({"note": f"Same call already made; reuse evidence "
                                                                f"{self._seen_calls[key]}."}),
                                            tool_call_id=call["id"], name=name))
                continue
            info = self.hub.tools.get(name)
            async with self.rec.span("tool", name, server=info.server if info else "?", input=args, origin="agent") as tev:
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
            if res.ok:
                self._seen_calls[key] = entry["id"]
                if name == "create_chart":
                    result: Any = {"chart_id": res.data.get("chart_id"), "created": True}
                else:
                    result = json.loads(compact_json(digest_view(name, res.data), rows=self.s.llm_result_rows,
                                                     max_chars=2500))
                content = json.dumps({"evidence_id": entry["id"], "result": result}, default=str,
                                     separators=(",", ":"))
            else:
                content = json.dumps({"evidence_id": entry["id"], "error": str(res.error)[:600],
                                      "hint": self._error_hint(state, name, str(res.error))})
            new_msgs.append(ToolMessage(content, tool_call_id=call["id"], name=name))
            self.rec.publish({"kind": "evidence", "evidence": {k: v for k, v in entry.items() if k != "data"}})
            if res.ok and name == "create_chart":
                self.rec.publish({"kind": "chart", "chart": res.data})
        return {"messages": state["messages"] + new_msgs, "tool_calls": state["tool_calls"],
                "evidence": state["evidence"], "charts": state["charts"], "errors": state["errors"]}

    def _error_hint(self, state: AgentState, tool: str, error: str) -> str:
        """Point the model at the fix, using the task contract where it knows the answer."""
        spec = state.get("spec") or {}
        group_by = spec.get("group_by") or []
        if ("dimensions" in error or "group_column" in error) and group_by:
            arg = "dimensions" if tool == "period_comparison" else "group_column"
            val = json.dumps(group_by) if arg == "dimensions" else json.dumps(group_by[0])
            return f"The task contract's grouping is {', '.join(group_by)}: retry with {arg}={val}."
        if "Unknown column" in error or "Binder Error" in error:
            return "Check the column names against the dataset profile and retry."
        return "Fix the arguments (check column names / SQL) and retry."

    async def reflect(self, state: AgentState) -> AgentState:
        async with self.rec.span("node", f"reflect (iteration {state['iteration']})") as ev:
            hyps = "\n".join(f"{h['id']}: {h['statement']}" for h in state["plan"].get("hypotheses", [])) or "-"
            msgs = [self._system_brief(state), HumanMessage(P.REFLECT.format(
                question=state["question"], hypotheses=hyps, evidence=self._evidence_digest(state, 6000, 1000)))]
            try:
                refl = await self._structured("reflect", "reflect", msgs, Reflection)
            except Exception as exc:
                if type(exc).__name__ == "QuotaExhaustedError":
                    raise
                state["errors"].append(f"reflect: {exc}")
                refl = Reflection(sufficient=True, confidence=0.3, gaps=[f"reflection failed: {exc}"])
            ev["output"] = refl.model_dump()
        reflections = state["reflections"] + [{"iteration": state["iteration"], **refl.model_dump()}]
        self.rec.publish({"kind": "reflection", "reflection": reflections[-1]})
        can_continue = (state["iteration"] < self.s.max_iterations and state["tool_calls"] < self.s.max_tool_calls
                        and self._time_left(state) > self.DEADLINE_RESERVE_S + 30)
        stop_reason = ""
        if refl.sufficient or not can_continue:
            stop_reason = "evidence sufficient" if refl.sufficient else (
                "max iterations reached" if state["iteration"] >= self.s.max_iterations else
                "tool budget exhausted" if state["tool_calls"] >= self.s.max_tool_calls else "time limit approaching")
            self.rec.log("decision", "stop_investigation", output={"reason": stop_reason, "iteration": state["iteration"],
                                                                   "confidence": refl.confidence})
        return {"reflections": reflections, "stop_reason": stop_reason}

    def _after_reflect(self, state: AgentState) -> str:
        return "synthesize" if state.get("stop_reason") else "replan"

    async def replan(self, state: AgentState) -> AgentState:
        refl = state["reflections"][-1]
        self.rec.log("decision", "replan", output={"gaps": refl["gaps"], "next_actions": refl["next_actions"]})
        msg = HumanMessage(P.REPLAN.format(iteration=state["iteration"], gaps="; ".join(refl["gaps"]) or "-",
                                           next_actions="; ".join(refl["next_actions"]) or "-",
                                           ledger=self._ledger(state)))
        return {"messages": state["messages"] + [msg], "iteration": state["iteration"] + 1, "steps_in_iteration": 0}

    async def realign(self, state: AgentState) -> AgentState:
        issues = state["alignment"]["issues"]
        self.rec.log("decision", "realign_to_question", status="warning", output=issues)
        msg = HumanMessage(P.REALIGN.format(issues="\n".join(f"- {i}" for i in issues), contract=self._contract(state)))
        return {"messages": state["messages"] + [msg], "iteration": state["iteration"] + 1, "steps_in_iteration": 0,
                "realign_attempts": state.get("realign_attempts", 0) + 1, "stop_reason": ""}

    async def synthesize(self, state: AgentState) -> AgentState:
        problems = []
        ver = state.get("verification")
        if ver and ver.get("unsupported"):
            problems += [f"- unsupported numbers in '{u['finding']}': {', '.join(u['values'])}" for u in ver["unsupported"]]
        al = state.get("alignment")
        if al and al.get("issues"):
            problems += [f"- {i}" for i in al["issues"]]
        feedback = P.VERIFY_FEEDBACK.format(issues="\n".join(problems)) if problems else ""
        async with self.rec.span("node", "synthesize" + (" (revision)" if feedback else "")) as ev:
            charts = ", ".join(f"{c['evidence_id']}: {c.get('title')}" for c in state["charts"]) or "none"
            msgs = [self._system_brief(state), HumanMessage(P.SYNTHESIZE.format(
                question=state["question"], evidence=self._evidence_digest(state, 9000, 1500), charts=charts,
                feedback=feedback))]
            syn = await self._structured("synthesize", "synthesize", msgs, Synthesis)
            ev["output"] = {"summary": syn.summary, "findings": len(syn.findings), "answer": syn.answer.model_dump()}
        return {"synthesis": syn.model_dump()}

    async def verify(self, state: AgentState) -> AgentState:
        async with self.rec.span("node", "verify_findings") as ev:
            findings = [dict(f) for f in state["synthesis"]["findings"]]
            all_ids = [e["id"] for e in state["evidence"] if e["ok"] and e["tool"] not in ("profile_dataset", "create_chart")]
            summary_check = {"title": "Executive summary", "statement": state["synthesis"]["summary"],
                             "evidence_ids": all_ids}
            result = verify_findings(findings + [summary_check], state["evidence"])
            state["synthesis"]["findings"] = findings
            state["synthesis"]["summary_verification"] = summary_check["verification"]
            spec = self._spec(state) or TaskSpec(question=state["question"])
            alignment = check_alignment(spec, state["synthesis"], state["evidence"])
            ev["output"] = {"numbers": {k: v for k, v in result.items()}, "alignment": alignment}
            ev["status"] = "ok" if result["hallucination_rate"] == 0 and alignment["status"] == "aligned" else "warning"
        self.rec.publish({"kind": "alignment", "alignment": alignment})
        return {"verification": result, "alignment": alignment, "synthesis": state["synthesis"],
                "verify_attempts": state["verify_attempts"] + 1}

    def _after_verify(self, state: AgentState) -> str:
        misaligned = state["alignment"]["status"] == "misaligned"
        unsupported = state["verification"]["hallucination_rate"] > 0
        tools_left = self.s.max_tool_calls - state["tool_calls"]
        time_left = self._time_left(state)
        if misaligned and state.get("realign_attempts", 0) < 1 and tools_left >= 2 and time_left > 120:
            return "realign"  # wrong question answered: collect the right evidence, not just reword
        if (misaligned or unsupported) and state["verify_attempts"] < 2 and time_left > 45:
            self.rec.log("decision", "revise_answer", output={"alignment": state["alignment"]["issues"],
                                                              "unsupported": state["verification"]["unsupported"]})
            return "synthesize"
        return "report"

    async def _ensure_chart(self, state: AgentState) -> None:
        if state["charts"]:
            return
        for e in reversed(state["evidence"]):
            d = e.get("data")
            if not (e["ok"] and e["tool"] == "run_sql" and isinstance(d, dict)) or "coverage of" in (e.get("purpose") or ""):
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
            self.rec.log("decision", "fallback_chart", output={
                "reason": "the agent created no chart", "source_evidence": e["id"], "chart_type": ctype})
            async with self.rec.span("tool", "create_chart", server="report", input=args, origin="fallback") as tev:
                res = await self.hub.call("create_chart", args, self._ctx(state))
                tev["status"] = "ok" if res.ok else "error"
            if res.ok:
                entry = self._add_evidence(state, "create_chart", args, res,
                                           f"Auto-generated fallback chart from {e['id']}", origin="fallback")
                tev["evidence_id"] = entry["id"]
                self.rec.publish({"kind": "chart", "chart": res.data})
                return

    async def report(self, state: AgentState) -> AgentState:
        async with self.rec.span("node", "generate_report") as ev:
            await self._ensure_chart(state)
            syn = state["synthesis"]
            ver = state["verification"]
            al = state.get("alignment") or {}
            metrics = self.rec.metrics()
            run_metrics = {
                "iterations": state["iteration"], "tool_calls": state["tool_calls"],
                "sql_queries": metrics["sql_queries"], "findings_verified": f"{ver['findings_verified']}/{ver['findings_total']}",
                "numeric_claims_supported": f"{ver['numeric_claims_supported']}/{ver['numeric_claims']}",
                "hallucination_rate": ver["hallucination_rate"], "task_alignment": al.get("status", "unchecked"),
                "total_tokens": metrics["total_tokens"], "latency_s": metrics["latency_s"],
            }
            spec = state.get("spec") or {}
            scope = {"current": (spec.get("current") or {}).get("label"), "previous": (spec.get("previous") or {}).get("label"),
                     "metric": spec.get("metric_column"), "grouping": ", ".join(spec.get("group_by") or []) or None,
                     "assumptions": spec.get("assumptions") or [], "alignment": al}
            args = {
                "run_id": state["run_id"], "title": syn.get("title") or "DataBeacon analysis",
                "question": state["question"], "dataset": state.get("dataset_name", state["dataset_id"]),
                "summary": syn["summary"], "findings": syn["findings"],
                "charts": [{k: v for k, v in c.items() if k != "figure"} for c in state["charts"]],
                "evidence": [{k: e.get(k) for k in ("id", "tool", "purpose", "sql", "summary", "preview")}
                             for e in state["evidence"] if e["ok"] and e["tool"] != "create_chart"],
                "limitations": syn.get("limitations", []),
                "timeline": [{k: e.get(k) for k in ("type", "name", "duration_ms", "status")} for e in self.rec.events],
                "metrics": run_metrics, "scope": scope,
            }
            res = await self.hub.call("generate_pdf_report", args, self._ctx(state))
            ev["status"] = "ok" if res.ok else "error"
            ev["output"] = res.data if res.ok else res.error
            report = {"ok": res.ok, "path": (res.data or {}).get("path") if res.ok else None, "error": res.error,
                      "metrics": run_metrics}
        return {"report": report, "charts": state["charts"], "evidence": state["evidence"]}


def _period_args(p: dict[str, Any]) -> dict[str, Any]:
    return {"start": date.fromisoformat(p["start"]), "end": date.fromisoformat(p["end"]), "label": p["label"],
            "complete": p.get("complete", True), "coverage_ratio": p.get("coverage_ratio", 1.0)}
