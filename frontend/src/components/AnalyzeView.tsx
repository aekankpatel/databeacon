import { useCallback, useEffect, useRef, useState } from "react";
import { api, type StreamMessage } from "../api";
import { fmtNum } from "../format";
import type {
  Alignment, Chart, Clarification, Dataset, Evidence, Plan, Reflection, Run, RunSummary, TaskSpec, TimelineEvent,
} from "../types";
import { EvidenceDrawer } from "./EvidenceDrawer";
import { Findings, VerificationBadge } from "./Findings";
import { PlotlyChart } from "./PlotlyChart";
import { Timeline } from "./Timeline";

const EXAMPLES = [
  "Why did our revenue decline last quarter, and which customer segments contributed most?",
  "Which product category saw the largest increase in discounting recently?",
  "Which region has the slowest shipping, and is the difference significant?",
  "Are there data quality problems such as missing values or outliers?",
];

interface Live {
  events: TimelineEvent[];
  plan?: Plan;
  reflections: Reflection[];
  charts: Chart[];
  evidence: Evidence[];
}

const emptyLive = (): Live => ({ events: [], reflections: [], charts: [], evidence: [] });

export function AnalyzeView({ dataset }: { dataset: Dataset }) {
  const [question, setQuestion] = useState("");
  const [runId, setRunId] = useState<string | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [live, setLive] = useState<Live>(emptyLive);
  const [history, setHistory] = useState<RunSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [drawer, setDrawer] = useState<Evidence | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [connection, setConnection] = useState<"live" | "reconnected" | "interrupted" | null>(null);

  const loadHistory = useCallback(() => {
    api.runs(dataset.id).then(setHistory).catch(() => setHistory([]));
  }, [dataset.id]);

  useEffect(() => {
    loadHistory();
    setRun(null);
    setRunId(null);
    setLive(emptyLive());
    return () => abortRef.current?.abort();
  }, [dataset.id, loadHistory]);

  // One handler for every stream: the inline POST response, live SSE and saved-progress replays.
  // Replays resend earlier messages, so charts/evidence/reflections are de-duplicated by id.
  const handle = useCallback((m: StreamMessage) => {
    if (m.kind === "event") {
      setLive((l) => {
        const idx = l.events.findIndex((e) => e.id === m.event.id);
        const events = idx >= 0 ? l.events.map((e, i) => (i === idx ? m.event : e)) : [...l.events, m.event];
        return { ...l, events };
      });
    } else if (m.kind === "plan") setLive((l) => ({ ...l, plan: m.plan }));
    else if (m.kind === "reflection")
      setLive((l) => (l.reflections.some((r) => r.iteration === m.reflection.iteration) ? l
        : { ...l, reflections: [...l.reflections, m.reflection] }));
    else if (m.kind === "chart")
      setLive((l) => (l.charts.some((c) => c.chart_id === m.chart.chart_id) ? l : { ...l, charts: [...l.charts, m.chart] }));
    else if (m.kind === "evidence")
      setLive((l) => (l.evidence.some((e) => e.id === m.evidence.id) ? l : { ...l, evidence: [...l.evidence, m.evidence] }));
    else if (m.kind === "connection") setConnection(m.mode);
    else if (m.kind === "interrupted") setConnection("interrupted");
  }, []);

  const finish = useCallback(async (id: string) => {
    const full = await api.run(id);
    setRun(full);
    if (full.status === "interrupted") setConnection("interrupted");
    setLive((l) => ({
      ...l,
      events: full.result?.timeline?.length ? full.result.timeline : l.events,
      charts: full.result?.charts?.length ? full.result.charts : l.charts,
      evidence: full.result?.evidence?.length ? full.result.evidence : l.evidence,
      plan: full.result?.plan ?? l.plan,
      reflections: full.result?.reflections?.length ? full.result.reflections : l.reflections,
    }));
    loadHistory();
  }, [loadHistory]);

  /** Follow an existing run: live if this server executes it, otherwise its saved progress. */
  const follow = useCallback(async (id: string) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await api.followRun(id, (m) => {
        handle(m);
        if (m.kind === "done") finish(id);
      }, ctrl.signal);
    } catch (e) {
      if ((e as Error).name !== "AbortError") api.run(id).then(setRun).catch(() => undefined);
    }
  }, [handle, finish]);

  const openRun = useCallback(
    async (id: string) => {
      abortRef.current?.abort();
      setRunId(id);
      setError(null);
      setConnection(null);
      const r = await api.run(id);
      setRun(r);
      setLive({
        events: r.result?.timeline ?? [],
        plan: r.result?.plan,
        reflections: r.result?.reflections ?? [],
        charts: r.result?.charts ?? [],
        evidence: r.result?.evidence ?? [],
      });
      if (r.status === "interrupted") setConnection("interrupted");
      if (r.status === "running") {
        setConnection(r.connection ?? "reconnected");
        follow(id);
      }
    },
    [follow],
  );

  async function start(clarification?: string, rerunOf?: string) {
    const q = (rerunOf || clarification ? run?.question : undefined) || question.trim();
    if (!q || q.length < 5) return;
    setError(null);
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    let id = "";
    setLive(emptyLive());
    setConnection("live");
    try {
      const res = await api.startRun(dataset.id, q, (m) => {
        if (m.kind === "run") {
          id = m.run_id;
          setRunId(id);
          setRun({ id, dataset_id: dataset.id, question: q, status: "running", created_at: new Date().toISOString(),
            finished_at: null, result: null, execution: "inline" });
          loadHistory();
          return;
        }
        handle(m);
        if (m.kind === "done" && id) finish(id);
      }, { clarification, rerun_of: rerunOf, signal: ctrl.signal });
      if (!res.inline) {
        await openRun(res.run_id); // long-lived server: follow over SSE
        loadHistory();
      } else if (id && !ctrl.signal.aborted) {
        const r = await api.run(id); // stream ended: if it ended early, switch to saved progress
        if (r.status === "running") {
          setConnection("reconnected");
          follow(id);
        }
      }
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      if (id) {
        setConnection("reconnected"); // the stream broke but the run keeps going on the server
        follow(id);
        return;
      }
      setError((e as Error).message);
    }
  }

  const running = run?.status === "running";
  const result = run?.result;
  const syn = result?.synthesis;
  const clar = run?.status === "needs_clarification" ? result?.needs_clarification : null;
  const ver = result?.verification;
  const metrics = result?.metrics;
  const evidence = result?.evidence?.length ? result.evidence : live.evidence;
  const showEvidence = (id: string) => setDrawer(evidence.find((e) => e.id === id) ?? null);

  return (
    <div className="grid-2">
      <div className="stack">
        <div className="card ask">
          <h3>Ask a question about <span className="mono">{dataset.name}</span></h3>
          <textarea
            value={question}
            placeholder="e.g. Why did revenue decline last quarter, and which segments contributed most?"
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) start();
            }}
          />
          <div className="row" style={{ justifyContent: "space-between", marginTop: 8 }}>
            <div className="chips">
              {EXAMPLES.map((q) => (
                <span key={q} className="chip" onClick={() => setQuestion(q)}>{q}</span>
              ))}
            </div>
            <button className="btn primary" onClick={() => start()} disabled={running || question.trim().length < 5}>
              {running ? "Investigating…" : "Run analysis"}
            </button>
          </div>
          {error && <div className="error-box small" style={{ marginTop: 8 }}>{error}</div>}
        </div>

        {!runId && history.length > 0 && (
          <div className="card">
            <h3>Previous analyses</h3>
            {history.map((h) => (
              <div key={h.id} className="row" style={{ justifyContent: "space-between", padding: "4px 0" }}>
                <a href="#" onClick={(e) => { e.preventDefault(); openRun(h.id); }}>{h.question}</a>
                <span className={`badge ${h.status === "completed" ? "ok" : h.status === "running" ? "info" : "warn"}`}>
                  {h.status}
                </span>
              </div>
            ))}
          </div>
        )}

        {run && (
          <div className="card">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h3 style={{ margin: 0 }}>{syn?.title || run.question}</h3>
              <div className="row">
                <span className={`badge ${running || run.status === "needs_clarification" ? "info" : run.status === "completed" ? "ok" : "warn"}`}>
                  {running ? "running" : run.status.replaceAll("_", " ")}
                </span>
                {result?.report?.ok && (
                  <a className="btn primary small" href={api.reportUrl(run.id)} target="_blank" rel="noreferrer">
                    Download PDF report
                  </a>
                )}
                {!running && <button className="btn small" onClick={() => { setRunId(null); setRun(null); }}>New</button>}
              </div>
            </div>
            {metrics && !running && (
              <div className="metrics" style={{ marginTop: 12 }}>
                <Metric label="Latency" value={`${metrics.latency_s}s`} />
                <Metric label="Iterations" value={result?.iterations ?? "–"} />
                <Metric label="Tool calls" value={metrics.tool_calls} />
                <Metric label="SQL queries" value={metrics.sql_queries} />
                <Metric label="Tokens" value={fmtNum(metrics.total_tokens)} />
                <Metric label="Verified claims" value={ver ? `${ver.numeric_claims_supported}/${ver.numeric_claims}` : "–"} />
                <Metric label="Hallucination" value={ver ? `${(ver.hallucination_rate * 100).toFixed(1)}%` : "–"} />
              </div>
            )}
            {running && <p className="muted">The agent is planning, querying and testing hypotheses. Follow along in the timeline →</p>}
            {result?.errors?.length && !running && !syn ? (
              <div className="error-box small" style={{ marginTop: 10 }}>{result.errors.join(" · ")}</div>
            ) : null}
          </div>
        )}

        {run && connection && connection !== "live" && (
          <ConnectionBanner mode={connection} run={run} onRerun={() => start(undefined, run.id)} />
        )}

        {clar && <ClarificationCard clar={clar} onAnswer={(c) => start(c)} />}

        {(result?.spec || result?.alignment) && !running && <ScopeCard spec={result?.spec} alignment={result?.alignment} />}

        {syn && (
          <div className="card">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h3 style={{ margin: 0 }}>Answer</h3>
              <VerificationBadge v={syn.summary_verification} />
            </div>
            <p className="summary">{syn.summary}</p>
            <h3>Findings</h3>
            <Findings findings={syn.findings} onEvidence={showEvidence} />
            {!!syn.recommendations?.length && (
              <>
                <h3 style={{ marginTop: 14 }}>Recommendations</h3>
                <ul>{syn.recommendations.map((r, i) => <li key={i}>{r}</li>)}</ul>
              </>
            )}
            {!!syn.limitations?.length && (
              <>
                <h3>Limitations</h3>
                <ul className="muted">{syn.limitations.map((r, i) => <li key={i}>{r}</li>)}</ul>
              </>
            )}
          </div>
        )}

        {live.charts.length > 0 && (
          <div className="card">
            <h3>Visualizations</h3>
            <div className="charts">
              {live.charts.map((c) => (
                <div key={c.chart_id}>
                  <PlotlyChart chart={c} />
                  {c.origin === "fallback" && (
                    <div className="small"><span className="badge purple">auto fallback</span>{" "}
                      <span className="muted">The agent created no chart; this one was generated from its last tabular result.</span></div>
                  )}
                  {c.evidence_id && (
                    <div className="small muted">Source: <span className="ev-chip" onClick={() => showEvidence(c.evidence_id!)}>{c.evidence_id}</span></div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {live.plan && (
          <div className="card">
            <h3>Investigation plan</h3>
            <p className="small"><strong>Objective:</strong> {live.plan.objective}</p>
            {live.plan.time_scope && <p className="small muted">Time scope: {live.plan.time_scope}</p>}
            {live.plan.hypotheses.map((h) => {
              const latest = [...live.reflections].reverse().find((r) => r.hypothesis_status?.[h.id]);
              const st = latest?.hypothesis_status[h.id];
              return (
                <div key={h.id} className="hyp">
                  <span className="mono">{h.id}</span>
                  <div>
                    <div>{h.statement}</div>
                    {h.approach && <div className="small muted">{h.approach}</div>}
                  </div>
                  {st && <span className={`badge ${st === "supported" ? "ok" : st === "rejected" ? "err" : "muted"}`}>{st}</span>}
                </div>
              );
            })}
            {live.reflections.map((r) => (
              <div key={r.iteration} className="small" style={{ marginTop: 10 }}>
                <span className={`badge ${r.sufficient ? "ok" : "warn"}`}>
                  Reflection {r.iteration}: {r.sufficient ? "sufficient" : "needs more evidence"}
                </span>{" "}
                {!r.sufficient && r.gaps.length > 0 && <span className="muted">Gaps: {r.gaps.join("; ")}</span>}
              </div>
            ))}
          </div>
        )}

        {evidence.length > 0 && (
          <div className="card">
            <h3>Evidence ({evidence.length})</h3>
            <table className="data">
              <thead><tr><th>ID</th><th>Tool</th><th>Purpose / result</th><th>Status</th></tr></thead>
              <tbody>
                {evidence.map((e) => (
                  <tr key={e.id} onClick={() => setDrawer(e)} style={{ cursor: "pointer" }}>
                    <td className="mono">{e.id}</td>
                    <td>{e.tool}{e.origin && e.origin !== "agent" && <span className={`badge ${e.origin === "fallback" ? "purple" : "muted"}`} style={{ marginLeft: 6 }}>{e.origin === "fallback" ? "auto fallback" : "pipeline"}</span>}</td>
                    <td style={{ whiteSpace: "normal" }}>
                      {e.purpose && <div>{e.purpose}</div>}
                      <div className="small muted">{e.summary}</div>
                    </td>
                    <td><span className={`badge ${e.ok ? "ok" : "err"}`}>{e.ok ? "ok" : "error"}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card" style={{ position: "sticky", top: 0 }}>
        <h3>Agent execution timeline</h3>
        <div className="row small muted" style={{ marginBottom: 8 }}>
          <Legend color="var(--accent)" label="node" />
          <Legend color="var(--purple)" label="LLM" />
          <Legend color="var(--ok)" label="tool" />
          <Legend color="var(--warn)" label="decision" />
          <Legend color="var(--err)" label="error" />
        </div>
        <Timeline events={live.events} onEvidence={showEvidence} />
      </div>
      <EvidenceDrawer evidence={drawer} onClose={() => setDrawer(null)} />
    </div>
  );
}

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="row" style={{ gap: 4 }}>
      <span className="tl-dot" style={{ background: color, marginTop: 0 }} />
      {label}
    </span>
  );
}

function ClarificationCard({ clar, onAnswer }: { clar: Clarification; onAnswer: (c: string) => void }) {
  const [text, setText] = useState("");
  return (
    <div className="card" style={{ borderColor: "var(--accent)" }}>
      <h3>Which period did you mean?</h3>
      <p>{clar.question}</p>
      <p className="small muted">
        DataBeacon stopped before spending any LLM tokens because the period could not be determined reliably from the data.
      </p>
      <div className="row">
        {clar.options.map((o) => (
          <button key={o} className="btn" onClick={() => onAnswer(o)}>{o}</button>
        ))}
      </div>
      <div className="row" style={{ marginTop: 8 }}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Or type it, e.g. Q4 2025 or 2025-10-01 to 2025-12-31"
          style={{ flex: 1, minWidth: 220, padding: "7px 10px", border: "1px solid var(--border)", borderRadius: 8, background: "var(--panel-2)" }}
        />
        <button className="btn primary" disabled={text.trim().length < 3} onClick={() => onAnswer(text.trim())}>Continue</button>
      </div>
    </div>
  );
}

export function ScopeCard({ spec, alignment }: { spec?: TaskSpec | null; alignment?: Alignment | null }) {
  const badge = (v: string) => (v === "pass" ? "ok" : v === "fail" ? "err" : "muted");
  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h3 style={{ margin: 0 }}>Scope &amp; task alignment</h3>
        {alignment && (
          <span className={`badge ${alignment.status === "aligned" ? "ok" : "err"}`}>
            {alignment.status === "aligned" ? "answers the question asked" : "does not match the question"}
          </span>
        )}
      </div>
      {spec && (
        <div className="small" style={{ marginTop: 8 }}>
          {spec.current && (
            <div>
              <strong>Period:</strong> {spec.current.label} ({spec.current.start} – {spec.current.end}, end exclusive)
              {spec.previous && <> vs {spec.previous.label}</>}
            </div>
          )}
          {spec.metric_column && <div><strong>Metric:</strong> {spec.aggregation || "sum"}({spec.metric_column})</div>}
          {spec.group_by.length > 0 && <div><strong>Grouping:</strong> {spec.group_by.join(", ")}</div>}
          {spec.answer_target && <div><strong>Answer target:</strong> {spec.answer_target}</div>}
          {spec.assumptions.length > 0 && (
            <ul className="muted" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
              {spec.assumptions.map((a, i) => <li key={i}>{a}</li>)}
            </ul>
          )}
        </div>
      )}
      {alignment && (
        <div className="row small" style={{ marginTop: 8 }}>
          {Object.entries(alignment.checks).map(([k, v]) => (
            <span key={k} className={`badge ${badge(v)}`}>{k.replace("_", " ")}: {v}</span>
          ))}
        </div>
      )}
      {alignment?.issues.length ? (
        <ul className="unsupported" style={{ margin: "8px 0 0", paddingLeft: 18 }}>
          {alignment.issues.map((i, k) => <li key={k}>{i}</li>)}
        </ul>
      ) : null}
    </div>
  );
}

function ConnectionBanner({ mode, run, onRerun }: { mode: "reconnected" | "interrupted"; run: Run; onRerun: () => void }) {
  if (mode === "reconnected") {
    return (
      <div className="card small" role="status" style={{ borderColor: "var(--accent)" }}>
        <strong>Reconnected to saved progress.</strong> The investigation is still running on the server; you are
        seeing what it has saved so far, and new steps appear as they are saved.
      </div>
    );
  }
  return (
    <div className="card small" role="alert" style={{ borderColor: "var(--warn)" }}>
      <strong>This investigation was interrupted.</strong>{" "}
      {capitalize(run.interruption?.reason ?? "the server stopped working on it before it finished")}.{" "}
      Everything saved up to that point is shown below. An investigation can't continue from where it stopped:
      starting it again runs a new, complete investigation.
      <div className="row" style={{ marginTop: 8 }}>
        <button className="btn primary small" onClick={onRerun}>Start it again</button>
      </div>
    </div>
  );
}

function capitalize(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}
