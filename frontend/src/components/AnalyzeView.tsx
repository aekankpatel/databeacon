import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { fmtNum } from "../format";
import type { Chart, Dataset, Evidence, Plan, Reflection, Run, RunSummary, TimelineEvent } from "../types";
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
  const esRef = useRef<EventSource | null>(null);

  const loadHistory = useCallback(() => {
    api.runs(dataset.id).then(setHistory).catch(() => setHistory([]));
  }, [dataset.id]);

  useEffect(() => {
    loadHistory();
    setRun(null);
    setRunId(null);
    setLive(emptyLive());
    return () => esRef.current?.close();
  }, [dataset.id, loadHistory]);

  const openRun = useCallback(
    async (id: string) => {
      esRef.current?.close();
      setRunId(id);
      setError(null);
      const r = await api.run(id);
      setRun(r);
      if (r.status !== "running") {
        setLive({
          events: r.result?.timeline ?? [],
          plan: r.result?.plan,
          reflections: r.result?.reflections ?? [],
          charts: r.result?.charts ?? [],
          evidence: r.result?.evidence ?? [],
        });
        return;
      }
      setLive(emptyLive());
      const es = new EventSource(api.eventsUrl(id));
      esRef.current = es;
      es.onmessage = (msg) => {
        const m = JSON.parse(msg.data);
        if (m.kind === "event") {
          setLive((l) => {
            const idx = l.events.findIndex((e) => e.id === m.event.id);
            const events = idx >= 0 ? l.events.map((e, i) => (i === idx ? m.event : e)) : [...l.events, m.event];
            return { ...l, events };
          });
        } else if (m.kind === "plan") setLive((l) => ({ ...l, plan: m.plan }));
        else if (m.kind === "reflection") setLive((l) => ({ ...l, reflections: [...l.reflections, m.reflection] }));
        else if (m.kind === "chart") setLive((l) => ({ ...l, charts: [...l.charts, m.chart] }));
        else if (m.kind === "evidence") setLive((l) => ({ ...l, evidence: [...l.evidence, m.evidence] }));
        else if (m.kind === "done") {
          es.close();
          api.run(id).then((full) => {
            setRun(full);
            setLive((l) => ({ ...l, events: full.result?.timeline ?? l.events, charts: full.result?.charts ?? l.charts,
              evidence: full.result?.evidence ?? l.evidence }));
            loadHistory();
          });
        }
      };
      es.onerror = () => {
        es.close();
        api.run(id).then(setRun).catch(() => undefined);
      };
    },
    [loadHistory],
  );

  async function start() {
    if (question.trim().length < 5) return;
    setError(null);
    try {
      const { run_id } = await api.startRun(dataset.id, question.trim());
      await openRun(run_id);
      loadHistory();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const running = run?.status === "running";
  const result = run?.result;
  const syn = result?.synthesis;
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
            <button className="btn primary" onClick={start} disabled={running || question.trim().length < 5}>
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
                <span className={`badge ${running ? "info" : run.status === "completed" ? "ok" : "warn"}`}>
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
                    <td>{e.tool}</td>
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
