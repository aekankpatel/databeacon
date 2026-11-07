import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { fmtNum } from "../format";
import type { Evidence, Replay, ReplaySummary, TimelineEvent } from "../types";
import { ScopeCard } from "./AnalyzeView";
import { EvidenceDrawer } from "./EvidenceDrawer";
import { Findings, VerificationBadge } from "./Findings";
import { PlotlyChart } from "./PlotlyChart";
import { Timeline } from "./Timeline";

const SPEED = 10; // playback speed vs the recorded wall-clock time
const MAX_GAP_MS = 800; // long LLM waits are compressed further

export function ReplayList({ onOpen }: { onOpen: (id: string) => void }) {
  const [items, setItems] = useState<ReplaySummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.replays().then(setItems).catch((e) => setError(e.message));
  }, []);
  if (error) return <div className="error-box small">{error}</div>;
  if (!items) return <div className="muted small">Loading recorded investigations…</div>;
  return (
    <div className="charts" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
      {items.map((r) => (
        <button key={r.id} className="card" style={{ textAlign: "left", cursor: "pointer" }} onClick={() => onOpen(r.id)}>
          <div className="small muted">Recorded investigation</div>
          <strong>{r.question}</strong>
          <div className="small" style={{ marginTop: 6 }}>
            Answer: <span className="badge ok">{r.answer ?? "see report"}</span>
          </div>
          <div className="small muted" style={{ marginTop: 6 }}>
            {r.metrics.llm_calls} LLM calls · {r.metrics.tool_calls} tool calls · {Math.round(r.metrics.latency_s)} s ·{" "}
            {fmtNum(r.metrics.total_tokens)} tokens
          </div>
        </button>
      ))}
    </div>
  );
}

export function ReplayView({ id, onBack }: { id: string; onBack: () => void }) {
  const [replay, setReplay] = useState<Replay | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [shown, setShown] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [drawer, setDrawer] = useState<Evidence | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    setReplay(null);
    setShown(0);
    setPlaying(true);
    api.replay(id).then(setReplay).catch((e) => setError(e.message));
  }, [id]);

  const events: TimelineEvent[] = useMemo(() => replay?.result.timeline ?? [], [replay]);
  const done = shown >= events.length;

  // Playback: reveal recorded events using their recorded spacing, sped up.
  useEffect(() => {
    if (!replay || !playing || done) return;
    const cur = events[shown - 1];
    const next = events[shown];
    const gap = cur && next ? Date.parse(next.started_at) - Date.parse(cur.started_at) : 0;
    const delay = Math.max(60, Math.min(MAX_GAP_MS, (Number.isFinite(gap) ? gap : 300) / SPEED));
    timer.current = window.setTimeout(() => setShown((n) => n + 1), delay);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [replay, playing, shown, done, events]);

  if (error) return <div className="error-box">{error}</div>;
  if (!replay) return <div className="empty">Loading recorded investigation…</div>;
  const res = replay.result;
  const syn = res.synthesis;
  const visible = events.slice(0, shown).map((e, i) => (i === shown - 1 && !done ? { ...e, status: "running" as const } : e));
  const evidence = res.evidence ?? [];
  const showEvidence = (eid: string) => setDrawer(evidence.find((e) => e.id === eid) ?? null);
  const s = replay.summary;

  return (
    <div className="grid-2">
      <div className="stack">
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <button className="btn small" onClick={onBack}>← All recorded investigations</button>
            <div className="row">
              {!done && <button className="btn small" onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play"}</button>}
              {!done && <button className="btn small" onClick={() => setShown(events.length)}>Show everything</button>}
              {done && <button className="btn small" onClick={() => { setShown(0); setPlaying(true); }}>Replay</button>}
              {s.has_report && (
                <a className="btn primary small" href={api.replayReportUrl(replay.id)} target="_blank" rel="noreferrer">
                  Download PDF report
                </a>
              )}
            </div>
          </div>
          <h3 style={{ margin: "12px 0 4px" }}>{replay.question}</h3>
          <div className="small muted">
            Recorded {new Date(replay.recorded_at).toLocaleString()} on {replay.recorded_with.provider === "cloudflare"
              ? "Cloudflare Workers AI" : replay.recorded_with.provider} ({replay.recorded_with.model}) against the sample
            dataset ({fmtNum(replay.dataset.row_count)} rows). Replaying it uses no AI budget.
          </div>
          <div className="metrics" style={{ marginTop: 12 }}>
            <M l="Recorded latency" v={`${Math.round(s.metrics.latency_s)}s`} />
            <M l="LLM calls" v={s.metrics.llm_calls} />
            <M l="Tool calls" v={s.metrics.tool_calls} />
            <M l="Tokens" v={fmtNum(s.metrics.total_tokens)} />
            <M l="Neurons used then" v={`~${s.metrics.approx_neurons}`} />
            <M l="Hallucination" v={s.hallucination_rate === null ? "–" : `${(s.hallucination_rate * 100).toFixed(1)}%`} />
          </div>
        </div>

        {!done && (
          <div className="card small muted" role="status">
            Replaying the agent's recorded steps ({shown}/{events.length}) at {SPEED}× speed…
          </div>
        )}

        {done && (
          <>
            <ScopeCard spec={res.spec} alignment={res.alignment} />
            {syn && (
              <div className="card">
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <h3 style={{ margin: 0 }}>Answer</h3>
                  <VerificationBadge v={syn.summary_verification} />
                </div>
                <p className="summary">{syn.summary}</p>
                <h3>Findings</h3>
                <Findings findings={syn.findings} onEvidence={showEvidence} />
                {!!syn.limitations?.length && (
                  <>
                    <h3>Limitations</h3>
                    <ul className="muted">{syn.limitations.map((x, i) => <li key={i}>{x}</li>)}</ul>
                  </>
                )}
              </div>
            )}
            {!!res.charts?.length && (
              <div className="card">
                <h3>Visualizations</h3>
                <div className="charts">
                  {res.charts.map((c) => (
                    <div key={c.chart_id}>
                      <PlotlyChart chart={c} />
                      {c.origin === "fallback" && <div className="small"><span className="badge purple">auto fallback</span></div>}
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div className="card">
              <h3>Evidence ({evidence.length})</h3>
              <table className="data">
                <thead><tr><th>ID</th><th>Tool</th><th>Purpose / result</th></tr></thead>
                <tbody>
                  {evidence.map((e) => (
                    <tr key={e.id} onClick={() => setDrawer(e)} style={{ cursor: "pointer" }}>
                      <td className="mono">{e.id}</td>
                      <td>{e.tool}</td>
                      <td style={{ whiteSpace: "normal" }}>
                        {e.purpose && <div>{e.purpose}</div>}
                        <div className="small muted">{e.summary}</div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
      <div className="card" style={{ position: "sticky", top: 0 }}>
        <h3>Agent execution timeline <span className="badge muted">recorded</span></h3>
        <Timeline events={visible} onEvidence={showEvidence} emptyText="Starting playback…" />
      </div>
      <EvidenceDrawer evidence={drawer} onClose={() => setDrawer(null)} />
    </div>
  );
}

function M({ l, v }: { l: string; v: React.ReactNode }) {
  return <div className="metric"><div className="label">{l}</div><div className="value">{v}</div></div>;
}
