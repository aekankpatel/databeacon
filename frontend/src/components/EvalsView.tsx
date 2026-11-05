import { useEffect, useState } from "react";
import { api } from "../api";

export function EvalsView() {
  const [data, setData] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.evals().then(setData).catch((e) => setError(e.message));
  }, []);
  if (error)
    return (
      <div className="card">
        <h3>Agent evaluation</h3>
        <p className="muted">{error}</p>
        <pre className="code">cd backend{"\n"}python -m evals.run            # live LLM{"\n"}python -m evals.run --mode scripted   # offline</pre>
      </div>
    );
  if (!data) return <div className="empty">Loading…</div>;
  const s = data.summary;
  const pct = (v: number | null) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(1)}%`);
  return (
    <div className="stack">
      <div className="card">
        <h3>Latest evaluation <span className="badge muted">{data.meta.mode}</span></h3>
        <p className="small muted">{data.meta.timestamp} · {data.meta.provider} · {data.meta.model}</p>
        <div className="metrics">
          <M l="Task success" v={pct(s.task_success_rate)} />
          <M l="SQL validity" v={pct(s.sql_validity)} />
          <M l="SQL gold match" v={pct(s.sql_gold_match_rate)} />
          <M l="Tool selection F1" v={pct(s.tool_selection_f1)} />
          <M l="Numerical accuracy" v={pct(s.numerical_accuracy)} />
          <M l="Hallucination rate" v={pct(s.hallucination_rate)} />
          <M l="Error recovery" v={pct(s.error_recovery_rate)} />
          <M l="Avg latency" v={`${s.avg_latency_s ?? "–"}s`} />
          <M l="Avg tokens" v={Math.round(s.avg_total_tokens ?? 0).toLocaleString()} />
        </div>
      </div>
      <div className="card">
        <h3>Per case</h3>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr><th>Case</th><th>Success</th><th>Entities</th><th>Gold SQL</th><th>Numbers</th><th>Tool F1</th><th>Halluc.</th><th>Latency</th><th>Tokens</th><th>Tools used</th></tr>
            </thead>
            <tbody>
              {data.cases.map((c: any) => (
                <tr key={c.case}>
                  <td className="mono">{c.case}</td>
                  <td><span className={`badge ${c.task_success ? "ok" : "err"}`}>{c.task_success ? "pass" : "fail"}</span></td>
                  <td className="num">{c.entity_recall}</td>
                  <td>{c.sql_gold_match ? "✓" : "✗"}</td>
                  <td className="num">{c.numerical_accuracy}</td>
                  <td className="num">{c.tool_selection.f1}</td>
                  <td className="num">{c.hallucination_rate}</td>
                  <td className="num">{c.latency_s}s</td>
                  <td className="num">{c.total_tokens?.toLocaleString()}</td>
                  <td className="small" style={{ whiteSpace: "normal" }}>{[...new Set(c.tools_used as string[])].join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function M({ l, v }: { l: string; v: React.ReactNode }) {
  return <div className="metric"><div className="label">{l}</div><div className="value">{v}</div></div>;
}
