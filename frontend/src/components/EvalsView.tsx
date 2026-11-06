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
        <p className="small muted">
          A case passes only if every check passes: primary conclusion (structured answer), period, metric, grouping,
          gold numbers in the cited evidence, task alignment and hallucination ≤ 10%.
        </p>
        <div className="metrics">
          <M l="Task success" v={pct(s.task_success_rate)} />
          <M l="Primary conclusion" v={pct(s.primary_conclusion_rate ?? null)} />
          <M l="Period correct" v={pct(s.period_correct_rate ?? null)} />
          <M l="Numbers in evidence" v={pct(s.numerical_evidence_rate ?? null)} />
          <M l="Aligned" v={pct(s.alignment_rate ?? null)} />
          <M l="Hallucination rate" v={pct(s.hallucination_rate)} />
          <M l="Scorer calibration" v={pct(s.scorer_calibration ?? null)} />
          <M l="Avg latency" v={`${s.avg_latency_s ?? "–"}s`} />
          <M l="Avg tokens" v={Math.round(s.avg_total_tokens ?? 0).toLocaleString()} />
        </div>
      </div>
      <div className="card">
        <h3>Per case</h3>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr><th>Case</th><th>Result</th><th>Conclusion</th><th>Period</th><th>Metric</th><th>Grouping</th><th>Numbers</th><th>Aligned</th><th>Tokens</th></tr>
            </thead>
            <tbody>
              {data.cases.map((c: any) => {
                const ch = c.checks ?? {};
                const cell = (k: string) => <td><span className={`badge ${ch[k] === "pass" ? "ok" : ch[k] === "fail" ? "err" : "muted"}`}>{ch[k] ?? "–"}</span></td>;
                return (
                  <tr key={c.case}>
                    <td className="mono">{c.case}{c.expect_success === false && <span className="badge muted" style={{ marginLeft: 6 }}>calibration</span>}</td>
                    <td><span className={`badge ${c.task_success ? "ok" : "err"}`}>{c.task_success ? "pass" : "fail"}</span></td>
                    {cell("primary_conclusion")}{cell("period")}{cell("metric")}{cell("grouping")}{cell("numerical_evidence")}{cell("alignment")}
                    <td className="num">{c.total_tokens?.toLocaleString()}</td>
                  </tr>
                );
              })}
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
