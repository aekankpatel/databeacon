import { useEffect, useState } from "react";
import { api } from "../api";
import { fmtNum } from "../format";
import type { Dataset, Profile } from "../types";
import { DataTable } from "./DataTable";

const ROLE_BADGE: Record<string, string> = {
  numeric: "info", temporal: "purple", categorical: "ok", identifier: "muted", text: "warn",
};

export function DataView({ dataset }: { dataset: Dataset }) {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sql, setSql] = useState(`SELECT * FROM "${dataset.table}" LIMIT 20`);
  const [result, setResult] = useState<{ columns: string[]; rows: unknown[][]; row_count: number; elapsed_ms: number } | null>(null);
  const [sqlError, setSqlError] = useState<string | null>(null);

  useEffect(() => {
    setProfile(null);
    setResult(null);
    setSql(`SELECT * FROM "${dataset.table}" LIMIT 20`);
    api.profile(dataset.id).then(setProfile).catch((e) => setError(e.message));
  }, [dataset.id, dataset.table]);

  async function runSql() {
    setSqlError(null);
    try {
      setResult(await api.sql(dataset.id, sql));
    } catch (e) {
      setSqlError((e as Error).message);
      setResult(null);
    }
  }

  if (error) return <div className="error-box">{error}</div>;
  if (!profile) return <div className="empty">Profiling dataset…</div>;
  const sampleCols = profile.sample_rows.length ? Object.keys(profile.sample_rows[0]) : [];

  return (
    <div className="stack">
      <div className="card">
        <h3>{dataset.name}</h3>
        <div className="metrics">
          <div className="metric"><div className="label">Rows</div><div className="value">{profile.row_count.toLocaleString()}</div></div>
          <div className="metric"><div className="label">Columns</div><div className="value">{profile.column_count}</div></div>
          <div className="metric"><div className="label">Numeric</div><div className="value">{profile.columns.filter((c) => c.role === "numeric").length}</div></div>
          <div className="metric"><div className="label">Categorical</div><div className="value">{profile.columns.filter((c) => c.role === "categorical").length}</div></div>
          <div className="metric"><div className="label">Temporal</div><div className="value">{profile.columns.filter((c) => c.role === "temporal").length}</div></div>
          <div className="metric"><div className="label">Warnings</div><div className="value">{profile.warnings.length}</div></div>
        </div>
        {profile.warnings.length > 0 && (
          <ul className="small" style={{ color: "var(--warn)" }}>{profile.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
        )}
      </div>

      <div className="card">
        <h3>Columns</h3>
        <div className="table-wrap" style={{ maxHeight: 520 }}>
          <table className="data">
            <thead>
              <tr><th>Column</th><th>Type</th><th>Role</th><th>Missing</th><th>Distinct</th><th>Summary</th></tr>
            </thead>
            <tbody>
              {profile.columns.map((c) => (
                <tr key={c.name}>
                  <td className="mono">{c.name}</td>
                  <td className="muted">{c.type}</td>
                  <td><span className={`badge ${ROLE_BADGE[c.role] ?? "muted"}`}>{c.role}</span></td>
                  <td style={{ minWidth: 110 }}>
                    <div className="small">{c.missing_pct}%</div>
                    <div className={`bar ${c.missing_pct > 20 ? "err" : c.missing_pct > 0 ? "warn" : ""}`}>
                      <span style={{ width: `${Math.min(100, c.missing_pct)}%` }} />
                    </div>
                  </td>
                  <td className="num">{c.distinct.toLocaleString()}</td>
                  <td className="small" style={{ whiteSpace: "normal" }}>
                    {c.role === "numeric" && c.stats && (
                      <>min {fmtNum(c.stats.min)} · median {fmtNum(c.stats.median)} · mean {fmtNum(c.stats.mean)} · max {fmtNum(c.stats.max)}
                        {c.outliers_iqr ? ` · ${c.outliers_iqr} IQR outliers` : ""}</>
                    )}
                    {c.role === "temporal" && c.stats && <>{String(c.stats.min)} → {String(c.stats.max)}</>}
                    {c.top_values && c.role !== "numeric" && c.top_values.map((t) => `${t.value} (${t.count})`).join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {profile.strong_correlations.length > 0 && (
          <p className="small muted">
            Strong correlations: {profile.strong_correlations.map((c) => `${c.a} ~ ${c.b} (r=${c.pearson_r})`).join(", ")}
          </p>
        )}
      </div>

      <div className="card">
        <h3>Sample rows</h3>
        <DataTable columns={sampleCols} rows={profile.sample_rows.map((r) => sampleCols.map((c) => r[c]))} />
      </div>

      <div className="card ask">
        <h3>SQL console <span className="badge muted">read-only sandbox</span></h3>
        <textarea className="mono" value={sql} onChange={(e) => setSql(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runSql(); }} />
        <div className="row" style={{ marginTop: 8 }}>
          <button className="btn primary" onClick={runSql}>Run query</button>
          {result && <span className="small muted">{result.row_count} rows · {result.elapsed_ms} ms</span>}
        </div>
        {sqlError && <div className="error-box small" style={{ marginTop: 8 }}>{sqlError}</div>}
        {result && <div style={{ marginTop: 10 }}><DataTable columns={result.columns} rows={result.rows} /></div>}
      </div>
    </div>
  );
}
