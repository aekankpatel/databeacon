import { pretty } from "../format";
import type { Evidence } from "../types";
import { DataTable } from "./DataTable";

export function EvidenceDrawer({ evidence, onClose }: { evidence: Evidence | null; onClose: () => void }) {
  if (!evidence) return null;
  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <div className="drawer" role="dialog" aria-label={`Evidence ${evidence.id}`}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2 style={{ margin: 0 }}>
            <span className="mono">{evidence.id}</span> · {evidence.tool}
          </h2>
          <button className="btn" onClick={onClose}>Close</button>
        </div>
        <div className="row small muted" style={{ marginTop: 6 }}>
          <span className={`badge ${evidence.ok ? "ok" : "err"}`}>{evidence.ok ? "success" : "error"}</span>
          <span>{evidence.server} server</span>·<span>iteration {evidence.iteration}</span>·<span>{Math.round(evidence.elapsed_ms)} ms</span>
        </div>
        {evidence.purpose && <p>{evidence.purpose}</p>}
        <p className="small">{evidence.summary}</p>
        {evidence.sql && (
          <>
            <div className="section-title">SQL</div>
            <pre className="code">{evidence.sql}</pre>
          </>
        )}
        {evidence.error && <div className="error-box">{evidence.error}</div>}
        {evidence.preview && (
          <>
            <div className="section-title">Result preview</div>
            <DataTable columns={evidence.preview.columns} rows={evidence.preview.rows} />
          </>
        )}
        <div className="section-title" style={{ marginTop: 12 }}>Arguments</div>
        <pre className="code">{pretty(evidence.args)}</pre>
        {evidence.data !== undefined && (
          <details>
            <summary className="small muted">Raw tool output</summary>
            <pre className="code">{pretty(evidence.data)}</pre>
          </details>
        )}
      </div>
    </>
  );
}
