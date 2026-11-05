import { useEffect, useState } from "react";
import { api } from "../api";
import { pretty } from "../format";
import type { Catalog } from "../types";

export function ToolsView() {
  const [cat, setCat] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.catalog().then(setCat).catch((e) => setError(e.message));
  }, []);
  if (error) return <div className="error-box">{error}</div>;
  if (!cat) return <div className="empty">Discovering MCP servers…</div>;
  const servers = [...new Set(cat.tools.map((t) => t.server))];
  return (
    <div className="stack">
      <div className="card">
        <h3>MCP tool discovery</h3>
        <p className="muted small">
          The agent connects to each MCP server at the start of a run, lists its tools, resources and prompts, and
          exposes the tools to the LLM. Nothing is hard-coded in the agent loop.
        </p>
        <div className="metrics">
          <div className="metric"><div className="label">Servers</div><div className="value">{servers.length}</div></div>
          <div className="metric"><div className="label">Tools</div><div className="value">{cat.tools.length}</div></div>
          <div className="metric"><div className="label">Resources</div><div className="value">{cat.resources.length}</div></div>
          <div className="metric"><div className="label">Prompts</div><div className="value">{cat.prompts.length}</div></div>
        </div>
      </div>
      {servers.map((s) => (
        <div className="card" key={s}>
          <h3>databeacon-{s} <span className="badge info">MCP server</span></h3>
          {cat.tools.filter((t) => t.server === s).map((t) => (
            <details key={t.name} style={{ padding: "6px 0", borderBottom: "1px dashed var(--border)" }}>
              <summary><span className="mono">{t.name}</span> <span className="small muted">{t.description}</span></summary>
              <pre className="code">{pretty(t.input_schema)}</pre>
            </details>
          ))}
          {cat.resources.filter((r) => r.server === s).length > 0 && (
            <p className="small"><strong>Resources:</strong>{" "}
              {cat.resources.filter((r) => r.server === s).map((r) => <span key={r.uri} className="mono" style={{ marginRight: 10 }}>{r.uri}</span>)}
            </p>
          )}
          {cat.prompts.filter((p) => p.server === s).length > 0 && (
            <p className="small"><strong>Prompts:</strong>{" "}
              {cat.prompts.filter((p) => p.server === s).map((p) => <span key={p.name} className="mono" style={{ marginRight: 10 }}>{p.name}</span>)}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}
