import { useState } from "react";
import { fmtMs, pretty } from "../format";
import type { TimelineEvent } from "../types";

const COLORS: Record<string, string> = {
  node: "var(--accent)",
  llm: "var(--purple)",
  tool: "var(--ok)",
  decision: "var(--warn)",
  guardrail: "var(--warn)",
  discovery: "var(--muted)",
  error: "var(--err)",
};

export function Timeline({ events, onEvidence }: { events: TimelineEvent[]; onEvidence: (id: string) => void }) {
  const [open, setOpen] = useState<string | null>(null);
  if (!events.length) return <div className="empty small">The agent's decisions, tool calls and SQL appear here live.</div>;
  return (
    <div className="timeline">
      {events.map((e) => {
        const color = e.status === "error" ? "var(--err)" : COLORS[e.type] ?? "var(--muted)";
        const isOpen = open === e.id;
        return (
          <div key={e.id} className={`tl-item ${isOpen ? "open" : ""}`} onClick={() => setOpen(isOpen ? null : e.id)}>
            <span className={`tl-dot ${e.status === "running" ? "running-dot" : ""}`} style={{ background: color }} />
            <div style={{ minWidth: 0 }}>
              <div className="tl-name">
                {e.name}
                {e.status === "error" && <span className="badge err" style={{ marginLeft: 6 }}>error</span>}
                {e.status === "warning" && <span className="badge warn" style={{ marginLeft: 6 }}>warning</span>}
                {e.origin === "fallback" && (
                  <span className="badge purple" style={{ marginLeft: 6 }} title="Added automatically because the agent created no chart">
                    auto fallback
                  </span>
                )}
                {e.origin === "system" && <span className="badge muted" style={{ marginLeft: 6 }}>pipeline</span>}
              </div>
              <div className="tl-type">
                {e.type}
                {e.server ? ` · ${e.server} server` : ""}
                {e.tokens ? ` · ${e.tokens.input_tokens + e.tokens.output_tokens} tok` : ""}
                {e.evidence_id && (
                  <>
                    {" · "}
                    <span
                      className="ev-chip"
                      onClick={(ev) => {
                        ev.stopPropagation();
                        onEvidence(e.evidence_id!);
                      }}
                    >
                      {e.evidence_id}
                    </span>
                  </>
                )}
              </div>
            </div>
            <span className="tl-dur">{e.status === "running" ? "…" : fmtMs(e.duration_ms)}</span>
            {isOpen && (
              <div className="tl-detail" onClick={(ev) => ev.stopPropagation()}>
                {e.sql && <pre className="code">{e.sql}</pre>}
                {e.input !== undefined && !e.sql && (
                  <>
                    <div className="small muted">input</div>
                    <pre className="code">{pretty(e.input)}</pre>
                  </>
                )}
                {e.error && <div className="error-box small">{e.error}</div>}
                {e.output !== undefined && (
                  <>
                    <div className="small muted">output</div>
                    <pre className="code">{pretty(e.output)}</pre>
                  </>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
