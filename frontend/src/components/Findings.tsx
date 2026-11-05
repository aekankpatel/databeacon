import type { Finding, Verification } from "../types";

export function VerificationBadge({ v }: { v?: Verification }) {
  if (!v) return <span className="badge muted">unchecked</span>;
  const map = {
    verified: ["ok", "✓ verified"],
    partially_verified: ["warn", "partially verified"],
    unverified: ["err", "unverified"],
    qualitative: ["muted", "qualitative"],
  } as const;
  const [cls, label] = map[v.status];
  return (
    <span className={`badge ${cls}`} title={`${v.supported}/${v.checked_numbers} numbers matched tool outputs`}>
      {label}
      {v.checked_numbers > 0 && ` ${v.supported}/${v.checked_numbers}`}
    </span>
  );
}

export function Findings({ findings, onEvidence }: { findings: Finding[]; onEvidence: (id: string) => void }) {
  if (!findings.length) return <div className="muted">No findings.</div>;
  return (
    <div>
      {findings.map((f, i) => (
        <div key={i} className="finding">
          <div className="head">
            <span className="title">{i + 1}. {f.title}</span>
            <VerificationBadge v={f.verification} />
          </div>
          <div className="statement">{f.statement}</div>
          <div className="row small">
            <span className="muted">Evidence:</span>
            {f.evidence_ids.length === 0 && <span className="muted">none cited</span>}
            {f.evidence_ids.map((id) => (
              <span key={id} className="ev-chip" onClick={() => onEvidence(id)}>{id}</span>
            ))}
            <span className="muted">· confidence {f.confidence}</span>
          </div>
          {!!f.verification?.unsupported_values.length && (
            <div className="unsupported">Not found in evidence: {f.verification.unsupported_values.join(", ")}</div>
          )}
        </div>
      ))}
    </div>
  );
}
