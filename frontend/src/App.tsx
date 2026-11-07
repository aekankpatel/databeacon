import { useCallback, useEffect, useState } from "react";
import { accessCode, api, ApiError, setAccessCode } from "./api";
import { AnalyzeView } from "./components/AnalyzeView";
import { DataView } from "./components/DataView";
import { EvalsView } from "./components/EvalsView";
import { Sidebar } from "./components/Sidebar";
import { ToolsView } from "./components/ToolsView";
import { Welcome } from "./components/Welcome";
import type { Dataset, Health } from "./types";

type Tab = "analyze" | "data" | "tools" | "evals";
const TABS: [Tab, string][] = [["analyze", "Analyze"], ["data", "Data"], ["tools", "MCP Tools"], ["evals", "Evaluation"]];
const REPO_URL = import.meta.env.VITE_REPO_URL ?? "https://github.com/aekankpatel/databeacon";

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("analyze");
  const [health, setHealth] = useState<Health | null>(null);
  const [connecting, setConnecting] = useState(true);
  const [offline, setOffline] = useState(false);
  const [left, setLeft] = useState<number | null>(null);

  const refresh = useCallback(async (select?: string) => {
    try {
      const list = await api.datasets();
      setDatasets(list);
      setOffline(false);
      setSelected((cur) => select ?? (cur && list.some((d) => d.id === cur) ? cur : list[0]?.id ?? null));
    } catch (e) {
      if (e instanceof ApiError && e.status === 0) setOffline(true);
    }
  }, []);

  // A free-tier backend may be asleep: keep retrying for ~2 minutes before showing "offline".
  useEffect(() => {
    let cancelled = false;
    (async () => {
      for (let attempt = 0; attempt < 12 && !cancelled; attempt++) {
        try {
          const h = await api.health();
          if (cancelled) return;
          setHealth(h);
          setConnecting(false);
          setOffline(false);
          refresh();
          api.usage().then((u) => setLeft(u.approx_analyses_left)).catch(() => undefined);
          return;
        } catch {
          await new Promise((r) => setTimeout(r, Math.min(15000, 2000 * (attempt + 1))));
        }
      }
      if (!cancelled) {
        setConnecting(false);
        setOffline(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refresh]);

  const dataset = datasets.find((d) => d.id === selected) ?? null;
  const needsDataset = tab === "analyze" || tab === "data";

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><img src="/beacon.svg" alt="" />DataBeacon</div>
        <nav className="tabs" aria-label="Sections">
          {TABS.map(([id, label]) => (
            <button key={id} className={`tab ${tab === id ? "active" : ""}`} aria-current={tab === id ? "page" : undefined}
              onClick={() => setTab(id)}>{label}</button>
          ))}
        </nav>
        <div className="spacer" />
        {health && (
          <span className={`badge ${health.llm.configured ? "ok" : "warn"}`} title={health.llm.configured ? "LLM configured" : "LLM credentials missing"}>
            {health.llm.provider} · {health.llm.model}
          </span>
        )}
        {left !== null && health?.public_mode && (
          <span className="badge muted" title="Shared free AI budget for today">~{left} analyses left today</span>
        )}
        {offline && <span className="badge err">API offline</span>}
      </header>
      <div className="layout">
        <Sidebar datasets={datasets} selected={selected} onSelect={setSelected} onChanged={refresh}
          maxUploadMb={health?.limits?.max_upload_mb} />
        <main className="main">
          {connecting && (
            <div className="card" role="status" style={{ marginBottom: 14 }}>
              <strong>Connecting to the analysis server…</strong>
              <div className="small muted">Free-tier servers sleep when idle; the first request can take up to a minute.</div>
            </div>
          )}
          {offline && !connecting && (
            <div className="error-box small" style={{ marginBottom: 14 }}>
              The analysis server is not reachable right now. Please try again in a few minutes.
            </div>
          )}
          {health?.public_mode && <DemoNotice health={health} />}
          {health && !health.llm.configured && tab === "analyze" && (
            <div className="error-box small" style={{ marginBottom: 14, background: "var(--warn-soft)", color: "var(--warn)" }}>
              No LLM credentials found for provider “{health.llm.provider}”. Set CLOUDFLARE_ACCOUNT_ID (and
              CLOUDFLARE_API_TOKEN) or GROQ_API_KEY in the backend environment and restart.
            </div>
          )}
          {needsDataset && !dataset ? (
            <Welcome health={health} onSample={async () => {
              const ds = await api.loadSample();
              await refresh(ds.id);
              setTab("analyze");
            }} />
          ) : tab === "analyze" && dataset ? (
            <AnalyzeView dataset={dataset} />
          ) : tab === "data" && dataset ? (
            <DataView dataset={dataset} />
          ) : tab === "tools" ? (
            <ToolsView />
          ) : (
            <EvalsView />
          )}
          <footer className="footer small muted">
            DataBeacon - an autonomous, MCP-powered data analyst by Aekank Patel ·{" "}
            <a href={REPO_URL} target="_blank" rel="noreferrer">Source on GitHub</a> ·
            FastAPI, LangGraph, MCP, DuckDB, React · LLM: Cloudflare Workers AI
          </footer>
        </main>
      </div>
    </div>
  );
}

function DemoNotice({ health }: { health: Health }) {
  const [code, setCode] = useState(accessCode());
  const [saved, setSaved] = useState(Boolean(accessCode()));
  const l = health.limits;
  return (
    <div className="card small" style={{ marginBottom: 14 }}>
      <strong>Public demo.</strong>{" "}
      {l && <>
        Uploads up to {l.max_upload_mb} MB and {l.max_rows.toLocaleString()} rows; {l.max_datasets_per_session} datasets per visitor;
        {l.runs_per_hour > 0 && <> {l.runs_per_hour} analyses per hour</>}{l.runs_per_day > 0 && <>, {l.runs_per_day} per day</>}.
        {l.data_ttl_hours > 0 && <> Your datasets and results are visible only to this browser and are deleted after {l.data_ttl_hours} hours.</>}
      </>}{" "}
      Please don't upload personal or confidential data.
      {health.access_code_required && (
        <div className="row" style={{ marginTop: 8 }}>
          <label htmlFor="code">Access code for live AI analyses:</label>
          <input id="code" type="password" value={code} onChange={(e) => { setCode(e.target.value); setSaved(false); }}
            style={{ padding: "5px 8px", border: "1px solid var(--border)", borderRadius: 6, background: "var(--panel-2)" }} />
          <button className="btn small" onClick={() => { setAccessCode(code.trim()); setSaved(true); }}>Save</button>
          {saved && code && <span className="badge ok">saved for this tab</span>}
        </div>
      )}
    </div>
  );
}
