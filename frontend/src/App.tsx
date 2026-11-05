import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { AnalyzeView } from "./components/AnalyzeView";
import { DataView } from "./components/DataView";
import { EvalsView } from "./components/EvalsView";
import { Sidebar } from "./components/Sidebar";
import { ToolsView } from "./components/ToolsView";
import type { Dataset, Health } from "./types";

type Tab = "analyze" | "data" | "tools" | "evals";
const TABS: [Tab, string][] = [["analyze", "Analyze"], ["data", "Data"], ["tools", "MCP Tools"], ["evals", "Evaluation"]];

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("analyze");
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);

  const refresh = useCallback(async (select?: string) => {
    try {
      const list = await api.datasets();
      setDatasets(list);
      setOffline(false);
      setSelected((cur) => select ?? (cur && list.some((d) => d.id === cur) ? cur : list[0]?.id ?? null));
    } catch {
      setOffline(true);
    }
  }, []);

  useEffect(() => {
    refresh();
    api.health().then(setHealth).catch(() => setOffline(true));
  }, [refresh]);

  const dataset = datasets.find((d) => d.id === selected) ?? null;
  const needsDataset = tab === "analyze" || tab === "data";

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><img src="/beacon.svg" alt="" />DataBeacon</div>
        <nav className="tabs">
          {TABS.map(([id, label]) => (
            <button key={id} className={`tab ${tab === id ? "active" : ""}`} onClick={() => setTab(id)}>{label}</button>
          ))}
        </nav>
        <div className="spacer" />
        {health && (
          <span className={`badge ${health.llm.configured ? "ok" : "warn"}`} title={health.llm.configured ? "LLM configured" : "LLM credentials missing"}>
            {health.llm.provider} · {health.llm.model}
          </span>
        )}
        {offline && <span className="badge err">API offline</span>}
      </header>
      <div className="layout">
        <Sidebar datasets={datasets} selected={selected} onSelect={setSelected} onChanged={refresh} />
        <main className="main">
          {health && !health.llm.configured && tab === "analyze" && (
            <div className="error-box small" style={{ marginBottom: 14, background: "var(--warn-soft)", color: "var(--warn)" }}>
              No LLM credentials found for provider “{health.llm.provider}”. Set CLOUDFLARE_API_TOKEN and
              CLOUDFLARE_ACCOUNT_ID (or GROQ_API_KEY) in the backend environment and restart.
            </div>
          )}
          {needsDataset && !dataset ? (
            <div className="empty">
              <h2>Upload a dataset to begin</h2>
              <p>Or load the sample e-commerce dataset from the sidebar and ask why revenue declined.</p>
            </div>
          ) : tab === "analyze" && dataset ? (
            <AnalyzeView dataset={dataset} />
          ) : tab === "data" && dataset ? (
            <DataView dataset={dataset} />
          ) : tab === "tools" ? (
            <ToolsView />
          ) : (
            <EvalsView />
          )}
        </main>
      </div>
    </div>
  );
}
