import type { Catalog, Dataset, Health, Profile, Run, RunSummary } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => req<Health>("/api/health"),
  datasets: () => req<Dataset[]>("/api/datasets"),
  upload: (file: File, name?: string) => {
    const fd = new FormData();
    fd.append("file", file);
    if (name) fd.append("name", name);
    return req<Dataset>("/api/datasets", { method: "POST", body: fd });
  },
  loadSample: () => req<Dataset>("/api/datasets/sample", { method: "POST" }),
  deleteDataset: (id: string) => req<void>(`/api/datasets/${id}`, { method: "DELETE" }),
  profile: (id: string) => req<Profile>(`/api/datasets/${id}/profile`),
  sql: (id: string, sql: string) =>
    req<{ columns: string[]; rows: unknown[][]; row_count: number; truncated: boolean; elapsed_ms: number }>(
      `/api/datasets/${id}/sql`,
      json({ sql }),
    ),
  catalog: () => req<Catalog>("/api/mcp/catalog"),
  startRun: (dataset_id: string, question: string, clarification?: string) =>
    req<{ run_id: string }>("/api/runs", json({ dataset_id, question, clarification })),
  runs: (dataset_id?: string) => req<RunSummary[]>(`/api/runs${dataset_id ? `?dataset_id=${dataset_id}` : ""}`),
  run: (id: string) => req<Run>(`/api/runs/${id}`),
  reportUrl: (id: string) => `${BASE}/api/runs/${id}/report.pdf`,
  eventsUrl: (id: string) => `${BASE}/api/runs/${id}/events`,
  evals: () => req<Record<string, any>>("/api/evals/latest"),
};
