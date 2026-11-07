import type { Catalog, Dataset, Health, Profile, Replay, ReplaySummary, Run, RunSummary } from "./types";

const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");
const SESSION_KEY = "databeacon.session";
const CODE_KEY = "databeacon.accessCode";
let memorySession: string | null = null;

/** Anonymous per-browser session: the backend scopes datasets, runs and reports to it. */
export function sessionToken(): string {
  try {
    const existing = localStorage.getItem(SESSION_KEY);
    if (existing && /^[A-Za-z0-9_-]{22,128}$/.test(existing)) return existing;
  } catch {
    /* storage blocked: fall back to memory */
  }
  if (!memorySession) {
    const bytes = new Uint8Array(24);
    crypto.getRandomValues(bytes);
    memorySession = btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }
  try {
    localStorage.setItem(SESSION_KEY, memorySession);
  } catch {
    /* ignore */
  }
  return memorySession;
}

export function resetSession(): void {
  memorySession = null;
  try {
    localStorage.removeItem(SESSION_KEY);
  } catch {
    /* ignore */
  }
}

export function accessCode(): string {
  try {
    return sessionStorage.getItem(CODE_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setAccessCode(code: string): void {
  try {
    sessionStorage.setItem(CODE_KEY, code);
  } catch {
    /* ignore */
  }
}

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

function headers(extra?: HeadersInit): Headers {
  const h = new Headers(extra);
  h.set("X-DataBeacon-Session", sessionToken());
  const code = accessCode();
  if (code) h.set("X-DataBeacon-Access-Code", code);
  return h;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers: headers(init?.headers) });
  } catch {
    throw new ApiError("Can't reach the DataBeacon API. It may be starting up - try again in a moment.", 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* ignore */
    }
    if (res.status === 413 && !detail) detail = "The file is too large for this demo.";
    throw new ApiError(detail, res.status);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export type StreamMessage = Record<string, any> & { kind: string };

/** Read a text/event-stream response body, calling onMessage for every `data:` line. */
export async function readEventStream(res: Response, onMessage: (m: StreamMessage) => void, signal?: AbortSignal) {
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    if (signal?.aborted) {
      await reader.cancel();
      return;
    }
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) {
          try {
            onMessage(JSON.parse(line.slice(6)));
          } catch {
            /* ignore malformed line */
          }
        }
      }
    }
  }
}

async function streamRequest(path: string, init: RequestInit, onMessage: (m: StreamMessage) => void,
                             signal?: AbortSignal): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers: headers(init.headers), signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("Can't reach the DataBeacon API. It may be starting up - try again in a moment.", 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* ignore */
    }
    throw new ApiError(detail, res.status);
  }
  if ((res.headers.get("content-type") ?? "").includes("text/event-stream")) {
    await readEventStream(res, onMessage, signal);
  }
  return res;
}

const withSession = (path: string) => `${BASE}${path}?session=${encodeURIComponent(sessionToken())}`;

export const api = {
  health: () => req<Health>("/api/health"),
  usage: () => req<{ remaining_neurons: number; approx_analyses_left: number; cloudflare_blocked: string | null }>("/api/usage"),
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
  /**
   * Start an investigation. On a long-lived server this returns {run_id} at once and progress is
   * followed with followRun(). On serverless (inline execution) the response *is* the event stream:
   * messages arrive through onMessage until the run is done.
   */
  startRun: async (dataset_id: string, question: string, onMessage: (m: StreamMessage) => void,
                   opts: { clarification?: string; rerun_of?: string; signal?: AbortSignal } = {}) => {
    const res = await streamRequest("/api/runs", json({ dataset_id, question, clarification: opts.clarification,
      rerun_of: opts.rerun_of }), onMessage, opts.signal);
    if ((res.headers.get("content-type") ?? "").includes("application/json")) {
      return { inline: false, ...(await res.json()) } as { inline: boolean; run_id: string };
    }
    return { inline: true, run_id: "" };
  },
  /** Follow a run's events (live in the executing process, or saved progress from anywhere else). */
  followRun: (id: string, onMessage: (m: StreamMessage) => void, signal?: AbortSignal) =>
    streamRequest(`/api/runs/${id}/events`, {}, onMessage, signal),
  runs: (dataset_id?: string) => req<RunSummary[]>(`/api/runs${dataset_id ? `?dataset_id=${dataset_id}` : ""}`),
  run: (id: string) => req<Run>(`/api/runs/${id}`),
  // EventSource and plain links cannot send headers, so the session goes in the query string
  reportUrl: (id: string) => withSession(`/api/runs/${id}/report.pdf`),
  replays: () => req<ReplaySummary[]>("/api/replays"),
  replay: (id: string) => req<Replay>(`/api/replays/${id}`),
  replayReportUrl: (id: string) => `${BASE}/api/replays/${id}/report.pdf`,
  evals: () => req<Record<string, any>>("/api/evals/latest"),
};
