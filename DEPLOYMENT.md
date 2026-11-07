# Deploying DataBeacon

Status: **ready for review, not deployed.** This document records the readiness audit, the
hosting analysis under a strict $0 budget, the recommended architecture, open decisions and the
exact launch steps.

## 1. Readiness audit

| Area | Before | Now |
|---|---|---|
| Visitor isolation | Datasets, runs and PDFs were global: anyone could list, query, delete or download anyone's data | Anonymous per-browser session token (`X-DataBeacon-Session`, or `?session=` on stream/download links). Every dataset, run and report is bound to a hash of it, and other sessions get 404 |
| Abuse and cost | No rate limits | Per-IP limits on analyses (3/hour, 8/day), uploads and requests. An optional shared **demo access code** for live analyses. The neuron guard still caps Cloudflare spend |
| Resource limits | 200 MB uploads, no row/column caps, no per-visitor quota | Public defaults: 4 MB, 250k rows, 100 columns, 3 datasets per session, DuckDB memory 512 MB |
| Retention | Data kept forever | Datasets, runs and PDFs deleted after 24 h (`DATA_TTL_HOURS`) |
| Information leaks | Tracebacks returned in run timelines; `/docs` public | Internals go to the server log only; unhandled errors return a generic 500; `/docs` off in public mode |
| HTTP hardening | None | `nosniff`, `X-Frame-Options: DENY`, Referrer and Permissions policies, `Cache-Control: no-store` on the API, and a CSP for the frontend. Verified in a browser: Plotly renders with no CSP violations |
| CORS | Wildcard methods and headers | Explicit origins (`DATABEACON_CORS_ORIGINS`), methods and headers; no credentials |
| Config | Mixed | Everything via `DATABEACON_*` env vars; `DATABEACON_PUBLIC_MODE=1` switches on the stricter defaults (`.env.example`) |
| Frontend | Developer-oriented | First-visit welcome with a one-click sample, a public-demo notice (limits, privacy, retention), access-code prompt, daily AI capacity indicator, cold-start handling, size check before upload, footer, page metadata, `vercel.json` |
| LLM | Unchanged | Cloudflare Workers AI primary; rolling-24 h neuron guard; error 4006 stops at once; Groq fallback optional |

What was already sound: the read-only SQL sandbox (parser-verified SELECT only, no file or network
functions, read-only connections), credentials only from the environment or injected by a proxy,
`.env` gitignored, and the neuron budget guard.

Everything is behind `DATABEACON_PUBLIC_MODE=1`. Local development behaves as before.

## 2. Can the current backend run on Vercel?

The frontend can: it is a static Vite build, and Vercel Hobby serves it for free.

**The backend, as built, cannot run on Vercel reliably.** Vercel runs Python as serverless
functions. As of November 2026 the Hobby plan allows 300 s per invocation, 2 GB of memory, a
500 MB uncompressed Python bundle and a request body of about 4.5 MB (verify the current figures
in Vercel's limits page). Against those limits:

| Requirement of the current backend | On Vercel functions |
|---|---|
| Investigations run as **background tasks** after `POST /api/runs` returns | Work after the response has been sent is not guaranteed; the instance may be frozen |
| Live timeline over **SSE from in-memory state** of that same process | The events request can land on a different instance with no knowledge of the run |
| **DuckDB files, SQLite run store, PDFs** on local disk shared across requests | Only `/tmp`, per instance and temporary; nothing is shared between instances |
| **Neuron ledger** must be shared and durable to enforce the free tier | A per-instance `/tmp` ledger cannot enforce a global budget |
| Dependency closure is **~491 MB unpacked** (SciPy, Plotly, pandas, NumPy, OpenAI SDK) | At the 500 MB Python limit. It would need trimming (e.g. building chart JSON without Plotly) |
| Uploads up to 4 MB | Fits under the ~4.5 MB body limit |
| 60–90 s investigations | Fit within 300 s, if they run inside the request |

So "just deploy the FastAPI app to Vercel" would fail in normal use: lost runs, broken live
timelines and data disappearing between requests.

## 3. $0 hosting options

| Option | Cost risk | Effort | Verdict |
|---|---|---|---|
| **A. Vercel only, stateless backend.** Each investigation runs inside one streaming request, and the browser keeps datasets and results | None: Hobby is free with no card; Cloudflare Free plan refuses beyond 10k neurons | **Major refactor** (below) | **Recommended** for a strict $0 public deployment |
| B. Vercel frontend + free container host (current backend unchanged) | Depends on host | Small (config only) | **No suitable host found.** New Hugging Face Docker Spaces now need a paid plan; Koyeb requires a card (with a $29 pre-authorisation); Render's free tier is reported to ask for a card, with 512 MB RAM and 15-minute sleep. None meets "no card, cannot be charged" reliably |
| C. Vercel frontend + a free managed store (Upstash, Turso, Neon…) for state | Free tiers, but third-party accounts with limits that can change | Medium | Not recommended under the strict rule |
| D. Run the existing Docker image yourself (laptop, home server) behind a free tunnel | None | None | Fine for demos; not a hosted public app |

### Recommended: option A, "stateless on Vercel"

```
Browser (React on Vercel CDN)
  │ dataset file (≤4 MB) or "sample" + question
  ▼
Vercel Python function  POST /api/investigate      ← one request per investigation, ≤300 s
  ├─ ingest to /tmp DuckDB → profile → LangGraph agent → MCP servers (in-process)
  ├─ streams timeline events (NDJSON) as they happen
  └─ final event: answer, verification, alignment, charts, PDF (base64); /tmp deleted
Browser keeps datasets and past results in IndexedDB (never on the server)
Cloudflare Workers AI (Free plan) ← the hard $0 limit; optional Groq fallback
```

**What changes (needs your approval):**
1. A new `POST /api/investigate` endpoint that runs the whole investigation inside the request and
   streams events. The existing session API stays for local and Docker use.
2. The Data tab's profile and SQL console send the file with each request (≤4 MB), or the SQL
   console is disabled in the hosted build.
3. Run history moves to IndexedDB in the browser. Isolation then holds by construction, because
   nothing is stored server-side.
4. The Python bundle is trimmed below 500 MB: Plotly is replaced by a small chart-JSON builder
   (charts look the same in the browser), and tests and evaluations are excluded from the bundle.
5. A `vercel.json` adds the Python function (`maxDuration` 300).

**Trade-offs:**
- Uploads are limited to ~4 MB.
- There's a cold start of a few seconds.
- History is per browser.
- The neuron ledger and rate limits become **per function instance**. The hard spending limit is
  then Cloudflare's Free plan, which refuses (error 4006) rather than charges. A busy public demo can
  use up the day's ~10 analyses, so enable `DEMO_ACCESS_CODE` and consider a *replay mode* that shows
  recorded investigations without using the LLM.
- Effort is about 1–2 days, plus tests.

## 4. Blockers

1. **Your decision on option A** (or on accepting a card-requiring host for option B).
2. **Cloudflare account must stay on the Workers Free plan.** This is what makes overspending
   impossible. If it is ever upgraded to Workers Paid, usage beyond the allocation is billed.
   Keep `DATABEACON_ALLOW_PAID_INFERENCE` unset.
3. **Vercel account** linked to GitHub (Hobby is for non-commercial use; a portfolio fits).
4. **Branch:** Vercel builds production from `main`. Merge `claude/hopeful-darwin-6escku` into `main`
   (or set the production branch in Vercel).
5. **Public AI capacity:** ~10k neurons/day is roughly 10 analyses. Decide between an access code,
   replay mode, or accepting that the daily budget runs out.

## 5. Launch steps

### Frontend on Vercel (works today; needs a reachable backend for analyses)
1. Vercel → *Add New Project* → import `aekankpatel/databeacon`.
2. **Root Directory:** `frontend`. Framework is detected as Vite (`frontend/vercel.json` sets build and
   output, SPA routing, security headers and CSP).
3. Environment variables: `VITE_API_BASE` = backend URL (empty if same origin), and optionally
   `VITE_REPO_URL`.
4. Deploy. No card is needed on Hobby.

### Backend (after option A is approved and implemented)
1. In the same Vercel project, set these environment variables:
   - `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` (secret)
   - `DATABEACON_PUBLIC_MODE=1`, `DATABEACON_TRUST_PROXY=1`, `DATABEACON_CORS_ORIGINS=https://<project>.vercel.app`
   - `DATABEACON_DATA_DIR=/tmp/databeacon`
   - `DATABEACON_MAX_ITERATIONS=2`, `DATABEACON_MAX_STEPS_PER_ITERATION=6` (the benchmarked configuration)
   - optionally `DATABEACON_DEMO_ACCESS_CODE`
   - optionally the Groq fallback (`DATABEACON_FALLBACK_PROVIDER=groq`, `GROQ_API_KEY`)
2. Deploy, then check `/api/health`, run the sample question once and confirm `/api/usage`.

### Self-hosting today (option D, no code changes)
```bash
cp .env.example .env    # set CLOUDFLARE_*, DATABEACON_PUBLIC_MODE=1, DATABEACON_CORS_ORIGINS
docker compose up --build
```
The container honours `$PORT`, so it also runs unchanged on any container host you choose later.
