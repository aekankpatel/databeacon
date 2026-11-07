# Deploying DataBeacon

Status: **implemented and tested locally, not deployed.** Architecture: Vercel Hobby (frontend +
Python function) + Cloudflare D1 (shared state) + Cloudflare Workers AI (LLM). Everything here is on
free plans that refuse rather than bill. This document records the readiness audit, the platform
findings the design relies on, how serverless execution works, what is still unverified and the
exact setup steps.

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

## 2. Architecture

```
Browser (React, served from Vercel's CDN)
  │  POST /api/runs  → the response *is* the live event stream (text/event-stream)
  ▼
Vercel Python function api/index.py (FastAPI, ≤300 s, 2 GB)
  ├─ LangGraph agent → MCP servers (in-process) → DuckDB SQL on a /tmp copy of the dataset
  ├─ evidence verification, task alignment, charts (Plotly JSON), PDF report
  └─ persists every timeline event, heartbeat, result and PDF as it goes
        │  HTTPS (D1 query API, atomic batches)
        ▼
Cloudflare D1 (Free)  datasets (zstd Parquet chunks) · runs · run_events · blobs
                      neuron ledger + reservations · rate limits · provider blocks
Cloudflare Workers AI (Free, 10k neurons/day)  ← optional Groq fallback
```

- **One Vercel project** serves both the static frontend (`frontend/dist`) and the function
  (`/api/*` is rewritten to `api/index.py`). Same origin, so no CORS and no `VITE_API_BASE`.
- **Nothing depends on instance-local state.** `/tmp` is only a cache: a dataset missing on a cold
  instance is rebuilt from D1 and checked against its SHA-256. Run status, events, results, PDFs,
  rate limits and the neuron ledger all live in D1, so every instance sees the same numbers.
- **Local development is unchanged.** Without `DATABEACON_STORAGE=d1` the same tables live in SQLite
  files under `DATABEACON_DATA_DIR`, and runs execute as background tasks (`DATABEACON_EXECUTION=background`).

### Atomic operations

D1's HTTP API has no `BEGIN`/`COMMIT`, so each decision is a single conditional statement:
`INSERT … SELECT … WHERE <limit not reached>`, judged by `meta.changes`.
- Neuron reservation: inserted only if spent + outstanding reservations + this estimate ≤ the daily
  limit. Recording a call (usage row + reservation release) is one atomic `batch`.
- Rate limits (runs per hour/day, uploads): inserted only if the window count is below the limit.
- If D1 cannot be reached, the budget **fails closed**: no LLM call is made.

### Serverless execution (inline mode)

Vercel gives no guarantee for work after a response ends, so on Vercel (`VERCEL` is set) runs are
**inline**: the investigation runs inside the `POST /api/runs` request and streams its events.
- The function keeps working if the browser disconnects (the stream keeps being consumed), and
  saves events every ~0.75 s plus a heartbeat every 15 s.
- The agent has a deadline (`RUN_DEADLINE_S`, 270 s). With 75 s left it stops investigating and
  writes the answer, verification and PDF from what it has (`deadline_reached` in the timeline).
  A hard limit (deadline + `RUN_DEADLINE_GRACE_S`) marks the run `interrupted`.

### Reconnecting vs. resuming

These are different and the UI says which one happened:
- **Reconnected to saved progress.** The page was reloaded, or the request reached another instance,
  while the run is still executing elsewhere. The browser reads the saved events from D1 and keeps
  following until the run finishes. Nothing is lost or repeated.
- **Interrupted.** The executing instance stopped (timeout, crash, platform recycle): no heartbeat for
  `RUN_STALE_S` (120 s). The run is marked `interrupted` with a reason and `resumable: false`; the
  saved timeline and partial evidence stay visible. An interrupted run is **not resumed** (LLM state
  and the investigation cannot be safely restored mid-step). "Start it again" begins a new run that
  records `rerun_of` and spends neurons again.

### Replay mode

`/api/replays` serves four recorded investigations from real Cloudflare benchmark runs
(`backend/replays/`): full timeline, answer, charts and PDF, played back in the "Recorded" tab with
no LLM calls and no neurons. `DATABEACON_REPLAY_ONLY=1` turns the public demo into replay-only (live
runs return 403), a safe fallback once the daily budget is spent or if you want zero AI usage.

## 3. Platform findings this design relies on (checked November 2026)

**Vercel Hobby** (no card): functions up to 300 s (streaming time counts), 2 GB / 1 vCPU, request
body ≈ 4.5 MB. Python runtime `@vercel/python` 31.x: bundles ≤ 225 MB are deployed as is; 225–500 MB
are installed at cold start with `uv` from `uv.lock`; > 500 MB fails.
- Validated with `vercel build` (CLI 65.12.1): **417.6 MB** total dependencies; SciPy, DuckDB,
  cryptography and jsonschema bundled (≈ 225 MB), the rest installed at cold start. Plotly is kept.
- A local build assumes glibc 2.17 wheels and would force-bundle NumPy/pandas/DuckDB/Pillow; the
  Vercel build image (Amazon Linux 2023, glibc 2.34) accepts their manylinux_2_27/2_28 wheels. Verified
  by simulating that image locally; **confirm on the first real deploy** (section 5).
- `excludeFiles` keeps the frontend, tests, docs and eval runs out of the function.
- Measured memory: ~155 MB after startup, ~315 MB peak during an analysis.

**Cloudflare D1 Free** (no card): 5 M rows read and 100 k rows written per day, 500 MB per database,
2 MB per row, 100 KB per statement, 30 s per API request. An analysis writes on the order of 100–300
rows, far below the daily cap; retention (24 h) keeps the database small.

**Dataset storage**: DuckDB → zstd Parquet → base64 chunks (512 KB, halved automatically if D1
rejects the size) → DuckDB, with SHA-256 checked on rebuild. `python evals/validate_dataset_codec.py`
round-trips realistic data byte-identically (schema, rows and an order-independent content hash):

| Dataset | Rows × cols | CSV | Parquet | Chunks |
|---|---|---|---|---|
| e-commerce sample | 6,000 × 12 | 1.53 MB | 0.27 MB | 1 |
| mixed types (tz timestamps, Unicode, NULLs, leading zeros, int64 extremes) | 30,000 × 11 | 8.82 MB | 0.87 MB | 2 |
| wide | 20,000 × 100 | 38.9 MB | 15.1 MB | 29 |
| telemetry at the 4 MB cap | 120,000 × 4 | 4.19 MB | 0.79 MB | 2 |
| nested JSON (structs, lists) | 8,000 × 4 | — | small | 1 |

**Excluded because they need a card**: Cloudflare R2, Workers Paid, Vercel Pro. Vercel Blob was
avoided. **Keep the Cloudflare account on the Workers Free plan**: it refuses (error 4006) instead of
billing beyond the allocation. Leave `DATABEACON_ALLOW_PAID_INFERENCE` unset.

## 4. Tests

- `pytest` (115 tests), including storage isolation per session on D1, concurrent neuron reservations
  that never exceed the limit, concurrent rate limits, dataset rebuild on a cold instance and checksum
  failure, inline streaming, client disconnect, reconnect from another instance, stale-executor
  interruption, deadline and hard time limit, and replay mode (no LLM calls, blocked live runs).
  D1 is exercised through an HTTP-level emulator (`tests/d1_emulator.py`) that applies the real
  request/response format, batch atomicity, row size limits and injected failures.
- `python -m evals.run --mode scripted` (offline, scripted LLM): all cases pass.
- `scripts/e2e_serverless.sh`: a browser test with two backend instances sharing one store: replay
  playback, a live inline run, reconnecting from the second instance, and an interrupted run after
  killing the first instance, then "Start it again".

### Not verified yet (needs the real platforms)

- A real Vercel deploy: cold-start time with the runtime `uv` install (expect tens of seconds on the
  first request after idle), and whether the function keeps running after the browser disconnects
  (designed for it; if not, the run is reported interrupted, never silently lost).
- Real D1: the current Cloudflare token has no D1 permission, so D1 was tested with the emulator.
  Latency per query and the exact parameter-size limit are to be checked (chunks shrink automatically).
- Cloudflare API rate limits under concurrent public use.

## 5. Setup (do not deploy until you are ready)

### Cloudflare (dash.cloudflare.com, Free plan)
1. **Create the database:** *Storage & Databases → D1 → Create* (name e.g. `databeacon`). Copy its
   **Database ID**. No schema step is needed: tables are created on first use.
2. **API token:** *My Profile → API Tokens → Create Token → Custom token*, account-scoped to your
   account, with **Account → Workers AI → Read** and **Account → D1 → Edit**. Either use one token for
   both (`CLOUDFLARE_API_TOKEN`) or a separate D1 token (`DATABEACON_D1_API_TOKEN`).
3. Note your **Account ID** (dashboard sidebar / Workers & Pages overview).
4. Confirm *Workers & Pages → Plans* shows **Workers Free**.

### Vercel (vercel.com, Hobby)
1. *Add New → Project* → import `aekankpatel/databeacon`. **Root Directory: the repository root**
   (not `frontend`). `vercel.json` sets the build, output, function (`maxDuration` 300), rewrites and
   security headers. Leave the framework preset as detected ("Other" is fine).
2. **Environment variables** (Production):

   | Variable | Value |
   |---|---|
   | `CLOUDFLARE_ACCOUNT_ID` | your account ID |
   | `CLOUDFLARE_API_TOKEN` | the token (mark *Sensitive*) |
   | `DATABEACON_STORAGE` | `d1` |
   | `DATABEACON_D1_DATABASE_ID` | the database ID |
   | `DATABEACON_PUBLIC_MODE` | `1` |
   | `DATABEACON_TRUST_PROXY` | `1` |
   | `DATABEACON_CORS_ORIGINS` | `https://<project>.vercel.app` |
   | `DATABEACON_MAX_ITERATIONS` | `2` |
   | `DATABEACON_MAX_STEPS_PER_ITERATION` | `6` |
   | optional `DATABEACON_DEMO_ACCESS_CODE` | a shared code for live analyses |
   | optional `DATABEACON_REPLAY_ONLY` | `1` for recorded investigations only |
   | optional `DATABEACON_D1_API_TOKEN` | separate D1 token |
   | optional `DATABEACON_FALLBACK_PROVIDER` / `GROQ_API_KEY` | Groq fallback |

   `DATABEACON_EXECUTION=inline` and `DATABEACON_DATA_DIR=/tmp/databeacon` are automatic on Vercel.
3. **Production branch:** merge `claude/hopeful-darwin-6escku` into `main`, or set the production
   branch under *Settings → Git*.
4. After the first deploy: check the build log for "Installing required dependencies" (runtime
   install) and no size error, then `/api/health` (`storage: d1`, `execution: inline`), one sample
   analysis, `/api/usage`, a page reload mid-run (should say *Reconnected to saved progress*) and the
   Recorded tab.

### Self-hosting (Docker, no Vercel or D1)
```bash
cp .env.example .env    # set CLOUDFLARE_*, DATABEACON_PUBLIC_MODE=1, DATABEACON_CORS_ORIGINS
docker compose up --build
```
The container honours `$PORT`, uses local SQLite storage and background execution.
