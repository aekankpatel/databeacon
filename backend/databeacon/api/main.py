"""FastAPI backend for DataBeacon."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .. import __version__
from ..config import get_settings
from ..datasets import DatasetError
from ..llm import provider_info
from ..mcp_client import MCPToolHub
from ..observability import setup_tracing
from ..profiling import profile_table, schema_summary
from ..runs import RunManager
from ..sql_guard import SQLValidationError, execute_readonly
from .security import (
    ACCESS_CODE_HEADER,
    FRONTEND_CSP,
    SECURITY_HEADERS,
    SESSION_HEADER,
    RateLimiter,
    SharedRateLimiter,
    check_access_code,
    client_ip,
    session_owner,
)

SAMPLE_CSV = Path(__file__).resolve().parents[2] / "sample_data" / "ecommerce_orders.csv"
SAMPLE_NAME = "E-commerce orders (sample)"
EVALS_DIR = Path(__file__).resolve().parents[2] / "evals" / "results"
CLEANUP_EVERY_S = 600
log = logging.getLogger("databeacon.api")


class RunRequest(BaseModel):
    dataset_id: str
    question: str = Field(min_length=5, max_length=2000)
    clarification: str | None = Field(default=None, max_length=500,
                                      description="Answer to a clarification request, e.g. 'Q4 2025'")

    @field_validator("question")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", v).strip()
        if len(v) < 5:
            raise ValueError("question is too short")
        return v


class SQLRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=20000)


def create_app(manager: RunManager | None = None) -> FastAPI:
    settings = get_settings()
    setup_tracing()
    from ..storage.datasets import make_registry

    registry = manager.registry if manager else make_registry()
    limiter = RateLimiter()  # per-instance flood guard (requests/minute): no shared writes per request
    from ..storage import open_store

    shared_limiter = SharedRateLimiter(open_store("app"))  # analyses and uploads: shared and atomic
    last_cleanup = {"t": 0.0}

    def cleanup(mgr: RunManager | None) -> None:
        """Retention: delete datasets, runs and reports older than DATA_TTL_HOURS."""
        if settings.data_ttl_hours <= 0:
            return
        n = registry.purge_older_than(settings.data_ttl_hours)
        ids = mgr.store.purge_older_than(settings.data_ttl_hours) if mgr else []
        for rid in ids:
            (settings.reports_dir / f"{rid}.pdf").unlink(missing_ok=True)
        if n or ids:
            log.info("retention: removed %d datasets, %d runs", n, len(ids))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runs = manager or RunManager(registry, settings)
        cleanup(app.state.runs)
        yield
        for t in app.state.runs.tasks.values():
            t.cancel()

    app = FastAPI(title="DataBeacon API", version=__version__, lifespan=lifespan,
                  docs_url="/docs" if settings.expose_docs else None,
                  redoc_url="/redoc" if settings.expose_docs else None,
                  openapi_url="/openapi.json" if settings.expose_docs else None)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False,
                       allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                       allow_headers=["Content-Type", SESSION_HEADER, ACCESS_CODE_HEADER],
                       expose_headers=["Retry-After"], max_age=600)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.url.path.startswith("/api/") and request.method != "OPTIONS":
            try:
                limiter.hit(f"req:{client_ip(request, settings)}", settings.requests_per_minute, 60, "requests")
            except HTTPException as exc:
                return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)
            if time.monotonic() - last_cleanup["t"] > CLEANUP_EVERY_S:
                last_cleanup["t"] = time.monotonic()
                await asyncio.to_thread(cleanup, getattr(request.app.state, "runs", None))
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        elif settings.public_mode or os.environ.get("DATABEACON_FRONTEND_CSP") == "1":
            response.headers.setdefault("Content-Security-Policy", FRONTEND_CSP)
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):  # never leak internals to clients
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse({"detail": "Internal error."}, status_code=500)

    def runs(request: Request) -> RunManager:
        return request.app.state.runs

    def owner_of(request: Request) -> str:
        return session_owner(request, settings)

    def ip_of(request: Request) -> str:
        return client_ip(request, settings)

    def own_run(request: Request, run_id: str, owner: str) -> dict[str, Any]:
        row = runs(request).store.get(run_id) if re.fullmatch(r"[0-9a-f]{12}", run_id) else None
        if row is None or row.get("owner", "local") != owner:
            raise HTTPException(404, "run not found")
        return row

    # ---------------------------------------------------------------- health
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "llm": provider_info(),
                "mcp_transport": settings.mcp_transport, "public_mode": settings.public_mode,
                "access_code_required": bool(settings.demo_access_code),
                "limits": {"max_upload_mb": settings.max_upload_mb, "max_rows": settings.max_rows,
                           "max_columns": settings.max_columns,
                           "max_datasets_per_session": settings.max_datasets_per_session,
                           "runs_per_hour": settings.runs_per_hour, "runs_per_day": settings.runs_per_day,
                           "data_ttl_hours": settings.data_ttl_hours}}

    @app.get("/api/usage")
    async def llm_usage() -> dict[str, Any]:
        """LLM usage in the guard's window and the Cloudflare free-tier neuron budget."""
        from ..budget import NeuronBudget

        b = NeuronBudget.default()
        out = b.summary()
        out["remaining_neurons"] = round(b.remaining(), 1)
        out["approx_analyses_left"] = int(b.remaining() // 1000)  # ~700-1,000 neurons per analysis measured
        if settings.public_mode:
            out.pop("by_model", None)
        return out

    # -------------------------------------------------------------- datasets
    @app.get("/api/datasets")
    async def list_datasets(owner: str = Depends(owner_of)) -> list[dict[str, Any]]:
        return [_public_ds(d) for d in registry.list(owner)]

    def _quota(owner: str) -> None:
        if len(registry.list(owner)) >= settings.max_datasets_per_session:
            raise HTTPException(409, f"This session already has {settings.max_datasets_per_session} datasets. "
                                     "Delete one to add another.")

    def _ingest(path: Path, name: str | None, filename: str, owner: str):
        return registry.ingest_file(path, name=name, original_filename=filename, owner=owner,
                                    max_rows=settings.max_rows, max_columns=settings.max_columns,
                                    memory_limit=settings.duckdb_memory_limit)

    @app.post("/api/datasets", status_code=201)
    async def upload_dataset(file: UploadFile = File(...), name: str | None = Form(None),
                             owner: str = Depends(owner_of), ip: str = Depends(ip_of)) -> dict[str, Any]:
        shared_limiter.hit(f"upload:{ip}", settings.uploads_per_hour, 3600, "uploads")
        _quota(owner)
        filename = Path(file.filename or "upload.csv").name[:120]
        limit = int(settings.max_upload_mb * 1024 * 1024)
        tmpdir = Path(tempfile.mkdtemp(prefix="upload_"))
        tmp = tmpdir / re.sub(r"[^\w.\-]", "_", filename)
        size = 0
        try:
            with tmp.open("wb") as fh:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"File exceeds the {settings.max_upload_mb:g} MB limit")
                    fh.write(chunk)
            ds = await asyncio.to_thread(_ingest, tmp, (name or None) and name[:120], filename, owner)
        except DatasetError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return _public_ds(ds)

    @app.post("/api/datasets/sample", status_code=201)
    async def load_sample(owner: str = Depends(owner_of), ip: str = Depends(ip_of)) -> dict[str, Any]:
        existing = next((d for d in registry.list(owner) if d.name == SAMPLE_NAME), None)
        if existing:  # one sample copy per session
            return _public_ds(existing)
        shared_limiter.hit(f"upload:{ip}", settings.uploads_per_hour, 3600, "uploads")
        _quota(owner)
        if not SAMPLE_CSV.exists():
            from sample_data.generate_ecommerce import generate

            await asyncio.to_thread(lambda: generate().to_csv(SAMPLE_CSV, index=False))
        ds = await asyncio.to_thread(_ingest, SAMPLE_CSV, SAMPLE_NAME, SAMPLE_CSV.name, owner)
        return _public_ds(ds)

    @app.get("/api/datasets/{dataset_id}")
    async def get_dataset(dataset_id: str, owner: str = Depends(owner_of)) -> dict[str, Any]:
        try:
            return _public_ds(registry.get(dataset_id, owner))
        except DatasetError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.delete("/api/datasets/{dataset_id}", status_code=204)
    async def delete_dataset(dataset_id: str, owner: str = Depends(owner_of)) -> None:
        try:
            registry.delete(dataset_id, owner)
        except DatasetError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/datasets/{dataset_id}/profile")
    async def dataset_profile(dataset_id: str, owner: str = Depends(owner_of)) -> dict[str, Any]:
        try:
            ds = registry.get(dataset_id, owner)
            prof = await asyncio.to_thread(profile_table, registry.db_path(dataset_id), ds.table)
        except DatasetError as exc:
            raise HTTPException(404, str(exc)) from exc
        prof["summary"] = schema_summary(prof)
        return prof

    @app.post("/api/datasets/{dataset_id}/sql")
    async def run_sql(dataset_id: str, body: SQLRequest, owner: str = Depends(owner_of)) -> dict[str, Any]:
        """Read-only SQL console (same sandbox the agent uses)."""
        try:
            registry.get(dataset_id, owner)
            res = await asyncio.to_thread(execute_readonly, registry.db_path(dataset_id), body.sql,
                                          max_rows=1000, timeout_s=settings.sql_timeout_s)
        except DatasetError as exc:
            raise HTTPException(404, str(exc)) from exc
        except SQLValidationError as exc:
            raise HTTPException(400, str(exc)) from exc
        return res.to_dict()

    # ------------------------------------------------------------------- MCP
    @app.get("/api/mcp/catalog")
    async def mcp_catalog() -> dict[str, Any]:
        async with MCPToolHub(registry) as hub:
            return hub.catalog()

    # ------------------------------------------------------------------ runs
    @app.post("/api/runs", status_code=202)
    async def start_run(body: RunRequest, request: Request, owner: str = Depends(owner_of),
                        ip: str = Depends(ip_of)) -> dict[str, Any]:
        check_access_code(request, settings)
        try:
            registry.get(body.dataset_id, owner)
        except DatasetError as exc:
            raise HTTPException(404, str(exc)) from exc
        shared_limiter.hit(f"runs-h:{ip}", settings.runs_per_hour, 3600, "analyses")
        shared_limiter.hit(f"runs-d:{ip}", settings.runs_per_day, 86400, "analyses")
        run_id = runs(request).start(body.dataset_id, body.question, body.clarification, owner=owner)
        return {"run_id": run_id, "status": "running"}

    @app.get("/api/runs")
    async def list_runs(request: Request, dataset_id: str | None = None,
                        owner: str = Depends(owner_of)) -> list[dict[str, Any]]:
        return runs(request).store.list(dataset_id, owner=owner)

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str, request: Request, owner: str = Depends(owner_of)) -> dict[str, Any]:
        own_run(request, run_id, owner)
        row = runs(request).get(run_id)
        row.pop("owner", None)
        return row

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request, owner: str = Depends(owner_of)) -> StreamingResponse:
        row = own_run(request, run_id, owner)
        mgr = runs(request)
        rec = mgr.recorders.get(run_id)

        async def stream():
            if rec is None:  # run from a previous server process: nothing live to stream
                yield f"data: {json.dumps({'kind': 'done', 'status': row['status']})}\n\n"
                return
            q = rec.subscribe()  # replays every event recorded so far
            try:
                task = mgr.tasks.get(run_id)
                if task is None or task.done():
                    while not q.empty():
                        yield f"data: {json.dumps(q.get_nowait(), default=str)}\n\n"
                    yield f"data: {json.dumps({'kind': 'done', 'status': mgr.store.get(run_id)['status']})}\n\n"
                    return
                while not await request.is_disconnected():
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=15)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {json.dumps(msg, default=str)}\n\n"
                    if msg.get("kind") == "done":
                        break
            finally:
                rec.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/runs/{run_id}/report.pdf")
    async def run_report(run_id: str, request: Request, owner: str = Depends(owner_of)) -> FileResponse:
        if not re.fullmatch(r"[0-9a-f]{12}", run_id):
            raise HTTPException(400, "invalid run id")
        own_run(request, run_id, owner)
        path = settings.reports_dir / f"{run_id}.pdf"
        if not path.exists():
            raise HTTPException(404, "report not available")
        return FileResponse(path, media_type="application/pdf", filename=f"databeacon-report-{run_id}.pdf")

    # ----------------------------------------------------------------- evals
    @app.get("/api/evals/latest")
    async def latest_eval() -> dict[str, Any]:
        """Latest live benchmark if one exists, otherwise the latest offline (scripted) evaluation."""
        for name in ("latest.json", "latest_scripted.json"):
            path = EVALS_DIR / name
            if path.exists():
                return json.loads(path.read_text())
        raise HTTPException(404, "no evaluation results yet - run `python -m evals.run --mode scripted`")

    # ------------------------------------------------------- static frontend
    dist = Path(os.environ.get("DATABEACON_FRONTEND_DIST", Path(__file__).resolve().parents[3] / "frontend" / "dist"))
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


def _public_ds(ds) -> dict[str, Any]:
    d = ds.to_dict()
    d.pop("owner", None)
    return d


app = create_app()
