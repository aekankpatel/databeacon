"""FastAPI backend for DataBeacon."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .. import __version__
from ..config import get_settings
from ..datasets import DatasetError, DatasetRegistry
from ..llm import provider_info
from ..mcp_client import MCPToolHub
from ..observability import setup_tracing
from ..profiling import profile_table, schema_summary
from ..runs import RunManager
from ..sql_guard import SQLValidationError, execute_readonly

SAMPLE_CSV = Path(__file__).resolve().parents[2] / "sample_data" / "ecommerce_orders.csv"
EVALS_DIR = Path(__file__).resolve().parents[2] / "evals" / "results"


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
    registry = manager.registry if manager else DatasetRegistry(settings.datasets_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runs = manager or RunManager(registry, settings)
        yield
        for t in app.state.runs.tasks.values():
            t.cancel()

    app = FastAPI(title="DataBeacon API", version=__version__, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"],
                       allow_headers=["*"])

    def runs(request: Request) -> RunManager:
        return request.app.state.runs

    # ---------------------------------------------------------------- health
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "llm": provider_info(),
                "mcp_transport": settings.mcp_transport}

    @app.get("/api/usage")
    async def llm_usage() -> dict[str, Any]:
        """Today's LLM usage and the Cloudflare free-tier neuron budget."""
        from ..budget import NeuronBudget

        return NeuronBudget(settings.data_dir / "llm_usage.sqlite").summary()

    # -------------------------------------------------------------- datasets
    @app.get("/api/datasets")
    async def list_datasets() -> list[dict[str, Any]]:
        return [d.to_dict() for d in registry.list()]

    @app.post("/api/datasets", status_code=201)
    async def upload_dataset(file: UploadFile = File(...), name: str | None = Form(None)) -> dict[str, Any]:
        filename = Path(file.filename or "upload.csv").name
        limit = settings.max_upload_mb * 1024 * 1024
        tmpdir = Path(tempfile.mkdtemp(prefix="upload_"))
        tmp = tmpdir / re.sub(r"[^\w.\-]", "_", filename)
        size = 0
        try:
            with tmp.open("wb") as fh:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
                    fh.write(chunk)
            ds = await asyncio.to_thread(registry.ingest_file, tmp, name=(name or None), original_filename=filename)
        except DatasetError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return ds.to_dict()

    @app.post("/api/datasets/sample", status_code=201)
    async def load_sample() -> dict[str, Any]:
        if not SAMPLE_CSV.exists():
            from sample_data.generate_ecommerce import generate

            await asyncio.to_thread(lambda: generate().to_csv(SAMPLE_CSV, index=False))
        ds = await asyncio.to_thread(registry.ingest_file, SAMPLE_CSV, name="E-commerce orders (sample)")
        return ds.to_dict()

    @app.get("/api/datasets/{dataset_id}")
    async def get_dataset(dataset_id: str) -> dict[str, Any]:
        try:
            return registry.get(dataset_id).to_dict()
        except DatasetError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.delete("/api/datasets/{dataset_id}", status_code=204)
    async def delete_dataset(dataset_id: str) -> None:
        try:
            registry.delete(dataset_id)
        except DatasetError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/datasets/{dataset_id}/profile")
    async def dataset_profile(dataset_id: str) -> dict[str, Any]:
        try:
            ds = registry.get(dataset_id)
            prof = await asyncio.to_thread(profile_table, registry.db_path(dataset_id), ds.table)
        except DatasetError as exc:
            raise HTTPException(404, str(exc)) from exc
        prof["summary"] = schema_summary(prof)
        return prof

    @app.post("/api/datasets/{dataset_id}/sql")
    async def run_sql(dataset_id: str, body: SQLRequest) -> dict[str, Any]:
        """Read-only SQL console (same sandbox the agent uses)."""
        try:
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
    async def start_run(body: RunRequest, request: Request) -> dict[str, Any]:
        try:
            run_id = runs(request).start(body.dataset_id, body.question, body.clarification)
        except DatasetError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"run_id": run_id, "status": "running"}

    @app.get("/api/runs")
    async def list_runs(request: Request, dataset_id: str | None = None) -> list[dict[str, Any]]:
        return runs(request).store.list(dataset_id)

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str, request: Request) -> dict[str, Any]:
        row = runs(request).get(run_id)
        if row is None:
            raise HTTPException(404, "run not found")
        return row

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request) -> StreamingResponse:
        mgr = runs(request)
        rec = mgr.recorders.get(run_id)
        row = mgr.store.get(run_id)
        if row is None:
            raise HTTPException(404, "run not found")

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
    async def run_report(run_id: str) -> FileResponse:
        if not re.fullmatch(r"[0-9a-f]{12}", run_id):
            raise HTTPException(400, "invalid run id")
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


app = create_app()
