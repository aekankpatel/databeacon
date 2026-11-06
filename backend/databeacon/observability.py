"""Agent observability: OpenTelemetry spans + an execution timeline streamed to the UI."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_configured = False


def setup_tracing() -> None:
    """Configure the global tracer provider once. OTLP if an endpoint is set, console if DATABEACON_OTEL_CONSOLE=1."""
    global _configured
    if _configured:
        return
    _configured = True
    provider = TracerProvider(resource=Resource.create({"service.name": "databeacon"}))
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        except ImportError:  # pragma: no cover - optional dependency
            pass
    if os.environ.get("DATABEACON_OTEL_CONSOLE") == "1":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


tracer = trace.get_tracer("databeacon.agent")


def _clip(value: Any, limit: int = 4000) -> Any:
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) <= limit:
        return value
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


class RunRecorder:
    """Collects timeline events for one agent run and fans them out to live subscribers."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self._subscribers: list[asyncio.Queue] = []
        self.tokens = {"input_tokens": 0, "output_tokens": 0}
        self.llm_calls = 0
        self.served_by: dict[str, int] = {}
        self.by_stage: dict[str, dict[str, int]] = {}
        self.started = time.perf_counter()

    # --------------------------------------------------------------- pub/sub
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        for e in self.events:
            q.put_nowait({"kind": "event", "event": e})
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def publish(self, message: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            q.put_nowait(message)

    # ---------------------------------------------------------------- events
    def add_tokens(self, usage: dict[str, int], served: dict[str, str] | None = None, stage: str = "other") -> None:
        self.llm_calls += 1
        st = self.by_stage.setdefault(stage, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        st["calls"] += 1
        st["input_tokens"] += int(usage.get("input_tokens", 0))
        st["output_tokens"] += int(usage.get("output_tokens", 0))
        if served:
            key = f"{served['provider']}:{served['model']}"
            self.served_by[key] = self.served_by.get(key, 0) + 1
        for k in self.tokens:
            self.tokens[k] += int(usage.get(k, 0))

    def log(self, type_: str, name: str, **fields: Any) -> dict[str, Any]:
        event = {
            "id": uuid.uuid4().hex[:10], "type": type_, "name": name, "status": fields.pop("status", "ok"),
            "started_at": datetime.now(timezone.utc).isoformat(), "duration_ms": fields.pop("duration_ms", 0.0),
            **{k: _clip(v) for k, v in fields.items()},
        }
        self.events.append(event)
        self.publish({"kind": "event", "event": event})
        return event

    @asynccontextmanager
    async def span(self, type_: str, name: str, **attrs: Any) -> AsyncIterator[dict[str, Any]]:
        """Times a step, emits a 'running' event then the completed event, and mirrors it as an OTel span."""
        event = {
            "id": uuid.uuid4().hex[:10], "type": type_, "name": name, "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(), "duration_ms": 0.0,
            **{k: _clip(v) for k, v in attrs.items()},
        }
        self.events.append(event)
        self.publish({"kind": "event", "event": event})
        t0 = time.perf_counter()
        with tracer.start_as_current_span(f"{type_}:{name}") as otel_span:
            otel_span.set_attribute("databeacon.run_id", self.run_id)
            otel_span.set_attribute("databeacon.step_type", type_)
            try:
                yield event
                if event["status"] == "running":
                    event["status"] = "ok"
            except Exception as exc:
                event["status"] = "error"
                event["error"] = f"{type(exc).__name__}: {exc}"
                otel_span.record_exception(exc)
                raise
            finally:
                event["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                for k in ("output", "input"):
                    if k in event:
                        event[k] = _clip(event[k])
                otel_span.set_attribute("databeacon.status", event["status"])
                if "tokens" in event:
                    otel_span.set_attribute("llm.tokens.total", sum(event["tokens"].values()))
                self.publish({"kind": "event", "event": event})

    def metrics(self) -> dict[str, Any]:
        tool_events = [e for e in self.events if e["type"] == "tool"]
        return {
            "latency_s": round(time.perf_counter() - self.started, 2),
            "llm_calls": self.llm_calls,
            "input_tokens": self.tokens["input_tokens"],
            "output_tokens": self.tokens["output_tokens"],
            "total_tokens": sum(self.tokens.values()),
            "tool_calls": len(tool_events),
            "tool_errors": sum(1 for e in tool_events if e["status"] == "error"),
            "sql_queries": sum(1 for e in tool_events if e["name"] == "run_sql"),
            "served_by": dict(self.served_by),
            "tokens_by_stage": {k: dict(v) for k, v in self.by_stage.items()},
            "duplicate_calls_blocked": sum(1 for e in self.events if e["name"] == "duplicate_call_blocked"),
        }
