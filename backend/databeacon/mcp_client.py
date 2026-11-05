"""MCP client: connects to the DataBeacon MCP servers, discovers tools/resources/prompts and executes calls.

Servers run in-process by default (zero overhead) or as stdio subprocesses
(``DATABEACON_MCP_TRANSPORT=stdio``), which is exactly how any third-party MCP
client would talk to them.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import Client, StdioServerParameters

from .config import get_settings
from .datasets import DatasetRegistry

SERVER_MODULES = {
    "database": "databeacon.mcp_servers.database",
    "analysis": "databeacon.mcp_servers.analysis",
    "report": "databeacon.mcp_servers.report",
}

# Arguments filled in by the client from run context and hidden from the LLM.
INJECTED_ARGS = ("dataset_id",)


@dataclass
class ToolInfo:
    name: str
    server: str
    description: str
    input_schema: dict[str, Any]

    def llm_schema(self) -> dict[str, Any]:
        schema = json.loads(json.dumps(self.input_schema))
        props = schema.get("properties", {})
        for arg in INJECTED_ARGS:
            props.pop(arg, None)
        schema["required"] = [r for r in schema.get("required", []) if r not in INJECTED_ARGS]
        schema.pop("title", None)
        for p in props.values():
            p.pop("title", None)
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": schema}}

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "server": self.server, "description": self.description,
                "input_schema": self.input_schema}


@dataclass
class ToolCallResult:
    tool: str
    server: str
    ok: bool
    data: Any
    error: str | None
    elapsed_ms: float


@dataclass
class MCPToolHub:
    registry: DatasetRegistry | None = None
    transport: str | None = None
    servers: tuple[str, ...] = ("database", "analysis", "report")
    tools: dict[str, ToolInfo] = field(default_factory=dict)
    resources: list[dict[str, Any]] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)
    _clients: dict[str, Client] = field(default_factory=dict)
    _stack: AsyncExitStack | None = None

    def _target(self, name: str):
        transport = self.transport or get_settings().mcp_transport
        if transport == "stdio":
            env = {**os.environ, "DATABEACON_DATA_DIR": str(get_settings().data_dir)}
            if self.registry is not None:
                env["DATABEACON_DATA_DIR"] = str(self.registry.root.parent)
            return StdioServerParameters(command=sys.executable, args=["-m", SERVER_MODULES[name]], env=env)
        from .mcp_servers.analysis import build_analysis_server
        from .mcp_servers.database import build_database_server
        from .mcp_servers.report import build_report_server

        factory = {"database": build_database_server, "analysis": build_analysis_server, "report": build_report_server}
        return factory[name](self.registry)

    async def __aenter__(self) -> MCPToolHub:
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        for name in self.servers:
            client = Client(self._target(name), read_timeout_seconds=120)
            self._clients[name] = await self._stack.enter_async_context(client)
        await self.discover()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(*exc)
        self._clients.clear()

    async def discover(self) -> dict[str, Any]:
        """Dynamic tool discovery across every connected server."""
        self.tools.clear()
        self.resources.clear()
        self.prompts.clear()
        for server, client in self._clients.items():
            listed = await client.list_tools()
            for t in listed.tools:
                schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
                self.tools[t.name] = ToolInfo(t.name, server, t.description or "", dict(schema))
            try:
                for r in (await client.list_resources()).resources:
                    self.resources.append({"server": server, "uri": str(r.uri), "name": r.name,
                                           "description": r.description})
                for r in (await client.list_resource_templates()).resource_templates:
                    self.resources.append({"server": server, "uri": r.uri_template, "name": r.name,
                                           "description": r.description, "template": True})
            except Exception:  # resources are optional
                pass
            try:
                for p in (await client.list_prompts()).prompts:
                    self.prompts.append({"server": server, "name": p.name, "description": p.description,
                                         "arguments": [a.name for a in (p.arguments or [])]})
            except Exception:
                pass
        return self.catalog()

    def catalog(self) -> dict[str, Any]:
        return {"tools": [t.to_dict() for t in self.tools.values()], "resources": self.resources,
                "prompts": self.prompts}

    def llm_tools(self, exclude: tuple[str, ...] = ("generate_pdf_report",)) -> list[dict[str, Any]]:
        return [t.llm_schema() for n, t in self.tools.items() if n not in exclude]

    async def call(self, name: str, arguments: dict[str, Any], context: dict[str, Any] | None = None) -> ToolCallResult:
        started = time.perf_counter()
        info = self.tools.get(name)
        if info is None:
            return ToolCallResult(name, "?", False, None,
                                  f"Unknown tool '{name}'. Available: {', '.join(self.tools)}", 0.0)
        args = dict(arguments or {})
        props = info.input_schema.get("properties", {})
        for k, v in (context or {}).items():
            if k in props:
                args[k] = v
        # drop hallucinated arguments that the tool does not accept
        args = {k: v for k, v in args.items() if k in props}
        try:
            res = await self._clients[info.server].call_tool(name, args)
        except Exception as exc:  # transport / protocol failures
            return ToolCallResult(name, info.server, False, None, f"{type(exc).__name__}: {exc}",
                                  (time.perf_counter() - started) * 1000)
        elapsed = (time.perf_counter() - started) * 1000
        texts = [c.text for c in res.content if getattr(c, "type", "") == "text"]
        payload: Any = getattr(res, "structured_content", None)
        if payload is None and texts:
            joined = "\n".join(texts)
            try:
                payload = json.loads(joined)
            except json.JSONDecodeError:
                payload = joined
        if isinstance(payload, dict) and set(payload) == {"result"}:
            payload = payload["result"]
        if res.is_error:
            msg = payload if isinstance(payload, str) else json.dumps(payload)
            return ToolCallResult(name, info.server, False, None, msg, elapsed)
        return ToolCallResult(name, info.server, True, payload, None, elapsed)
