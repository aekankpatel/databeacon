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
# Tools the agent loop never needs to choose: the profile is computed up front and put in the prompt,
# and the PDF is rendered by the report node.
AGENT_EXCLUDED_TOOLS = ("profile_dataset", "list_tables", "generate_pdf_report")
MAX_TOOL_DESCRIPTION = 240


def _compact_description(text: str, limit: int = MAX_TOOL_DESCRIPTION) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[: end + 1] if end > limit // 2 else cut.rstrip() + "…").strip()


def _compact_schema(node: Any, _in_properties: bool = False) -> Any:
    """Drop schema metadata titles and collapse Optional[...] (anyOf with null): same contract, fewer tokens.

    Keys inside a ``properties`` mapping are parameter names and are always kept (a parameter may be
    called ``title``)."""
    if isinstance(node, dict):
        if _in_properties:
            return {k: _compact_schema(v) for k, v in node.items()}
        node = {k: _compact_schema(v, _in_properties=(k == "properties")) for k, v in node.items()
                if not (k == "title" and isinstance(v, str))}
        any_of = node.get("anyOf")
        if isinstance(any_of, list):
            non_null = [x for x in any_of if x.get("type") != "null"]
            rest = {k: v for k, v in node.items() if k != "anyOf"}
            if len(non_null) == 1:
                node = {**non_null[0], **rest}
            elif all(set(x) == {"type"} for x in non_null):
                node = {"type": [x["type"] for x in non_null], **rest}
            else:
                node = {"anyOf": non_null, **rest}
        if node.get("default", "missing") is None:
            node.pop("default")
        return node
    if isinstance(node, list):
        return [_compact_schema(x) for x in node]
    return node


def validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Validate tool arguments against the tool's JSON schema; return a readable problem or None."""
    import jsonschema

    try:
        errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(args), key=lambda e: list(e.path))
    except jsonschema.SchemaError:
        return None
    if not errors:
        return None
    msgs = []
    for e in errors[:3]:
        where = ".".join(str(p) for p in e.path) or (e.validator == "required" and "") or "arguments"
        if e.validator == "required":
            msgs.append(e.message.replace(" is a required property", " is required"))
        elif e.validator == "minItems":
            msgs.append(f"'{where}' must list at least {e.validator_value} item(s), got {json.dumps(e.instance)}")
        elif e.validator == "type":
            msgs.append(f"'{where}' must be of type {e.validator_value}, got {json.dumps(e.instance)[:60]}")
        else:
            msgs.append(f"'{where}': {e.message[:160]}")
    return "; ".join(msgs)


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
        if not schema["required"]:
            schema.pop("required")
        return {"type": "function", "function": {"name": self.name,
                                                 "description": _compact_description(self.description),
                                                 "parameters": _compact_schema(schema)}}

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

    def llm_tools(self, exclude: tuple[str, ...] = AGENT_EXCLUDED_TOOLS) -> list[dict[str, Any]]:
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
        problem = validate_arguments(info.input_schema, args)
        if problem:  # reject before the round-trip, with a message the model can act on
            return ToolCallResult(name, info.server, False, None, f"Invalid arguments for {name}: {problem}",
                                  (time.perf_counter() - started) * 1000)
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
