"""LLM provider layer. All providers are reached via their OpenAI-compatible APIs.

Providers (``DATABEACON_LLM_PROVIDER``):
  * ``cloudflare`` - Cloudflare Workers AI (needs CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID)
  * ``groq``       - Groq (needs GROQ_API_KEY)
  * ``scripted``   - offline demo: a deterministic policy for the bundled sample dataset (no key needed)
  * ``openai``     - any OpenAI-compatible server (OPENAI_API_KEY, DATABEACON_LLM_BASE_URL),
                     e.g. a local Ollama / vLLM / LM Studio instance.
If unset, the first provider with credentials configured is used.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

DEFAULT_MODELS = {
    "cloudflare": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
    "scripted": "scripted-demo",
}

T = TypeVar("T", bound=BaseModel)


class LLMConfigError(RuntimeError):
    pass


def detect_provider() -> str:
    explicit = os.environ.get("DATABEACON_LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if os.environ.get("CLOUDFLARE_API_TOKEN"):
        return "cloudflare"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("DATABEACON_LLM_BASE_URL"):
        return "openai"
    return "cloudflare"


def provider_info() -> dict[str, Any]:
    p = detect_provider()
    model = os.environ.get("DATABEACON_MODEL") or DEFAULT_MODELS.get(p, "")
    configured = {
        "cloudflare": bool(os.environ.get("CLOUDFLARE_API_TOKEN") and os.environ.get("CLOUDFLARE_ACCOUNT_ID")),
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "openai": bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("DATABEACON_LLM_BASE_URL")),
        "scripted": True,
    }.get(p, False)
    return {"provider": p, "model": model, "configured": configured}


def build_chat_model(temperature: float = 0.1, timeout: float = 90.0, model: str | None = None) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    provider = detect_provider()
    if provider == "scripted":
        from .agent.testing import scripted_ecommerce_llm

        return scripted_ecommerce_llm()
    model = model or os.environ.get("DATABEACON_MODEL") or DEFAULT_MODELS.get(provider)
    if provider == "cloudflare":
        token, account = os.environ.get("CLOUDFLARE_API_TOKEN"), os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        if not token or not account:
            raise LLMConfigError("Set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID to use Cloudflare Workers AI.")
        base_url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"
        api_key = token
    elif provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise LLMConfigError("Set GROQ_API_KEY to use Groq.")
        base_url = "https://api.groq.com/openai/v1"
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
        base_url = os.environ.get("DATABEACON_LLM_BASE_URL") or None
    else:
        raise LLMConfigError(f"Unknown LLM provider '{provider}'")
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=temperature, timeout=timeout,
                      max_retries=2)


# --------------------------------------------------------------------------- helpers
_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)\s*(ms|s)", re.IGNORECASE)


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return ("ratelimit" in name or "429" in msg or "rate limit" in msg or "timeout" in name
            or "503" in msg or "502" in msg or "overloaded" in msg)


async def ainvoke_with_retry(llm: Any, messages: list[BaseMessage], attempts: int = 6) -> AIMessage:
    """Invoke with backoff on rate limits / transient errors (honours 'try again in Xs' hints)."""
    import asyncio

    delay = 2.0
    for i in range(attempts):
        try:
            return await llm.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001
            if i == attempts - 1 or not _is_retryable(exc):
                raise
            m = _RETRY_AFTER_RE.search(str(exc))
            wait = (float(m.group(1)) / (1000 if m.group(2).lower() == "ms" else 1)) + 0.5 if m else delay
            await asyncio.sleep(min(wait, 60.0))
            delay = min(delay * 2, 30.0)
    raise RuntimeError("unreachable")


def usage_of(msg: BaseMessage) -> dict[str, int]:
    u = getattr(msg, "usage_metadata", None) or {}
    return {"input_tokens": int(u.get("input_tokens", 0) or 0), "output_tokens": int(u.get("output_tokens", 0) or 0)}


def extract_json(text: str) -> Any:
    """Return the first JSON object/array found in ``text`` (handles ```json fences and chatter)."""
    if not text:
        raise ValueError("empty response")
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    candidates.append(text)
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = cand.find(opener)
            while start != -1:
                depth, in_str, esc = 0, False, False
                for i in range(start, len(cand)):
                    ch = cand[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                    elif ch == '"':
                        in_str = True
                    elif ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(cand[start:i + 1])
                            except json.JSONDecodeError:
                                break
                start = cand.find(opener, start + 1)
    raise ValueError("no JSON object found in model output")


def text_of(msg: BaseMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


async def invoke_structured(llm: BaseChatModel, messages: list[BaseMessage], schema: type[T],
                            retries: int = 2) -> tuple[T, dict[str, int]]:
    """Structured output via JSON-schema prompting + validation + repair retries (portable across providers)."""
    usage = {"input_tokens": 0, "output_tokens": 0}
    schema_json = json.dumps(schema.model_json_schema())
    msgs = list(messages) + [HumanMessage(
        f"Respond with ONLY a JSON object (no prose, no code fences) that validates against this JSON schema:\n{schema_json}"
    )]
    last_err: Exception | None = None
    for _ in range(retries + 1):
        resp = await ainvoke_with_retry(llm, msgs)
        u = usage_of(resp)
        usage = {k: usage[k] + u[k] for k in usage}
        raw = text_of(resp)
        try:
            data = extract_json(raw)
            return schema.model_validate(data), usage
        except (ValueError, ValidationError) as exc:
            last_err = exc
            msgs = msgs + [AIMessage(raw[:4000]), HumanMessage(
                f"That output was invalid: {str(exc)[:800]}. Return ONLY the corrected JSON object.")]
    raise ValueError(f"Model failed to produce valid {schema.__name__}: {last_err}")


_TEXT_TOOL_RE = re.compile(r"<\|?(?:tool_call|function)[^>]*>", re.IGNORECASE)


def recover_text_tool_calls(msg: AIMessage, tool_names: set[str]) -> AIMessage:
    """Some open models emit tool calls as JSON in the message text. Convert them into real tool calls."""
    if msg.tool_calls:
        return msg
    text = _TEXT_TOOL_RE.sub("", text_of(msg))
    if "{" not in text:
        return msg
    try:
        data = extract_json(text)
    except ValueError:
        return msg
    items = data if isinstance(data, list) else [data]
    calls = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name") or it.get("tool") or (it.get("function") or {}).get("name")
        args = it.get("arguments") or it.get("parameters") or it.get("args") or (it.get("function") or {}).get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name in tool_names and isinstance(args, dict):
            calls.append({"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:10]}", "type": "tool_call"})
    if not calls:
        return msg
    return AIMessage(content="", tool_calls=calls, usage_metadata=getattr(msg, "usage_metadata", None))
