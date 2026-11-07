"""LLM provider layer. All providers are reached via their OpenAI-compatible APIs.

Providers (``DATABEACON_LLM_PROVIDER``):
  * ``cloudflare`` - Cloudflare Workers AI (primary). Needs CLOUDFLARE_ACCOUNT_ID; the API token comes
                     from CLOUDFLARE_API_TOKEN or is injected by an egress proxy (see below).
  * ``groq``       - Groq (needs GROQ_API_KEY, or proxy-injected auth)
  * ``openai``     - any OpenAI-compatible server (OPENAI_API_KEY, DATABEACON_LLM_BASE_URL),
                     e.g. a local Ollama / vLLM / LM Studio instance.
  * ``scripted``   - offline demo: a deterministic policy for the bundled sample dataset (no key needed)

``DATABEACON_FALLBACK_PROVIDER`` (e.g. ``groq``) is used when the primary fails or the Cloudflare
free-tier budget is exhausted.

If an egress proxy injects the Authorization header (``DATABEACON_INJECTED_AUTH=1``), no token is read
or needed: a placeholder is sent and the proxy replaces it.
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
    "cloudflare": "@cf/openai/gpt-oss-120b",
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
    "scripted": "scripted-demo",
}
INJECTED_PLACEHOLDER = "injected-by-egress-proxy"

T = TypeVar("T", bound=BaseModel)


class LLMConfigError(RuntimeError):
    pass


def _injected() -> bool:
    return os.environ.get("DATABEACON_INJECTED_AUTH", "").lower() in {"1", "true", "yes"}


def detect_provider() -> str:
    explicit = os.environ.get("DATABEACON_LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if os.environ.get("CLOUDFLARE_ACCOUNT_ID"):
        return "cloudflare"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("DATABEACON_LLM_BASE_URL"):
        return "openai"
    return "cloudflare"


def fallback_provider() -> str | None:
    fb = os.environ.get("DATABEACON_FALLBACK_PROVIDER", "").strip().lower()
    return fb if fb and fb != detect_provider() else None


def model_for(provider: str) -> str:
    if provider == detect_provider() and os.environ.get("DATABEACON_MODEL"):
        return os.environ["DATABEACON_MODEL"]
    if provider == fallback_provider() and os.environ.get("DATABEACON_FALLBACK_MODEL"):
        return os.environ["DATABEACON_FALLBACK_MODEL"]
    return DEFAULT_MODELS.get(provider, "")


def is_configured(provider: str) -> bool:
    return {
        "cloudflare": bool(os.environ.get("CLOUDFLARE_ACCOUNT_ID")
                           and (os.environ.get("CLOUDFLARE_API_TOKEN") or _injected())),
        "groq": bool(os.environ.get("GROQ_API_KEY") or _injected()),
        "openai": bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("DATABEACON_LLM_BASE_URL")),
        "scripted": True,
    }.get(provider, False)


def provider_info() -> dict[str, Any]:
    p = detect_provider()
    fb = fallback_provider()
    return {"provider": p, "model": model_for(p), "configured": is_configured(p),
            "fallback": {"provider": fb, "model": model_for(fb), "configured": is_configured(fb)} if fb else None}


def _cloudflare_chat_class():
    from langchain_openai import ChatOpenAI

    class CloudflareChatOpenAI(ChatOpenAI):
        """Workers AI's OpenAI-compatible endpoint is stricter than OpenAI's:

        * an assistant message that carries tool_calls must have string content - ``null`` (what the
          OpenAI spec and LangChain send) is rejected with a bare HTTP 400;
        * a ``name`` on tool-result messages makes gpt-oss leak its reasoning into the answer.
        """

        def _get_request_payload(self, input_, *, stop=None, **kwargs):  # noqa: ANN001
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            if "max_completion_tokens" in payload:  # send both spellings; Workers AI default is 256
                payload.setdefault("max_tokens", payload["max_completion_tokens"])
            for m in payload.get("messages", []):
                if m.get("content") is None:
                    m["content"] = ""
                if m.get("role") == "tool":
                    m.pop("name", None)
            return payload

    return CloudflareChatOpenAI


def build_provider_model(provider: str, temperature: float = 0.1, timeout: float = 90.0,
                         model: str | None = None) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if provider == "scripted":
        from .agent.testing import scripted_ecommerce_llm

        return scripted_ecommerce_llm()
    model = model or model_for(provider)
    if provider == "cloudflare":
        account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        if not account:
            raise LLMConfigError("Set CLOUDFLARE_ACCOUNT_ID to use Cloudflare Workers AI.")
        if not re.fullmatch(r"[0-9a-f]{32}", account):
            raise LLMConfigError("CLOUDFLARE_ACCOUNT_ID should be a 32-character hex id.")
        api_key = os.environ.get("CLOUDFLARE_API_TOKEN") or (INJECTED_PLACEHOLDER if _injected() else None)
        if not api_key:
            raise LLMConfigError("Set CLOUDFLARE_API_TOKEN (or DATABEACON_INJECTED_AUTH=1 behind an auth proxy).")
        base_url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"
    elif provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY") or (INJECTED_PLACEHOLDER if _injected() else None)
        if not api_key:
            raise LLMConfigError("Set GROQ_API_KEY to use Groq.")
        base_url = "https://api.groq.com/openai/v1"
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
        base_url = os.environ.get("DATABEACON_LLM_BASE_URL") or None
    else:
        raise LLMConfigError(f"Unknown LLM provider '{provider}'")
    # Set max_tokens explicitly: Workers AI defaults to 256 output tokens, which a reasoning model can spend
    # entirely on hidden reasoning, leaving an empty answer (finish_reason=length).
    extra: dict[str, Any] = {}
    effort = os.environ.get("DATABEACON_REASONING_EFFORT", "").strip().lower()
    if effort in {"low", "medium", "high"}:
        extra["reasoning_effort"] = effort
    cls = _cloudflare_chat_class() if provider == "cloudflare" else ChatOpenAI
    # Cloudflare: no SDK-level retries. The SDK would silently resend a 429 - including error 4006
    # ("daily free allocation used up") - before our code can see it; ainvoke_with_retry decides instead.
    return cls(model=model, api_key=api_key, base_url=base_url, temperature=temperature, timeout=timeout,
                      max_retries=0 if provider == "cloudflare" else 1, max_tokens=int(os.environ.get("DATABEACON_MAX_OUTPUT_TOKENS", 4096)), **extra)


class ProviderChain:
    """Ordered list of providers: the first that succeeds serves the call.

    Cloudflare calls are checked against (and recorded in) the daily neuron budget, so the agent never
    spends past the free tier; once the budget is gone the next provider is used.
    """

    def __init__(self, entries: list[tuple[str, str, Any]], budget: Any = None, _shared: dict | None = None):
        self.entries = entries  # (provider, model, chat model)
        self.budget = budget
        self.shared = _shared if _shared is not None else {"last": None, "fallbacks": 0, "errors": []}
        self.model_name = entries[0][1]

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> ProviderChain:
        bound = ProviderChain([(p, m, llm.bind_tools(tools, **kwargs)) for p, m, llm in self.entries],
                              self.budget, dict(self.shared))
        bound.shared = self.shared  # share last-served/fallback state with the unbound chain
        bound._tool_schema_chars = len(json.dumps(tools, separators=(",", ":"), default=str))
        return bound

    @property
    def last_served(self) -> dict[str, str] | None:
        return self.shared["last"]

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        from .budget import BudgetUnavailable, worst_case_neurons

        last_exc: Exception | None = None
        max_out = int(os.environ.get("DATABEACON_MAX_OUTPUT_TOKENS", 4096))
        prompt_chars = sum(len(text_of(m)) for m in messages) + getattr(self, "_tool_schema_chars", 0)
        for i, (provider, model, llm) in enumerate(self.entries):
            reservation = None
            if provider == "cloudflare":
                try:
                    block = self.budget.blocked(provider) if self.budget is not None else None
                except Exception as exc:  # noqa: BLE001 - unreadable ledger: fail closed
                    block = f"neuron ledger unavailable ({type(exc).__name__})"
                if block:
                    last_exc = QuotaExhaustedError(f"Cloudflare call refused: {block}")
                    self.shared["errors"].append(str(last_exc))
                    continue
                if self.budget is None:  # fail closed: never call Cloudflare without accounting
                    last_exc = QuotaExhaustedError("Cloudflare call refused: no neuron budget configured.")
                    self.shared["errors"].append(str(last_exc))
                    continue
                need = worst_case_neurons(model, prompt_chars, max_out)
                try:
                    reservation = self.budget.reserve(need)
                except BudgetUnavailable as exc:
                    last_exc = QuotaExhaustedError(f"Cloudflare call refused: {exc}")
                    self.shared["errors"].append(str(last_exc))
                    continue
                if reservation is None:
                    last_exc = QuotaExhaustedError(
                        f"Cloudflare daily neuron budget reached ({self.budget.used_today():.0f}/"
                        f"{self.budget.daily_limit:.0f} used, this call may need up to {need:.0f}); "
                        "not calling Cloudflare until 00:00 UTC.")
                    self.shared["errors"].append(str(last_exc))
                    continue
            try:
                resp = await ainvoke_with_retry(llm, messages, attempts=4 if i < len(self.entries) - 1 else 6)
            except Exception as exc:  # noqa: BLE001 - try the next provider
                if self.budget is not None:
                    self.budget.release(reservation)
                    if provider == "cloudflare" and isinstance(exc, QuotaExhaustedError):
                        # Cloudflare's own meter says the free allocation is gone: stop calling it
                        self.budget.block_provider(provider, "Cloudflare reported its daily free allocation "
                                                             "exhausted (error 4006)")
                last_exc = exc
                self.shared["errors"].append(f"{provider}: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            u = usage_of(resp)
            if not (u["input_tokens"] or u["output_tokens"]):  # some endpoints omit usage: estimate
                u = {"input_tokens": prompt_chars // 4,
                     "output_tokens": len(text_of(resp)) // 4 + 16 * len(resp.tool_calls or [])}
                resp.usage_metadata = {**u, "total_tokens": sum(u.values())}
            if self.budget is not None:
                token_usage = (getattr(resp, "response_metadata", None) or {}).get("token_usage") or {}
                reported = token_usage.get("neurons") if isinstance(token_usage, dict) else None
                self.budget.record(provider, model, u["input_tokens"], u["output_tokens"], reported,
                                   reservation_id=reservation)
            if i > 0:
                self.shared["fallbacks"] += 1
            self.shared["last"] = {"provider": provider, "model": model}
            return resp
        assert last_exc is not None
        raise last_exc


def build_chat_model(temperature: float = 0.1, timeout: float = 90.0, model: str | None = None,
                     budget: Any = None) -> Any:
    primary = detect_provider()
    if primary == "scripted":
        return build_provider_model("scripted")
    entries = [(primary, model or model_for(primary), build_provider_model(primary, temperature, timeout, model))]
    fb = fallback_provider()
    if fb and is_configured(fb):
        entries.append((fb, model_for(fb), build_provider_model(fb, temperature, timeout)))
    if budget is None:
        from .budget import NeuronBudget

        budget = NeuronBudget.default()
    return ProviderChain(entries, budget)


# --------------------------------------------------------------------------- helpers
_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)\s*(ms|s)", re.IGNORECASE)


class QuotaExhaustedError(RuntimeError):
    """The provider's daily quota is used up; retrying within this run is pointless."""


def error_text(exc: BaseException) -> str:
    """The full error, including the provider's response body: OpenAI-SDK errors often stringify to just
    'Error code: 429', hiding Cloudflare's code 4006 'daily free allocation' message."""
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if body:
        parts.append(json.dumps(body, default=str) if not isinstance(body, str) else body)
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            parts.append(resp.text[:2000])
        except Exception:  # noqa: BLE001 - best effort
            pass
    cause = exc.__cause__
    if cause is not None and cause is not exc:
        parts.append(error_text(cause))
    return " | ".join(p for p in parts if p)


def _is_daily_quota(exc: BaseException) -> bool:
    msg = error_text(exc).lower()
    if '"code": 4006' in msg or "4006" in msg and "allocation" in msg:
        return True
    return ("per day" in msg or "(tpd)" in msg or "(rpd)" in msg or "neuron" in msg
            or ("daily" in msg and ("limit" in msg or "allocation" in msg)))


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
            if _is_daily_quota(exc):
                raise QuotaExhaustedError(
                    f"LLM provider daily quota exhausted: {error_text(exc)[:400]}") from exc
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
    from .mcp_client import _compact_schema

    schema_json = json.dumps(_compact_schema(schema.model_json_schema()), separators=(",", ":"))
    msgs = list(messages) + [HumanMessage(
        f"Respond with ONLY a JSON object (no prose, no code fences) that validates against this JSON schema:\n{schema_json}"
    )]
    last_err: Exception | None = None
    for _ in range(retries + 1):
        resp = await ainvoke_with_retry(llm, msgs)
        u = usage_of(resp)
        usage = {k: usage[k] + u[k] for k in usage}
        raw = text_of(resp)
        if not raw.strip() and (resp.response_metadata or {}).get("finish_reason") == "length":
            last_err = ValueError("output truncated (finish_reason=length); raise DATABEACON_MAX_OUTPUT_TOKENS")
            msgs = msgs + [HumanMessage("Your previous answer was cut off. Keep reasoning brief and output the JSON.")]
            continue
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
