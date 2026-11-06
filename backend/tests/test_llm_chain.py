import pytest
from langchain_core.messages import AIMessage, HumanMessage

from databeacon.budget import NeuronBudget, estimate_neurons
from databeacon.llm import ProviderChain, QuotaExhaustedError

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class Fake:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.calls = reply, exc, 0

    def bind_tools(self, tools, **kw):
        return self

    async def ainvoke(self, messages, **kw):
        self.calls += 1
        if self.exc:
            raise self.exc
        return AIMessage(self.reply, usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200})


def test_neuron_estimate():
    # 1M input + 1M output tokens of gpt-oss-120b = 31,818 + 68,182 neurons
    assert estimate_neurons("@cf/openai/gpt-oss-120b", 1_000_000, 1_000_000) == pytest.approx(100_000)


async def test_records_cloudflare_neurons(tmp_path):
    budget = NeuronBudget(tmp_path / "u.sqlite", daily_limit=9000)
    chain = ProviderChain([("cloudflare", "@cf/openai/gpt-oss-120b", Fake("ok"))], budget)
    out = await chain.ainvoke([HumanMessage("hi")])
    assert out.content == "ok" and chain.last_served["provider"] == "cloudflare"
    assert budget.used_today() == pytest.approx(estimate_neurons("@cf/openai/gpt-oss-120b", 1000, 200))


async def test_falls_back_on_error(tmp_path):
    cf = Fake(exc=RuntimeError("Error code: 400 - model does not support tools"))
    groq = Fake("from groq")
    chain = ProviderChain([("cloudflare", "@cf/x", cf), ("groq", "g", groq)], NeuronBudget(tmp_path / "u.sqlite"))
    out = await chain.bind_tools([]).ainvoke([HumanMessage("hi")])
    assert out.content == "from groq" and chain.last_served == {"provider": "groq", "model": "g"}
    assert chain.shared["fallbacks"] == 1


async def test_budget_exhausted_skips_cloudflare(tmp_path):
    budget = NeuronBudget(tmp_path / "u.sqlite", daily_limit=1)
    budget.record("cloudflare", "@cf/openai/gpt-oss-120b", 100_000, 0)
    cf, groq = Fake("cf"), Fake("groq")
    chain = ProviderChain([("cloudflare", "@cf/openai/gpt-oss-120b", cf), ("groq", "g", groq)], budget)
    assert (await chain.ainvoke([HumanMessage("hi")])).content == "groq"
    assert cf.calls == 0  # never called once over budget
    with pytest.raises(QuotaExhaustedError):
        await ProviderChain([("cloudflare", "@cf/openai/gpt-oss-120b", cf)], budget).ainvoke([HumanMessage("hi")])


def test_cloudflare_payload_normalisation(monkeypatch):
    from langchain_core.messages import SystemMessage, ToolMessage

    from databeacon.llm import build_provider_model

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "0" * 32)
    monkeypatch.setenv("DATABEACON_INJECTED_AUTH", "1")
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    model = build_provider_model("cloudflare")
    msgs = [SystemMessage("s"), HumanMessage("q"),
            AIMessage("", tool_calls=[{"name": "run_sql", "args": {"sql": "SELECT 1"}, "id": "c1", "type": "tool_call"}]),
            ToolMessage('{"rows": [[1]]}', tool_call_id="c1", name="run_sql")]
    payload = model._get_request_payload(msgs)
    assert all(m["content"] is not None for m in payload["messages"])
    assert "name" not in payload["messages"][-1]
    assert payload["max_tokens"] == 4096
    assert model.openai_api_key.get_secret_value() == "injected-by-egress-proxy"
