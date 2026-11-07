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


def test_reservation_blocks_calls_that_could_exceed_budget(tmp_path):
    from databeacon.budget import worst_case_neurons

    budget = NeuronBudget(tmp_path / "u.sqlite", daily_limit=500)
    need = worst_case_neurons("@cf/openai/gpt-oss-120b", 20_000, 4096)  # ~280 output + ~200 input neurons
    r1 = budget.reserve(need)
    assert r1 is not None
    assert budget.reserve(need) is None  # a concurrent second call would overshoot: refused
    budget.release(r1)
    assert budget.reserve(need) is not None


def test_budget_cannot_exceed_free_allocation_without_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABEACON_CLOUDFLARE_DAILY_NEURONS", "50000")
    monkeypatch.delenv("DATABEACON_ALLOW_PAID_INFERENCE", raising=False)
    assert NeuronBudget(tmp_path / "a.sqlite").daily_limit == 10_000
    monkeypatch.setenv("DATABEACON_ALLOW_PAID_INFERENCE", "1")
    assert NeuronBudget(tmp_path / "b.sqlite").daily_limit == 50_000


async def test_cloudflare_fails_closed(tmp_path):
    cf = Fake("cf")
    with pytest.raises(QuotaExhaustedError):  # no budget configured -> never call Cloudflare
        await ProviderChain([("cloudflare", "@cf/openai/gpt-oss-120b", cf)], None).ainvoke([HumanMessage("hi")])
    budget = NeuronBudget(tmp_path / "u.sqlite")
    budget.path = tmp_path / "missing_dir" / "u.sqlite"  # ledger unusable
    with pytest.raises(QuotaExhaustedError):
        await ProviderChain([("cloudflare", "@cf/openai/gpt-oss-120b", cf)], budget).ainvoke([HumanMessage("hi")])
    assert cf.calls == 0


async def test_failed_call_releases_reservation(tmp_path):
    budget = NeuronBudget(tmp_path / "u.sqlite", daily_limit=9000)
    chain = ProviderChain([("cloudflare", "@cf/openai/gpt-oss-120b", Fake(exc=RuntimeError("400 bad")))], budget)
    with pytest.raises(RuntimeError):
        await chain.ainvoke([HumanMessage("hi")])
    assert budget.remaining() == 9000 and budget.used_today() == 0


class Cf4006(Exception):
    """Mimics the OpenAI SDK error: str() hides Cloudflare's body."""

    def __init__(self):
        super().__init__("Error code: 429")
        self.body = {"errors": [{"code": 4006, "message": "AiError: you have used up your daily free allocation "
                                                         "of 10,000 neurons"}]}


async def test_cloudflare_4006_fails_fast_and_blocks_provider(tmp_path):
    budget = NeuronBudget(tmp_path / "u.sqlite", daily_limit=9000)
    cf = Fake(exc=Cf4006())
    chain = ProviderChain([("cloudflare", "@cf/openai/gpt-oss-120b", cf)], budget)
    with pytest.raises(QuotaExhaustedError, match="daily free allocation"):
        await chain.ainvoke([HumanMessage("hi")])
    assert cf.calls == 1  # no retry loop on an exhausted allocation
    assert budget.blocked() and "4006" in budget.blocked()
    with pytest.raises(QuotaExhaustedError, match="no cloudflare calls until"):
        await chain.ainvoke([HumanMessage("hi")])
    assert cf.calls == 1  # blocked: Cloudflare not contacted again


async def test_block_falls_through_to_configured_fallback(tmp_path):
    budget = NeuronBudget(tmp_path / "u.sqlite")
    budget.block_provider("cloudflare", "test block")
    cf, groq = Fake("cf"), Fake("groq")
    out = await ProviderChain([("cloudflare", "@cf/openai/gpt-oss-120b", cf), ("groq", "g", groq)], budget).ainvoke(
        [HumanMessage("hi")])
    assert out.content == "groq" and cf.calls == 0


def test_rolling_window_counts_last_evening(tmp_path):
    import sqlite3
    import time

    path = tmp_path / "u.sqlite"
    NeuronBudget(path)
    with sqlite3.connect(path) as c:  # 7,000 neurons spent 3 hours ago (e.g. yesterday evening)
        c.execute("INSERT INTO llm_calls (ts, provider, model, neurons) VALUES (?, 'cloudflare', 'm', 7000)",
                  (time.time() - 3 * 3600,))
        c.execute("INSERT INTO llm_calls (ts, provider, model, neurons) VALUES (?, 'cloudflare', 'm', 5000)",
                  (time.time() - 30 * 3600,))  # outside 24h
    rolling = NeuronBudget(path, daily_limit=9000)
    assert rolling.used() == 7000 and rolling.remaining() == 2000
    assert rolling.reserve(2500) is None


def test_legacy_daily_totals_migrate_conservatively(tmp_path):
    import sqlite3
    from datetime import UTC, datetime, timedelta

    path = tmp_path / "u.sqlite"
    with sqlite3.connect(path) as c:  # a ledger written before per-call timestamps existed
        c.execute("""CREATE TABLE llm_usage (day TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
            calls INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0, neurons REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (day, provider, model))""")
        yesterday = (datetime.now(UTC) - timedelta(days=1)).date().isoformat()
        c.execute("INSERT INTO llm_usage VALUES (?, 'cloudflare', 'm', 57, 0, 0, 6815.3)", (yesterday,))
    b = NeuronBudget(path, daily_limit=9000)
    # counted at the end of yesterday, i.e. still inside the last 24 hours
    assert b.used() == pytest.approx(6815.3)
    assert NeuronBudget(path, daily_limit=9000, window="utc_day").used() == 0
