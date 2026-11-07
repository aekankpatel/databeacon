"""Replay mode: recorded investigations served without any LLM use."""

import re
from pathlib import Path

from fastapi.testclient import TestClient

from databeacon.replays import REPLAY_DIR, get_replay, list_replays, replay_pdf


def _client(registry, llm_factory, monkeypatch, **env):
    from databeacon.api.main import create_app
    from databeacon.config import get_settings, reset_settings_cache
    from databeacon.runs import RunManager

    for k, v in env.items():
        monkeypatch.setenv(f"DATABEACON_{k}", v)
    reset_settings_cache()
    return TestClient(create_app(RunManager(registry, get_settings(), llm_factory=llm_factory)))


def _no_llm():
    raise AssertionError("replays must never call the LLM")


def test_replays_are_complete_verified_investigations():
    items = list_replays()
    assert {r["id"] for r in items} >= {"revenue-decline", "discount-shift", "shipping-by-region", "return-rate"}
    for r in items:
        full = get_replay(r["id"])
        res = full["result"]
        assert full["recorded_with"]["model"] == "@cf/openai/gpt-oss-120b"
        assert res["synthesis"]["summary"] and res["timeline"] and res["evidence"] and res["charts"]
        assert res["alignment"]["status"] == "aligned" and res["verification"]["hallucination_rate"] == 0
        assert r["metrics"]["approx_neurons"] > 0 and r["has_report"]
        assert replay_pdf(r["id"])[:4] == b"%PDF"
    assert get_replay("revenue-decline")["result"]["synthesis"]["answer"]["primary_entity"] == "Enterprise"


def test_replay_files_contain_no_local_paths_or_identifiers():
    bad = re.compile(r"/tmp/|/home/|scratchpad|\bclaude\b|[0-9a-f]{32}|\"owner\"", re.I)
    for p in Path(REPLAY_DIR).glob("*.json"):
        m = bad.search(p.read_text())
        assert m is None, f"{p.name}: {m.group(0)}"


def test_replay_ids_are_validated():
    for evil in ("../runs", "..%2F..", "revenue-decline/../../x", "", "A" * 80):
        assert get_replay(evil) is None and replay_pdf(evil) is None


def test_replay_api_works_without_session_or_llm(registry, monkeypatch):
    with _client(registry, _no_llm, monkeypatch, PUBLIC_MODE="1") as c:
        items = c.get("/api/replays").json()  # public: no session header needed
        assert len(items) >= 4
        r = c.get("/api/replays/revenue-decline").json()
        assert r["result"]["synthesis"]["answer"]["primary_entity"] == "Enterprise"
        pdf = c.get("/api/replays/revenue-decline/report.pdf")
        assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
        assert c.get("/api/replays/nope").status_code == 404
        assert c.get("/api/health").json()["replays"] >= 4
    monkeypatch.delenv("DATABEACON_PUBLIC_MODE")


def test_replay_only_mode_blocks_live_analyses(registry, monkeypatch):
    s = {"X-DataBeacon-Session": "r" * 32}
    with _client(registry, _no_llm, monkeypatch, REPLAY_ONLY="1") as c:
        ds = c.post("/api/datasets/sample", headers=s).json()
        r = c.post("/api/runs", headers=s, json={"dataset_id": ds["id"], "question": "Why did revenue fall?"})
        assert r.status_code == 403 and "recorded investigations" in r.json()["detail"]
        assert c.get("/api/health").json()["replay_only"] is True
        assert c.get("/api/replays/return-rate").status_code == 200
    monkeypatch.delenv("DATABEACON_REPLAY_ONLY")
    from databeacon.config import reset_settings_cache

    reset_settings_cache()


def test_sample_is_generated_when_not_bundled(registry, monkeypatch, tmp_path):
    import databeacon.api.main as api_main

    monkeypatch.setattr(api_main, "SAMPLE_CSV", tmp_path / "missing" / "ecommerce_orders.csv")  # as on Vercel
    with _client(registry, _no_llm, monkeypatch) as c:
        r = c.post("/api/datasets/sample", headers={"X-DataBeacon-Session": "g" * 32})
        assert r.status_code == 201 and r.json()["row_count"] > 10000
