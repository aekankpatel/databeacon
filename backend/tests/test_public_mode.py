"""Public-deployment protections: isolation, quotas, rate limits, access code, headers, retention."""

import io
import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from databeacon.agent.testing import scripted_ecommerce_llm

A = {"X-DataBeacon-Session": "a" * 32}
B = {"X-DataBeacon-Session": "b" * 32}


@contextmanager
def public_client(registry, monkeypatch, llm_factory=scripted_ecommerce_llm, **env):
    from databeacon.api.main import create_app
    from databeacon.config import get_settings, reset_settings_cache
    from databeacon.runs import RunManager

    monkeypatch.setenv("DATABEACON_PUBLIC_MODE", "1")
    for k, v in env.items():
        monkeypatch.setenv(f"DATABEACON_{k}", str(v))
    reset_settings_cache()
    try:
        settings = get_settings()
        with TestClient(create_app(RunManager(registry, settings, llm_factory=llm_factory))) as client:
            yield client
    finally:
        for k in ["PUBLIC_MODE", *env]:
            monkeypatch.delenv(f"DATABEACON_{k}", raising=False)
        reset_settings_cache()


def _csv(rows=50) -> bytes:
    return pd.DataFrame({"g": ["x", "y"] * (rows // 2), "v": range(rows)}).to_csv(index=False).encode()


def _wait(client, run_id, headers):
    for _ in range(200):
        r = client.get(f"/api/runs/{run_id}", headers=headers).json()
        if r["status"] != "running":
            return r
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_session_required_and_headers(registry, monkeypatch):
    with public_client(registry, monkeypatch) as c:
        assert c.get("/api/datasets").status_code == 401
        assert c.get("/api/datasets", headers={"X-DataBeacon-Session": "short"}).status_code == 400
        r = c.get("/api/datasets", headers=A)
        assert r.status_code == 200 and r.json() == []
        assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["Cache-Control"] == "no-store"
        assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404
        h = c.get("/api/health").json()
        assert h["public_mode"] and h["limits"]["max_upload_mb"] == 4


def test_visitors_are_isolated(registry, monkeypatch):
    with public_client(registry, monkeypatch) as c:
        ds = c.post("/api/datasets/sample", headers=A).json()
        assert "owner" not in ds
        assert c.post("/api/datasets/sample", headers=A).json()["id"] == ds["id"]  # one sample per session
        run_id = c.post("/api/runs", headers=A, json={"dataset_id": ds["id"],
                                                      "question": "Why did revenue decline last quarter?"}).json()["run_id"]
        assert _wait(c, run_id, A)["status"] == "completed"

        assert c.get("/api/datasets", headers=B).json() == []
        for method, path in [("get", f"/api/datasets/{ds['id']}"), ("get", f"/api/datasets/{ds['id']}/profile"),
                             ("delete", f"/api/datasets/{ds['id']}"), ("get", f"/api/runs/{run_id}"),
                             ("get", f"/api/runs/{run_id}/events"), ("get", f"/api/runs/{run_id}/report.pdf")]:
            assert getattr(c, method)(path, headers=B).status_code == 404, path
        assert c.post(f"/api/datasets/{ds['id']}/sql", headers=B, json={"sql": "SELECT 1"}).status_code == 404
        assert c.post("/api/runs", headers=B, json={"dataset_id": ds["id"], "question": "steal this data"}).status_code == 404
        assert c.get("/api/runs", headers=B).json() == []
        # the owner still has everything; EventSource/download links can pass the session as a query param
        assert c.get(f"/api/runs/{run_id}/report.pdf?session={'a' * 32}").status_code == 200
        assert c.get(f"/api/runs/{run_id}", headers=A).status_code == 200 and "owner" not in c.get(
            f"/api/runs/{run_id}", headers=A).json()


def test_dataset_quota_size_and_row_caps(registry, monkeypatch):
    C, D = {"X-DataBeacon-Session": "c" * 32}, {"X-DataBeacon-Session": "d" * 32}  # fresh sessions
    with public_client(registry, monkeypatch, MAX_DATASETS_PER_SESSION=2, MAX_ROWS=100, MAX_UPLOAD_MB=0.01) as c:
        up = lambda data, h=C: c.post("/api/datasets", headers=h, files={"file": ("d.csv", io.BytesIO(data), "text/csv")})  # noqa: E731
        assert up(_csv(50)).status_code == 201
        assert up(_csv(50)).status_code == 201
        assert up(_csv(50)).status_code == 409  # per-session dataset quota
        assert up(_csv(150), D).status_code == 400 and "limit" in up(_csv(150), D).json()["detail"]  # row cap
        assert up(b"g,v\n" + b"x,1\n" * 4000, D).status_code == 413  # ~16 KB > 0.01 MB


def test_analysis_rate_limit_and_access_code(registry, monkeypatch):
    with public_client(registry, monkeypatch, RUNS_PER_HOUR=1, DEMO_ACCESS_CODE="letmein") as c:
        ds = c.post("/api/datasets/sample", headers=A).json()
        body = {"dataset_id": ds["id"], "question": "Why did revenue decline last quarter?"}
        assert c.post("/api/runs", headers=A, json=body).status_code == 403
        assert c.post("/api/runs", headers={**A, "X-DataBeacon-Access-Code": "nope"}, json=body).status_code == 403
        ok = c.post("/api/runs", headers={**A, "X-DataBeacon-Access-Code": "letmein"}, json=body)
        assert ok.status_code == 202
        _wait(c, ok.json()["run_id"], A)
        limited = c.post("/api/runs", headers={**A, "X-DataBeacon-Access-Code": "letmein"}, json=body)
        assert limited.status_code == 429 and int(limited.headers["Retry-After"]) > 0
        assert c.get("/api/health").json()["access_code_required"] is True


def test_failed_runs_do_not_leak_tracebacks(registry, monkeypatch):
    def broken():
        raise RuntimeError("secret internal path /srv/app/thing.py")

    with public_client(registry, monkeypatch, llm_factory=broken) as c:
        ds = c.post("/api/datasets/sample", headers=A).json()
        run_id = c.post("/api/runs", headers=A, json={"dataset_id": ds["id"], "question": "Why did revenue fall?"}).json()["run_id"]
        run = _wait(c, run_id, A)
        assert run["status"] == "failed"
        text = json.dumps(run)
        assert "Traceback" not in text and 'File "' not in text


def test_retention_purges_old_data(registry, monkeypatch, tmp_path):
    from databeacon.datasets import DatasetError, DatasetRegistry

    reg = DatasetRegistry(tmp_path / "ds")
    p = tmp_path / "x.csv"
    p.write_bytes(_csv(10))
    old, new = reg.ingest_file(p, owner="o"), reg.ingest_file(p, owner="o")
    meta = reg.root / old.id / "meta.json"
    m = json.loads(meta.read_text())
    m["created_at"] = (datetime.now(UTC) - timedelta(hours=30)).isoformat()
    meta.write_text(json.dumps(m))
    assert reg.purge_older_than(24) == 1
    assert [d.id for d in reg.list("o")] == [new.id]
    with pytest.raises(DatasetError):
        reg.get(old.id, "o")
