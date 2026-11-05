import time

from fastapi.testclient import TestClient

from databeacon.agent.testing import scripted_ecommerce_llm


def _client(registry):
    from databeacon.api.main import create_app
    from databeacon.runs import RunManager

    return TestClient(create_app(RunManager(registry, llm_factory=scripted_ecommerce_llm)))


def test_end_to_end_api(registry, sample_csv):
    with _client(registry) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        with sample_csv.open("rb") as fh:
            r = client.post("/api/datasets", files={"file": ("orders.csv", fh, "text/csv")}, data={"name": "Orders"})
        assert r.status_code == 201, r.text
        ds = r.json()
        assert ds["table"] == "orders" and ds["row_count"] > 10000

        prof = client.get(f"/api/datasets/{ds['id']}/profile").json()
        assert prof["row_count"] == ds["row_count"]
        assert client.post(f"/api/datasets/{ds['id']}/sql", json={"sql": "DROP TABLE orders"}).status_code == 400
        ok = client.post(f"/api/datasets/{ds['id']}/sql", json={"sql": "SELECT COUNT(*) AS n FROM orders"})
        assert ok.json()["rows"][0][0] == ds["row_count"]

        cat = client.get("/api/mcp/catalog").json()
        assert len(cat["tools"]) >= 12

        r = client.post("/api/runs", json={"dataset_id": ds["id"], "question": "Why did revenue decline last quarter?"})
        assert r.status_code == 202
        run_id = r.json()["run_id"]
        for _ in range(100):
            run = client.get(f"/api/runs/{run_id}").json()
            if run["status"] != "running":
                break
            time.sleep(0.1)
        assert run["status"] == "completed", run
        events = client.get(f"/api/runs/{run_id}/events").text
        assert '"kind": "done"' in events
        pdf = client.get(f"/api/runs/{run_id}/report.pdf")
        assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
        assert client.get("/api/runs", params={"dataset_id": ds["id"]}).json()[0]["id"] == run_id


def test_upload_validation(registry, tmp_path):
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"MZ")
    with _client(registry) as client:
        with bad.open("rb") as fh:
            r = client.post("/api/datasets", files={"file": ("x.exe", fh)})
        assert r.status_code == 400
        assert client.post("/api/runs", json={"dataset_id": "000000000000", "question": "hello there"}).status_code == 404
        assert client.get("/api/runs/zzz/report.pdf").status_code == 400
