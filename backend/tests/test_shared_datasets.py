"""D1-backed dataset storage: reconstruction on another instance, isolation, integrity, limits."""

import json

import numpy as np
import pandas as pd
import pytest
from d1_emulator import D1Emulator, d1_store

from databeacon.datasets import DatasetError
from databeacon.storage.dataset_codec import table_fingerprint
from databeacon.storage.datasets import SharedDatasetRegistry


@pytest.fixture
def emu(tmp_path):
    return D1Emulator(tmp_path / "d1.sqlite")


def _mixed_csv(path, n=5000):
    rng = np.random.default_rng(1)
    pd.DataFrame({
        "id": np.arange(n, dtype=np.int64) * 1_000_003,
        "ts": pd.date_range("2024-01-01", periods=n, freq="7min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "amount": np.round(rng.normal(100, 30, n), 2),
        "flag": rng.random(n) < 0.5,
        "city": rng.choice(["Zürich", "東京", "naïve, café", "São Paulo"], n),
        "zip": [f"{k % 1000:05d}" for k in range(n)],
        "maybe": [None if k % 9 == 0 else k / 7 for k in range(n)],
    }).to_csv(path, index=False)
    return path


def test_dataset_rebuilds_identically_on_a_fresh_instance(tmp_path, emu, sample_csv):
    a = SharedDatasetRegistry(d1_store(emu), tmp_path / "cache_a")
    for src in (sample_csv, _mixed_csv(tmp_path / "mixed.csv")):
        ds = a.ingest_file(src, owner="o1")
        before = table_fingerprint(a.db_path(ds.id), ds.table)
        b = SharedDatasetRegistry(d1_store(emu), tmp_path / "cache_b")  # another serverless instance
        assert not (b.root / ds.id).exists()
        after = table_fingerprint(b.db_path(ds.id), ds.table)
        assert before == after and after["rows"] > 0
        assert b.get(ds.id, "o1").columns == ds.columns
        # second access uses the local cache (no chunk download)
        calls = emu.calls
        b.db_path(ds.id)
        assert emu.calls == calls + 1  # metadata lookup only


def test_owner_isolation_and_delete(tmp_path, emu, sample_csv):
    reg = SharedDatasetRegistry(d1_store(emu), tmp_path / "cache")
    ds = reg.ingest_file(sample_csv, owner="alice")
    other = SharedDatasetRegistry(d1_store(emu), tmp_path / "cache2")
    assert [d.id for d in other.list("alice")] == [ds.id] and other.list("bob") == []
    with pytest.raises(DatasetError):
        other.get(ds.id, "bob")
    with pytest.raises(DatasetError):
        other.delete(ds.id, "bob")
    other.delete(ds.id, "alice")
    with pytest.raises(DatasetError):
        reg.db_path(ds.id)
    assert emu_count(emu, "dataset_chunks") == 0


def emu_count(emu, table):
    import sqlite3

    return sqlite3.connect(emu.path).execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_half_written_upload_is_invisible(tmp_path, emu, sample_csv):
    reg = SharedDatasetRegistry(d1_store(emu), tmp_path / "cache")
    ds = reg.ingest_file(sample_csv, owner="o")
    import sqlite3

    with sqlite3.connect(emu.path) as c:
        c.execute("UPDATE datasets SET ready = 0 WHERE id = ?", (ds.id,))
    fresh = SharedDatasetRegistry(d1_store(emu), tmp_path / "cache2")
    assert fresh.list("o") == []
    with pytest.raises(DatasetError):
        fresh.db_path(ds.id)


def test_corrupted_or_missing_chunks_are_detected(tmp_path, emu, sample_csv):
    import sqlite3

    reg = SharedDatasetRegistry(d1_store(emu), tmp_path / "cache")
    ds = reg.ingest_file(sample_csv, owner="o")
    with sqlite3.connect(emu.path) as c:
        data = c.execute("SELECT data FROM dataset_chunks WHERE dataset_id = ?", (ds.id,)).fetchone()[0]
        c.execute("UPDATE dataset_chunks SET data = ? WHERE dataset_id = ?", ("AAAA" + data[4:], ds.id))
    with pytest.raises(DatasetError, match="checksum"):
        SharedDatasetRegistry(d1_store(emu), tmp_path / "c2").db_path(ds.id)
    with sqlite3.connect(emu.path) as c:
        c.execute("DELETE FROM dataset_chunks WHERE dataset_id = ?", (ds.id,))
    with pytest.raises(DatasetError, match="incomplete"):
        SharedDatasetRegistry(d1_store(emu), tmp_path / "c3").db_path(ds.id)


def test_chunks_shrink_when_the_store_rejects_row_size(tmp_path, sample_csv):
    emu = D1Emulator(tmp_path / "small.sqlite", max_row_bytes=120_000)  # stricter than D1's 2 MB
    reg = SharedDatasetRegistry(d1_store(emu), tmp_path / "cache")
    ds = reg.ingest_file(sample_csv, owner="o")
    import sqlite3

    sizes = [r[0] for r in sqlite3.connect(emu.path).execute("SELECT length(data) FROM dataset_chunks")]
    assert len(sizes) > 1 and max(sizes) <= 120_000
    fresh = SharedDatasetRegistry(d1_store(emu), tmp_path / "cache2")
    assert table_fingerprint(fresh.db_path(ds.id), ds.table) == table_fingerprint(reg.db_path(ds.id), ds.table)


def test_cache_limit_and_purge(tmp_path, emu, sample_csv, monkeypatch):
    import databeacon.storage.datasets as mod

    monkeypatch.setattr(mod, "CACHE_LIMIT_MB", 0.0001)  # everything but the dataset in use is evicted
    reg = SharedDatasetRegistry(d1_store(emu), tmp_path / "cache")
    a, b = reg.ingest_file(sample_csv, owner="o"), reg.ingest_file(sample_csv, owner="o")
    fresh = SharedDatasetRegistry(d1_store(emu), tmp_path / "cache2")
    fresh.db_path(a.id)
    fresh.db_path(b.id)
    assert not (fresh.root / a.id).exists() and (fresh.root / b.id).exists()
    import sqlite3

    with sqlite3.connect(emu.path) as c:
        c.execute("UPDATE datasets SET created_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (a.id,))
    assert reg.purge_older_than(24) == 1
    assert [d.id for d in reg.list("o")] == [b.id]
    assert json.loads(json.dumps(reg.get(b.id).to_dict()))["owner"] == "o"
