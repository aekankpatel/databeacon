"""Round-trip validation of the dataset codec (DuckDB -> zstd Parquet -> D1 chunks -> DuckDB).

    python evals/validate_dataset_codec.py <work dir>

Builds realistic datasets (the e-commerce sample, mixed types with time zones, Unicode, NULLs and
leading-zero codes, a wide 100-column table, a file at the 4 MB public upload cap, nested JSON) and
checks that schema, row count and an order-independent content hash are identical after the trip.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from databeacon.datasets import DatasetRegistry  # noqa: E402
from databeacon.storage.dataset_codec import (  # noqa: E402
    join_chunks,
    parquet_to_duckdb,
    sha256,
    split_chunks,
    table_fingerprint,
    table_to_parquet,
)


def make_datasets(out: Path) -> dict[str, Path]:
    rng = np.random.default_rng(7)
    files = {"ecommerce_orders.csv": ROOT / "sample_data" / "ecommerce_orders.csv"}
    n = 30_000
    mixed = pd.DataFrame({
        "id": np.arange(n, dtype=np.int64) * 1_000_003,
        "big_int": rng.integers(-2**62, 2**62, n),
        "created_at": pd.date_range("2023-01-01", periods=n, freq="17min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "day": pd.date_range("2020-01-01", periods=n, freq="h").date.astype(str),
        "amount": np.round(rng.normal(250, 120, n), 2),
        "ratio": rng.random(n),
        "flag": rng.random(n) < 0.3,
        "city": rng.choice(["São Paulo", "Zürich", "東京", "Kraków", "New York", "Ελλάδα", "naïve, café"], n),
        "note": [("long text " * int(k % 40)) + f"#{k}" if k % 5 else None for k in range(n)],
        "maybe_num": [None if k % 7 == 0 else float(k) / 3 for k in range(n)],
        "zip": [f"{k % 100000:05d}" for k in range(n)],  # leading zeros must survive
    })
    files["mixed_types.csv"] = out / "mixed_types.csv"
    mixed.to_csv(files["mixed_types.csv"], index=False)

    wide = pd.DataFrame(rng.normal(size=(20_000, 98)), columns=[f"m{i}" for i in range(98)])
    wide.insert(0, "segment", rng.choice(list("ABCDEFG"), 20_000))
    wide.insert(0, "dt", pd.date_range("2024-01-01", periods=20_000, freq="30min"))
    files["wide_100cols.csv"] = out / "wide_100cols.csv"
    wide.to_csv(files["wide_100cols.csv"], index=False)

    rows = 120_000  # close to the 4 MB public upload cap
    big = pd.DataFrame({"ts": pd.date_range("2021-01-01", periods=rows, freq="5min"),
                        "sensor": rng.choice(["s1", "s2", "s3", "s4"], rows),
                        "value": np.round(rng.normal(20, 5, rows), 3),
                        "ok": rng.random(rows) < 0.99})
    files["telemetry_4mb.csv"] = out / "telemetry_4mb.csv"
    big.to_csv(files["telemetry_4mb.csv"], index=False)

    nested = [{"order": k, "customer": {"id": k % 97, "tier": ["gold", "silver"][k % 2]},
               "items": [{"sku": f"S{j}", "qty": j} for j in range(k % 4)],
               "paid_at": f"2025-0{1 + k % 9}-1{k % 9}T10:00:00"} for k in range(8000)]
    files["nested.json"] = out / "nested.json"
    files["nested.json"].write_text(json.dumps(nested))
    return files


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
    out.mkdir(parents=True, exist_ok=True)
    reg = DatasetRegistry(Path(tempfile.mkdtemp()))
    print(f"{'dataset':22} {'rows':>8} {'cols':>4} {'file':>8} {'parquet':>8} {'chunks':>6} "
          f"{'export':>7} {'rebuild':>7}  identical")
    ok = True
    for name, path in make_datasets(out).items():
        ds = reg.ingest_file(path)
        db = reg.db_path(ds.id)
        before = table_fingerprint(db, ds.table)
        t0 = time.perf_counter()
        pq = table_to_parquet(db, ds.table)
        t1 = time.perf_counter()
        chunks = split_chunks(pq)
        data = join_chunks(chunks, sha256(pq), len(pq))
        rebuilt = Path(tempfile.mkdtemp()) / "rebuilt.duckdb"
        t2 = time.perf_counter()
        parquet_to_duckdb(data, rebuilt, ds.table)
        t3 = time.perf_counter()
        after = table_fingerprint(rebuilt, ds.table)
        same = before == after
        ok &= same
        print(f"{name:22} {before['rows']:>8} {len(before['schema']):>4} {path.stat().st_size / 1e6:>7.2f}M "
              f"{len(pq) / 1e6:>7.2f}M {len(chunks):>6} {1000 * (t1 - t0):>6.0f}ms {1000 * (t3 - t2):>6.0f}ms  {same}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
