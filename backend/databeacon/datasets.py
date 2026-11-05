"""Dataset registry: ingest uploaded files into per-dataset DuckDB databases."""

from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".parquet", ".json", ".jsonl", ".ndjson", ".xlsx"}


class DatasetError(ValueError):
    pass


def sanitize_identifier(name: str, fallback: str = "data") -> str:
    ident = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()
    if not ident:
        ident = fallback
    if ident[0].isdigit():
        ident = f"t_{ident}"
    return ident[:63]


@dataclass
class Dataset:
    id: str
    name: str
    table: str
    filename: str
    created_at: str
    row_count: int
    column_count: int
    columns: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetRegistry:
    """Stores each dataset in ``<datasets_dir>/<id>/`` with a DuckDB file and metadata."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ paths
    def _dir(self, dataset_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", dataset_id or ""):
            raise DatasetError(f"Invalid dataset id: {dataset_id!r}")
        return self.root / dataset_id

    def db_path(self, dataset_id: str) -> Path:
        path = self._dir(dataset_id) / "data.duckdb"
        if not path.exists():
            raise DatasetError(f"Dataset {dataset_id} not found")
        return path

    # ----------------------------------------------------------------- ingest
    def ingest_file(self, source: Path, *, name: str | None = None, original_filename: str | None = None) -> Dataset:
        source = Path(source)
        filename = original_filename or source.name
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise DatasetError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        dataset_id = uuid.uuid4().hex[:12]
        ddir = self.root / dataset_id
        ddir.mkdir(parents=True)
        table = sanitize_identifier(Path(filename).stem)
        db_file = ddir / "data.duckdb"
        try:
            con = duckdb.connect(str(db_file))
            try:
                self._load(con, source, ext, table)
                row_count = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                cols = con.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_name = ? ORDER BY ordinal_position",
                    [table],
                ).fetchall()
            finally:
                con.close()
        except Exception as exc:
            shutil.rmtree(ddir, ignore_errors=True)
            if isinstance(exc, DatasetError):
                raise
            raise DatasetError(f"Could not load file: {exc}") from exc
        if row_count == 0:
            shutil.rmtree(ddir, ignore_errors=True)
            raise DatasetError("The file contains no rows.")
        ds = Dataset(
            id=dataset_id,
            name=name or Path(filename).stem,
            table=table,
            filename=filename,
            created_at=datetime.now(timezone.utc).isoformat(),
            row_count=int(row_count),
            column_count=len(cols),
            columns=[{"name": c, "type": t} for c, t in cols],
        )
        (ddir / "meta.json").write_text(json.dumps(ds.to_dict(), indent=2))
        return ds

    @staticmethod
    def _load(con: duckdb.DuckDBPyConnection, source: Path, ext: str, table: str) -> None:
        src = str(source)
        if ext in {".csv", ".tsv"}:
            con.execute(
                f'CREATE TABLE "{table}" AS SELECT * FROM read_csv_auto(?, sample_size=-1, normalize_names=true)', [src]
            )
        elif ext == ".parquet":
            con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM read_parquet(?)', [src])
        elif ext in {".json", ".jsonl", ".ndjson"}:
            con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM read_json_auto(?)', [src])
        elif ext == ".xlsx":
            try:
                import pandas as pd

                df = pd.read_excel(src)
            except ImportError as exc:  # pragma: no cover
                raise DatasetError("Reading .xlsx requires openpyxl") from exc
            df.columns = [sanitize_identifier(str(c), f"col_{i}") for i, c in enumerate(df.columns)]
            con.register("_upload_df", df)
            con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM _upload_df')
            con.unregister("_upload_df")

    # ------------------------------------------------------------------ query
    def get(self, dataset_id: str) -> Dataset:
        meta = self._dir(dataset_id) / "meta.json"
        if not meta.exists():
            raise DatasetError(f"Dataset {dataset_id} not found")
        return Dataset(**json.loads(meta.read_text()))

    def list(self) -> list[Dataset]:
        out = []
        for meta in self.root.glob("*/meta.json"):
            try:
                out.append(Dataset(**json.loads(meta.read_text())))
            except Exception:
                continue
        return sorted(out, key=lambda d: d.created_at, reverse=True)

    def delete(self, dataset_id: str) -> None:
        ddir = self._dir(dataset_id)
        if not ddir.exists():
            raise DatasetError(f"Dataset {dataset_id} not found")
        with self._lock:
            shutil.rmtree(ddir)
