import { useRef, useState } from "react";
import { api } from "../api";
import type { Dataset } from "../types";

interface Props {
  datasets: Dataset[];
  selected: string | null;
  onSelect: (id: string) => void;
  onChanged: (select?: string) => void;
  maxUploadMb?: number;
}

export function Sidebar({ datasets, selected, onSelect, onChanged, maxUploadMb }: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  async function upload(file: File) {
    if (maxUploadMb && file.size > maxUploadMb * 1024 * 1024) {
      setError(`That file is ${(file.size / 1048576).toFixed(1)} MB; the limit here is ${maxUploadMb} MB.`);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const ds = await api.upload(file);
      onChanged(ds.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function sample() {
    setBusy(true);
    setError(null);
    try {
      const ds = await api.loadSample();
      onChanged(ds.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string) {
    if (!confirm("Delete this dataset?")) return;
    await api.deleteDataset(id);
    onChanged();
  }

  return (
    <aside className="sidebar">
      <div className="section-title">Datasets</div>
      <div
        className={`dropzone ${over ? "over" : ""}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          const f = e.dataTransfer.files[0];
          if (f) upload(f);
        }}
      >
        {busy ? "Loading…" : (
          <>
            <strong>Upload a dataset</strong>
            <div className="small">CSV, TSV, Parquet, JSON or XLSX{maxUploadMb ? ` · max ${maxUploadMb} MB` : ""}</div>
          </>
        )}
        <input
          ref={input}
          type="file"
          hidden
          accept=".csv,.tsv,.parquet,.json,.jsonl,.ndjson,.xlsx"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) upload(f);
            e.target.value = "";
          }}
        />
      </div>
      <button className="btn" onClick={sample} disabled={busy}>Load sample e-commerce data</button>
      {error && <div className="error-box small">{error}</div>}
      {datasets.length === 0 && <div className="muted small">No datasets yet.</div>}
      {datasets.map((d) => (
        <div key={d.id} className={`dataset ${selected === d.id ? "active" : ""}`} onClick={() => onSelect(d.id)}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className="name">{d.name}</span>
            <button
              className="btn ghost small"
              title="Delete dataset"
              onClick={(e) => {
                e.stopPropagation();
                remove(d.id);
              }}
            >
              ✕
            </button>
          </div>
          <div className="meta">
            {d.row_count.toLocaleString()} rows · {d.column_count} cols · <span className="mono">{d.table}</span>
          </div>
        </div>
      ))}
    </aside>
  );
}
