"""Token profile of an agent run, by stage and prompt component.

    python -m evals.token_profile                 # scripted (offline) run, no API usage
    python -m evals.token_profile --run-id <id>   # profile a stored run (needs DATABEACON_DATA_DIR)

Each LLM call records the size of its prompt components (system prompt, tool schemas, tool results,
tool-call history, other messages). Sizes are characters; ~4 characters per token is used for estimates.
Providers count tool schemas and chat formatting slightly differently, so treat these as estimates and
use the provider-reported totals (``tokens``) for billing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COMPONENTS = ("system", "tool_schemas", "tool_results", "tool_calls", "other")


def profile(timeline: list[dict]) -> dict:
    stages: dict[str, dict] = defaultdict(lambda: {"calls": 0, "reported_in": 0, "reported_out": 0,
                                                   **{c: 0 for c in COMPONENTS}})
    for e in timeline:
        if e.get("type") != "llm":
            continue
        st = stages[e.get("stage") or e["name"].split(" ")[0]]
        st["calls"] += 1
        tok = e.get("tokens") or {}
        st["reported_in"] += tok.get("input_tokens", 0)
        st["reported_out"] += tok.get("output_tokens", 0)
        for c in COMPONENTS:
            st[c] += (e.get("prompt_chars") or {}).get(c, 0)
    total_chars = sum(st[c] for st in stages.values() for c in COMPONENTS) or 1
    out = {}
    for name, st in stages.items():
        chars = sum(st[c] for c in COMPONENTS)
        out[name] = {"calls": st["calls"], "est_prompt_tokens": chars // 4,
                     "share_of_prompt": round(chars / total_chars, 3),
                     "components_est_tokens": {c: st[c] // 4 for c in COMPONENTS},
                     "reported_input_tokens": st["reported_in"], "reported_output_tokens": st["reported_out"]}
    comp_tot = {c: sum(st[c] for st in stages.values()) // 4 for c in COMPONENTS}
    return {"stages": out, "components_est_tokens": comp_tot, "est_prompt_tokens": total_chars // 4}


async def _scripted() -> dict:
    os.environ.setdefault("DATABEACON_DATA_DIR", tempfile.mkdtemp(prefix="databeacon_tokprof_"))
    from databeacon.agent.testing import scripted_ecommerce_llm
    from databeacon.config import get_settings
    from databeacon.datasets import DatasetRegistry
    from databeacon.runs import RunManager
    from sample_data.generate_ecommerce import generate

    csv = Path(tempfile.mkdtemp()) / "ecommerce_orders.csv"
    generate().to_csv(csv, index=False)
    reg = DatasetRegistry(get_settings().datasets_dir)
    ds = reg.ingest_file(csv)
    m = RunManager(reg, llm_factory=lambda: scripted_ecommerce_llm(hallucinate_first=False))
    run = await m.run_sync(ds.id, "Why did our revenue decline last quarter, and which customer segments contributed most?")
    return run["result"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    args = ap.parse_args()
    if args.run_id:
        from databeacon.config import get_settings
        from databeacon.runs import RunStore

        result = RunStore(get_settings().runs_db).get(args.run_id)["result"]
    else:
        result = asyncio.run(_scripted())
    rep = profile(result["timeline"])
    rep["tool_calls"] = result["metrics"]["tool_calls"]
    rep["duplicate_calls_blocked"] = result["metrics"].get("duplicate_calls_blocked")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
