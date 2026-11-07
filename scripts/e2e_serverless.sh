#!/usr/bin/env bash
# Browser test of the serverless execution model with two backend "instances" sharing one store:
# replay mode, a live inline run, reconnecting from another instance, and an interrupted run
# (instance A is killed mid-run). Offline: uses the scripted LLM, no API keys or neurons.
#   scripts/e2e_serverless.sh            (needs: built frontend, Python deps, Playwright)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
export DATABEACON_DATA_DIR=$WORK/data DATABEACON_PUBLIC_MODE=1 DATABEACON_EXECUTION=inline \
  DATABEACON_RUN_STALE_S=6 DATABEACON_LLM_PROVIDER=scripted DATABEACON_SCRIPTED_DELAY_S=0.6 \
  DATABEACON_RUNS_PER_HOUR=20 DATABEACON_RUNS_PER_DAY=50
cd "$ROOT/backend"
uvicorn databeacon.api.main:app --port 8000 > "$WORK/a.log" 2>&1 & A=$!
uvicorn databeacon.api.main:app --port 8001 > "$WORK/b.log" 2>&1 & B=$!
trap 'kill $A $B 2>/dev/null || true' EXIT
for p in 8000 8001; do until curl -sf "localhost:$p/api/health" >/dev/null; do sleep 0.5; done; done
E2E_OUT=$WORK E2E_KILL_A="kill -9 $A" node "$ROOT/scripts/e2e_serverless.js"
echo "screenshots and logs: $WORK"
