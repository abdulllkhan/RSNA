#!/usr/bin/env bash
# A5 smoke test — the whole chain (runner -> csv -> scorer) against the MOCK
# endpoint. No GPU, no model weights, no network. Safe to run while a fold trains.
#
#   dev/a5/smoke_mock.sh [N_REPORTS]     (default 60)
#
# Writes data/a5/synovitis_llm_mock.{jsonl,csv}; leaves the real
# data/a5/synovitis_llm.* untouched.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
N="${1:-60}"
PORT="${A5_MOCK_PORT:-8099}"
PREFIX="$REPO/data/a5/synovitis_llm_mock"
PIDFILE="$(mktemp -t a5mock.XXXXXX.pid)"

cd "$REPO"

cleanup() {
  if [[ -f "$PIDFILE" ]]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
}
trap cleanup EXIT

echo "== 0. reports table =="
[[ -f data/a5/reports.parquet ]] || nice -n 19 "$PY" dev/a5/extract_reports.py >/dev/null

echo "== 1. start mock endpoint on :$PORT =="
nice -n 19 "$PY" dev/a5/mock_server.py --port "$PORT" --messy-rate 0.20 --fail-rate 0.05 \
  > "$REPO/data/a5/mock_server.log" 2>&1 &
echo $! > "$PIDFILE"
for _ in $(seq 1 30); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  sleep 0.3
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "mock failed to start"; cat "$REPO/data/a5/mock_server.log"; exit 1; }

echo "== 2. run_relabel over the gold studies first (so the scorer has an anchor) =="
nice -n 19 "$PY" dev/a5/run_relabel.py \
  --base-url "http://127.0.0.1:$PORT/v1" --model mock \
  --limit "$N" --parallel 4 --order gold-first \
  --out-prefix "$PREFIX" --no-resume --log-every 25

echo "== 3. resume check (re-run must be a no-op) =="
nice -n 19 "$PY" dev/a5/run_relabel.py \
  --base-url "http://127.0.0.1:$PORT/v1" --model mock \
  --limit "$N" --parallel 4 --order gold-first --out-prefix "$PREFIX" | tail -3

echo "== 4. score =="
nice -n 19 "$PY" dev/a5/score_synovitis.py --llm-csv "$PREFIX.csv" --top-disagree 5

echo
echo "SMOKE OK — mock chain ran end to end."
