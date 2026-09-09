#!/usr/bin/env bash
# A5 Synovitis relabel, unattended: fetch model (guarded) -> llama-server :8081 -> health
# -> 100-report probe -> full resumable run -> score vs 58 gold -> STOP server (frees GPU).
# Markers: MODEL READY | SERVER HEALTHY | PROBE DONE | FULL DONE | SCORE DONE | A5 FAILED
set -uo pipefail
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
SRV_PID=""
fail(){ log "A5 FAILED: $*"; [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null; exit 1; }
MODELS_DIR=/home/abdulllkhan/Documents/qwen-models/models
LLAMA=/home/abdulllkhan/Documents/qwen-models/llama.cpp/build/bin/llama-server
GGUF_NAME=google_gemma-3-27b-it-Q4_K_M.gguf
PY=.venv/bin/python; BASE=http://127.0.0.1:8081/v1; ALIAS=gemma-3-27b-it
log "A5 start"
GGUF=$(find "$MODELS_DIR" -name "$GGUF_NAME" 2>/dev/null | head -1)
if [ -z "$GGUF" ]; then
  bash dev/a5/fetch_model.sh --model gemma3-27b || fail "fetch_model refused/failed"
  GGUF=$(find "$MODELS_DIR" -name "$GGUF_NAME" 2>/dev/null | head -1)
fi
[ -n "$GGUF" ] && [ -f "$GGUF" ] || fail "gguf not found after fetch"
log "MODEL READY $GGUF ($(du -h "$GGUF" | cut -f1))"
mkdir -p logs data/a5
nohup "$LLAMA" --model "$GGUF" --alias "$ALIAS" --host 127.0.0.1 --port 8081 --n-gpu-layers 99 \
  --ctx-size 16384 --parallel 4 --cache-reuse 256 --jinja --flash-attn on > logs/a5_server.log 2>&1 &
SRV_PID=$!
log "server PID $SRV_PID; waiting for /health"
for i in $(seq 1 180); do curl -sf http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 5; done
curl -sf http://127.0.0.1:8081/health >/dev/null 2>&1 || fail "server never healthy (logs/a5_server.log)"
log "SERVER HEALTHY"
log "== probe: 100 reports, gold-first =="
$PY dev/a5/run_relabel.py --base-url "$BASE" --model "$ALIAS" --parallel 4 --no-system \
   --limit 100 --order gold-first --out-prefix data/a5/synovitis_llm_probe 2>&1 | tail -6 || fail "probe failed"
log "PROBE DONE"
log "== full run (resumable, gold-first) =="
$PY dev/a5/run_relabel.py --base-url "$BASE" --model "$ALIAS" --parallel 4 --no-system \
   --order gold-first 2>&1 | tail -8 || fail "full run failed"
log "FULL DONE"
log "== score vs 58 gold =="
$PY dev/a5/score_synovitis.py --llm-csv data/a5/synovitis_llm.csv --top-disagree 15 2>&1 | tee logs/a5_score.log | tail -45
log "SCORE DONE (logs/a5_score.log)"
kill "$SRV_PID" 2>/dev/null; sleep 5; kill -9 "$SRV_PID" 2>/dev/null || true
log "server stopped; GPU freed; A5 end"
