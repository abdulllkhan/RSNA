#!/usr/bin/env bash
# CoAtNet-2 Raptor-recipe throughput bench (reports/coat2_raptor.md §3). <= 10 min of GPU, under the shared box lock.
#   flock -w 14400 logs/box.lock bash dev/bench_coat2.sh            (results -> reports/coat2_bench.jsonl)
# Each config runs in its own process (clean CUDA context, clean MemAvailable baseline) with a hard timeout.
cd /home/abdulllkhan/Documents/RSNA || exit 1
source dev/mem_guard.sh
log(){ echo "[$(date '+%F %T')] $*"; }
export HF_HUB_OFFLINE=1 PYTHONPATH=/home/abdulllkhan/Documents/RSNA
OUT=${OUT:-reports/coat2_bench.jsonl}
if ps -eo cmd | grep -qE '[p]ython[^ ]* -m src\.rsna\.train_'; then log "BENCH ABORT: a trainer is running"; exit 1; fi
require_mem_gb 60 0 || { log "BENCH ABORT: MemAvailable < 60 GB"; exit 1; }
T_END=$(( $(date +%s) + ${BUDGET_S:-540} ))
run(){ # tag timeout args...
  local tag="$1" to="$2"; shift 2
  local left=$(( T_END - $(date +%s) )); [ "$left" -gt 30 ] || { log "skip $tag (10-min budget spent)"; return; }
  [ "$to" -gt "$left" ] && to=$left
  require_mem_gb 60 0 >/dev/null || { log "skip $tag: MemAvailable < 60 GB"; return; }
  log "bench $tag (timeout ${to}s): $*"
  timeout -k 10 "$to" .venv/bin/python dev/bench_coat2.py --tag "$tag" --out "$OUT" "$@" > "logs/coat2_bench_${tag}.log" 2>&1
  local rc=$?; log "  $tag rc=$rc (full log logs/coat2_bench_${tag}.log)"
  grep -E '^\{|Error|error|Traceback' "logs/coat2_bench_${tag}.log" | tail -3
  sleep 3
}
# (2026-10-12 02:46 first attempt: config A hit its 130 s timeout before printing a result, C was stopped by hand -> budgets raised, full per-config logs kept)
# Config A (b8 x K12, block ckpt, cudnn.benchmark=1) is measured by the real trainer (GPU smoke, 3 epochs) instead.
[ -n "${WITH_A:-}" ] && run A_b8k12_blockckpt 330 --mode train --batch 8 --k-scale 0.1875 --block-ckpt 1 --warmup 2 --time-s 30
run C_b4k12_nockpt      ${TC:-300} --mode train --batch 4  --k-scale 0.1875 --block-ckpt 0 --warmup 2 --time-s 30
run B_b8k12_ckpt_nobench ${TB:-150} --mode train --batch 8 --k-scale 0.1875 --block-ckpt 1 --cudnn-benchmark 0 --warmup 2 --time-s 30
run I_infer_k64_b2      ${TI:-150} --mode infer --infer-batch 2 --infer-iters 4 --cudnn-benchmark 0
[ -n "${WITH_D:-}" ] && run D_b16k12_blockckpt 330 --mode train --batch 16 --k-scale 0.1875 --block-ckpt 1 --warmup 2 --time-s 30
log "BENCH DONE ($(( $(date +%s) - T_END + ${BUDGET_S:-540} )) s)"
