#!/usr/bin/env bash
# Post-folds autonomous runbook (run under nohup once "ALL FOLDS 1-4 DONE"):
#   1) submission #2 = 5-fold 224 ensemble (dev/package_and_push.sh -> self-gated submit)
#   2) guarded raw DICOM download (dev/fetch_raw.sh)          [network/IO, GPU idle]
#   3) decode raw -> data/cache384 + data/cache512 (CPU)       [never overlaps GPU training]
#   4) sampler-identity gate (dev/hr_sampler_proof.py)
#   5) wait until the GPU is free, then launch 384px FOLD-0 ONLY (the resolution-gain test)
# Stage markers (grep these): SUB2 LAUNCHED | RAW DONE | DECODE DONE | SAMPLER PASS | HR FOLD0 LAUNCHED | RUNBOOK FAILED
set -uo pipefail
cd /home/abdulllkhan/Documents/RSNA || exit 1
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
log() { echo "[$(date '+%F %T')] $*"; }
fail() { log "RUNBOOK FAILED: $*"; exit 1; }

log "post_folds start"
ps -eo cmd | grep -qE '[p]ython[^ ]* -m src\.rsna\.train' && fail "a trainer is still running"
grep -q 'ALL FOLDS 1-4 DONE' logs/effv2s_folds1-4.log 2>/dev/null || fail "folds not marked done"

# ---- 1) submission #2: 5-fold 224 ensemble ------------------------------------------------
log "== step 1: package + push 5-fold ensemble (sub #2) =="
if [ "${SKIP_SUB2:-0}" = "1" ]; then log "step 1 skipped (sub #2 already launched, kernel v$(cat dev/kaggle/last_kernel_version 2>/dev/null))"; else
bash dev/package_and_push.sh "5-fold effv2s_k24 ensemble (224px, DICOM infer)" || fail "package_and_push failed"
log "SUB2 LAUNCHED (driver: logs/submit1.log; version: $(cat dev/kaggle/last_kernel_version 2>/dev/null))"
fi

# ---- 2) raw DICOM download (guarded) ------------------------------------------------------
log "== step 2: guarded raw download =="
bash dev/fetch_raw.sh || fail "fetch_raw refused or failed"
[ -d data/raw/train_series ] || fail "data/raw/train_series missing after fetch"
log "RAW DONE ($(find data/raw/train_series -name '*.dcm' | wc -l) train dcm)"

# ---- 3) decode -> 384 + 512 caches (CPU only; GPU is idle here) ----------------------------
log "== step 3: decode raw -> cache384 + cache512 =="
.venv/bin/python dev/decode_raw.py data/raw --out data --sizes 384 512 --workers 10 || fail "decode_raw failed"
n384=$(find data/cache384 -name '*.jpg' | wc -l); n512=$(find data/cache512 -name '*.jpg' | wc -l)
log "DECODE DONE (cache384=$n384 jpg, cache512=$n512 jpg)"
[ "$n384" -gt 800000 ] || fail "cache384 looks incomplete ($n384)"

# ---- 4) sampler-identity gate ---------------------------------------------------------------
log "== step 4: sampler identity gate =="
.venv/bin/python dev/hr_sampler_proof.py --cache384 data/cache384 --jpeg224 data/processed_jpeg224 || fail "hr_sampler_proof FAILED"
log "SAMPLER PASS"

# ---- 5) wait for a free GPU, then 384 fold-0 only -----------------------------------------------
log "== step 5: waiting for a free GPU (no compute apps) =="
for i in $(seq 1 240); do
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  [ "$busy" -eq 0 ] && break
  sleep 60
done
[ "$busy" -eq 0 ] || fail "GPU still busy after 4h"
mkdir -p logs
nohup .venv/bin/python -m src.rsna.train_hr --fold 0 --size 384 --k 24 \
    --batch 6 --grad-accum 3 --workers 6 --epochs 10 --lr 3e-4 \
    --cache-root data/cache384 --name effv2s_k24_384 --max-train-minutes 170 \
    > logs/effv2s384_fold0.log 2>&1 &
log "HR FOLD0 LAUNCHED (PID $!, log logs/effv2s384_fold0.log)"
log "post_folds end"
