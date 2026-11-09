#!/usr/bin/env bash
# After the 2.5D fold-0 decision (QUEUE_V2M PASS|FAIL) and once no trainer is running: compute slot-phase TTA S=5 OOF
# frames for the two DISTILLED v2 arms (infer-only, ~80 min each) so the S=3 -> S=5 gain can be measured at the
# ensemble level before paying +1.5 h of Kaggle runtime. Markers: TTA5B START | TTA5B <arm> DONE | TTA5B DONE | TTA5B FAILED
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
source dev/mem_guard.sh
until grep -qE 'TTA5 DONE' logs/queue_v2.log 2>/dev/null; do sleep 300; done
while ps -eo cmd | grep -qE '[p]ython[^ ]* -m src\.rsna\.train_'; do sleep 300; done
require_mem_gb 30 60 || { log "TTA5B FAILED: memory"; exit 1; }
log "TTA5B START"
for spec in "v2_effv2s384_lr data/cache_c140_384" "v2_effv2s384_c110 data/cache_c110_384"; do
  set -- $spec
  PYTHONPATH=. nice -n 10 .venv/bin/python dev/tta_eval_v2.py --name $1 --cache-root $2 --variants 5 --report-s 1,3,5 --device cuda --batch 2 --workers 4 > logs/tta5b_$1.log 2>&1; rc=$?
  log "TTA5B $1 DONE rc=$rc: $(grep -E 'S=5|S=3' logs/tta5b_$1.log | tail -2 | tr '\n' ' ' | cut -c1-160)"
done
log "TTA5B DONE"
