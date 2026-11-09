#!/usr/bin/env bash
# Queue PART 13 (Sep 24): WIDE 170 mm v2 arm as a replacement for the old-contract hr arm. Waits for the c170 cache,
# trains fold 0 (next_v2_cmd13), gates on the fold-0 hr swap (replace >= +0.002 or add >= +0.003 steven), PASS -> folds 1-4.
# Markers: QUEUE_V2N GATE | QUEUE_V2N PASS | QUEUE_V2N FAIL | QUEUE_V2N DONE
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
NAME=v2_effv2s384_c170; CMD_FILE=dev/kaggle/next_v2_cmd13
until grep -q 'C170_384 CACHE DONE' logs/decode_v2_c170_384.log 2>/dev/null; do sleep 120; done
log "QUEUE_V2N: c170 cache ready -> fold 0 of $NAME (~4.5 h)"
CMD_FILE=$CMD_FILE bash dev/train_folds_v2.sh 1 "0" > logs/${NAME}_fold0_queue.log 2>&1
log "QUEUE_V2N: $(tail -1 logs/${NAME}_fold0_queue.log)"
best=$(grep -oE '^\[done\].*best OOF_macroAUC=[0-9.]+' logs/${NAME}_fold0.log 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)
[ -n "$best" ] && [ -f data/oof/${NAME}_fold0.parquet ] || { log "QUEUE_V2N FAIL: no fold-0 result"; exit 1; }
PYTHONPATH=. .venv/bin/python dev/fold0_hr_swap.py data/oof/${NAME}_fold0.parquet 2>&1 | grep -v Warning | sed 's/^/[hrswap] /' | tee /tmp/hrswap_c170.txt
rep=$(grep -oE 'REPLACE_DELTA=-?[0-9.]+' /tmp/hrswap_c170.txt | cut -d= -f2); add=$(grep -oE 'ADD_DELTA=-?[0-9.]+' /tmp/hrswap_c170.txt | cut -d= -f2)
log "QUEUE_V2N GATE: fold-0 best OOF=$best; replace-hr delta=$rep add delta=$add (bars +0.002 / +0.003)"
if python3 -c "import sys; sys.exit(0 if (float('$rep')>=0.002 or float('$add')>=0.003) else 1)"; then
  log "QUEUE_V2N PASS -> folds 1-4 of $NAME"; CMD_FILE=$CMD_FILE bash dev/train_folds_v2.sh 1 "1 2 3 4" > logs/${NAME}_folds1-4.log 2>&1
  log "QUEUE_V2N: $(tail -1 logs/${NAME}_folds1-4.log)"; log "QUEUE_V2N DONE"
else log "QUEUE_V2N FAIL: gate not met (replace $rep / add $add); GPU left idle"; fi
