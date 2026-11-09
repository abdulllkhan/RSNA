#!/usr/bin/env bash
# Queue PART 10 (2026-09-20): after the 512 px / 110 mm fold-0 test finishes (queue_v2i), gate its best OOF against the
# 384 px / 110 mm fold-0 (0.8828) with the +0.005 bar and, if it passes, train folds 1-4 (next_v2_cmd7, guarded).
# Markers: QUEUE_V2J GATE | QUEUE_V2J PASS | QUEUE_V2J FAIL | QUEUE_V2J DONE
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
NAME=v2_effv2s512_c110; REF=0.8828; BAR=0.005
until grep -q "V2 $NAME FOLD0 DONE" logs/queue_v2.log 2>/dev/null; do sleep 300; done
best=$(grep -oE '^\[done\].*best OOF_macroAUC=[0-9.]+' logs/${NAME}_fold0.log | grep -oE '[0-9.]+$' | tail -1)
[ -n "$best" ] || { log "QUEUE_V2J FAIL: no [done] line in logs/${NAME}_fold0.log (rc in queue_v2.log)"; exit 1; }
[ -f models/${NAME}_fold0.pt ] || { log "QUEUE_V2J FAIL: no checkpoint models/${NAME}_fold0.pt"; exit 1; }
log "QUEUE_V2J GATE: fold-0 best OOF=$best vs 384/110mm fold-0 $REF (bar +$BAR)"
if python3 -c "import sys; sys.exit(0 if float('$best') >= $REF + $BAR else 1)"; then
  log "QUEUE_V2J PASS -> folds 1-4 of $NAME (~7.4 h each)"
  CMD_FILE=dev/kaggle/next_v2_cmd7 bash dev/train_folds_v2.sh 1 "1 2 3 4" > logs/${NAME}_folds1-4.log 2>&1
  log "QUEUE_V2J DONE: $(tail -1 logs/${NAME}_folds1-4.log)"
else
  log "QUEUE_V2J FAIL: gate not met ($best < $(python3 -c "print($REF+$BAR)")); GPU left idle for a decision"
fi
