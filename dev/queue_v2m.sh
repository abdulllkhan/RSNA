#!/usr/bin/env bash
# Queue PART 13 (2026-09-22): 2.5D SLOT CONTEXT on the 110 mm arm. Every sampled slot slice becomes the
# 3-channel stack [previous, this, next] of anatomically adjacent slices of the SAME series (clamped at the
# series ends, neighbour direction canonicalised for a left knee), replacing today's replicated grayscale.
# Fold 0 -> gate vs the c110 fold-0 baseline (0.8828, +0.005) -> fold-0 member swap -> on PASS folds 1-4.
#   nohup bash dev/queue_v2m.sh >> logs/queue_v2.log 2>&1 &
# Markers: QUEUE_V2M GATE | QUEUE_V2M PASS | QUEUE_V2M FAIL | QUEUE_V2M DONE
# Swap mapping (fold0_member_swap.py defaults): --c140 slot = v2_effv2s384_lr (fixed), --c110 slot =
# v2_effv2s384_c110 (the incumbent being replaced), X = v2_effv2s384_c110_ctx1. So "3-arm ref" = hr + c140 +
# c110, "X=new" = hr + c140 + ctx1, "4-arm" = all four.
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
NAME=v2_effv2s384_c110_ctx1; REF=0.8828; BAR=0.005; CMD_FILE=dev/kaggle/next_v2_cmd10
[ -f "$CMD_FILE" ] || { log "QUEUE_V2M FAIL: no $CMD_FILE"; exit 1; }
log "QUEUE_V2M: fold 0 of $NAME (~4.5 h)"
CMD_FILE=$CMD_FILE bash dev/train_folds_v2.sh 1 "0" > logs/${NAME}_fold0_queue.log 2>&1
log "QUEUE_V2M: $(tail -1 logs/${NAME}_fold0_queue.log)"
best=$(grep -oE '^\[done\].*best OOF_macroAUC=[0-9.]+' logs/${NAME}_fold0.log 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)
[ -n "$best" ] || { log "QUEUE_V2M FAIL: no [done] line in logs/${NAME}_fold0.log"; exit 1; }
[ -f models/${NAME}_fold0.pt ] || { log "QUEUE_V2M FAIL: no checkpoint"; exit 1; }
log "QUEUE_V2M GATE: $NAME fold-0 best OOF=$best vs c110 fold-0 $REF (bar +$BAR)"
if [ -f data/oof/${NAME}_fold0.parquet ]; then
  PYTHONPATH=. .venv/bin/python dev/fold0_member_swap.py data/oof/${NAME}_fold0.parquet 2>&1 | sed 's/^/[swapM] /'
fi
if python3 -c "import sys; sys.exit(0 if float('$best') >= $REF + $BAR else 1)"; then
  log "QUEUE_V2M PASS -> folds 1-4 of $NAME"
  CMD_FILE=$CMD_FILE bash dev/train_folds_v2.sh 1 "1 2 3 4" > logs/${NAME}_folds1-4.log 2>&1
  log "QUEUE_V2M: $(tail -1 logs/${NAME}_folds1-4.log)"; log "QUEUE_V2M DONE"
else
  log "QUEUE_V2M FAIL: gate not met ($best < $(python3 -c "print(round($REF+$BAR,6))")); GPU left idle for a decision"
fi
