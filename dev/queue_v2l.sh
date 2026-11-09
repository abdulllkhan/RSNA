#!/usr/bin/env bash
# Queue PART 12 (2026-09-21): distil the SECOND v2 arm (140 mm crop, `v2_effv2s384_lr` recipe) on the same distilled
# labels. Fold 0 -> gate vs the lr fold-0 baseline (0.8794, +0.005) -> fold-0 member swap -> on PASS folds 1-4.
#   nohup bash dev/queue_v2l.sh >> logs/queue_v2.log 2>&1 &
# Markers: QUEUE_V2L GATE | QUEUE_V2L PASS | QUEUE_V2L FAIL | QUEUE_V2L DONE
# Swap mapping (fold0_member_swap.py swaps X for its "c110" slot): --c140 slot = c110_distill (fixed),
# --c110 slot = lr (the member being replaced), X = lr_distill. So "3-arm ref" = hr + c110_distill + lr,
# "X=new" = hr + c110_distill + lr_distill, "4-arm" = all four.
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
NAME=v2_effv2s384_lr_distill; REF=0.8794; BAR=0.005; CMD_FILE=dev/kaggle/next_v2_cmd9
[ -f "$CMD_FILE" ] || { log "QUEUE_V2L FAIL: no $CMD_FILE"; exit 1; }
log "QUEUE_V2L: fold 0 of $NAME (~4.5 h)"
CMD_FILE=$CMD_FILE bash dev/train_folds_v2.sh 1 "0" > logs/${NAME}_fold0_queue.log 2>&1
log "QUEUE_V2L: $(tail -1 logs/${NAME}_fold0_queue.log)"
best=$(grep -oE '^\[done\].*best OOF_macroAUC=[0-9.]+' logs/${NAME}_fold0.log 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)
[ -n "$best" ] || { log "QUEUE_V2L FAIL: no [done] line in logs/${NAME}_fold0.log"; exit 1; }
[ -f models/${NAME}_fold0.pt ] || { log "QUEUE_V2L FAIL: no checkpoint"; exit 1; }
log "QUEUE_V2L GATE: $NAME fold-0 best OOF=$best vs lr fold-0 $REF (bar +$BAR); NOTE: distilled OOF is optimistic (teacher leakage)"
if [ -f data/oof/${NAME}_fold0.parquet ]; then
  PYTHONPATH=. .venv/bin/python dev/fold0_member_swap.py data/oof/${NAME}_fold0.parquet --c140 data/oof/v2_effv2s384_c110_distill_fold0.parquet --c110 data/oof/v2_effv2s384_lr_fold0.parquet 2>&1 | sed 's/^/[swapL] /'
fi
if python3 -c "import sys; sys.exit(0 if float('$best') >= $REF + $BAR else 1)"; then
  log "QUEUE_V2L PASS -> folds 1-4 of $NAME"
  CMD_FILE=$CMD_FILE bash dev/train_folds_v2.sh 1 "1 2 3 4" > logs/${NAME}_folds1-4.log 2>&1
  log "QUEUE_V2L: $(tail -1 logs/${NAME}_folds1-4.log)"; log "QUEUE_V2L DONE"
else
  log "QUEUE_V2L FAIL: gate not met ($best < $(python3 -c "print(round($REF+$BAR,6))")); GPU left idle for a decision"
fi
