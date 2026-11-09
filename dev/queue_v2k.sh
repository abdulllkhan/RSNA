#!/usr/bin/env bash
# Queue PART 11 (2026-09-20): knowledge distillation. After the 512 px decision (queue_v2j logs PASS or FAIL),
# train fold 0 of the best arm on the DISTILLED labels (consensus blended 50/50 with the 2-arm OOF teacher,
# gold rows untouched), gate it against the c110 fold-0 baseline, print the fold-0 ensemble member swap, and
# on PASS train folds 1-4.
#   nohup bash dev/queue_v2k.sh >> logs/queue_v2.log 2>&1 &
# Markers: QUEUE_V2K GATE | QUEUE_V2K PASS | QUEUE_V2K FAIL | QUEUE_V2K DONE
# CAVEAT: the student's fold-0 OOF is slightly OPTIMISTIC — its training targets carry teacher values from
# models that saw fold-0 labels (second-order OOF-stacking leakage). The swap numbers and the LB decide.
# Test hooks (production leaves them all unset): ROOTDIR, QUEUE_LOG, FOLD0_LOG, CKPT, REF, BAR, DRY_RUN=1.
ROOTDIR="${ROOTDIR:-/home/abdulllkhan/Documents/RSNA}"
cd "$ROOTDIR" || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
NAME=v2_effv2s384_c110_distill; REF="${REF:-0.8828}"; BAR="${BAR:-0.005}"
CMD_FILE=dev/kaggle/next_v2_cmd8
QUEUE_LOG="${QUEUE_LOG:-logs/queue_v2.log}"; FOLD0_LOG="${FOLD0_LOG:-logs/${NAME}_fold0.log}"
CKPT="${CKPT:-models/${NAME}_fold0.pt}"; OOF="${OOF:-data/oof/${NAME}_fold0.parquet}"; DRY_RUN="${DRY_RUN:-0}"
[ -f "$CMD_FILE" ] || { log "QUEUE_V2K FAIL: no $CMD_FILE"; exit 1; }
[ -f data/labels/consensus_v2_distill.parquet ] || { log "QUEUE_V2K FAIL: no data/labels/consensus_v2_distill.parquet (run dev/build_distill_labels.py)"; exit 1; }
# 1) wait for the 512 px decision (queue_v2j logs exactly one of these)
until grep -qE 'QUEUE_V2J (PASS|FAIL)' "$QUEUE_LOG" 2>/dev/null; do sleep 300; done
log "QUEUE_V2K: 512 px decision seen -> $(grep -oE 'QUEUE_V2J (PASS|FAIL).*' "$QUEUE_LOG" | tail -1)"
# 2) fold 0 on the distilled labels (train_folds_v2.sh waits for a free GPU and applies the memory guard)
if [ "$DRY_RUN" = "1" ]; then
  log "QUEUE_V2K DRY_RUN: would run CMD_FILE=$CMD_FILE bash dev/train_folds_v2.sh 1 \"0\""
else
  log "QUEUE_V2K: fold 0 of $NAME (~5.5 h)"
  CMD_FILE=$CMD_FILE bash dev/train_folds_v2.sh 1 "0" > logs/${NAME}_fold0_queue.log 2>&1
  log "QUEUE_V2K: $(tail -1 logs/${NAME}_fold0_queue.log)"
fi
# 3) the gate
best=$(grep -oE '^\[done\].*best OOF_macroAUC=[0-9.]+' "$FOLD0_LOG" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)
[ -n "$best" ] || { log "QUEUE_V2K FAIL: no [done] line in $FOLD0_LOG"; exit 1; }
[ -f "$CKPT" ] || { log "QUEUE_V2K FAIL: no checkpoint $CKPT"; exit 1; }
log "QUEUE_V2K GATE: $NAME fold-0 best OOF=$best vs c110 fold-0 $REF (bar +$BAR); NOTE: distilled OOF is optimistic (teacher leakage)"
# 4) the swap that actually decides whether it ships (logged on both branches)
if [ -f "$OOF" ]; then
  PYTHONPATH="$ROOTDIR" .venv/bin/python dev/fold0_member_swap.py "$OOF" 2>&1 | sed 's/^/[swap] /'
else
  log "QUEUE_V2K: no $OOF -> member swap skipped"
fi
if python3 -c "import sys; sys.exit(0 if float('$best') >= $REF + $BAR else 1)"; then
  log "QUEUE_V2K PASS -> folds 1-4 of $NAME (~5.5 h each)"
  if [ "$DRY_RUN" = "1" ]; then
    log "QUEUE_V2K DRY_RUN: would run CMD_FILE=$CMD_FILE bash dev/train_folds_v2.sh 1 \"1 2 3 4\""
  else
    CMD_FILE=$CMD_FILE bash dev/train_folds_v2.sh 1 "1 2 3 4" > logs/${NAME}_folds1-4.log 2>&1
    log "QUEUE_V2K: $(tail -1 logs/${NAME}_folds1-4.log)"
  fi
  log "QUEUE_V2K DONE"
else
  log "QUEUE_V2K FAIL: gate not met ($best < $(python3 -c "print(round($REF+$BAR, 6))")); GPU left idle for a decision"
fi
