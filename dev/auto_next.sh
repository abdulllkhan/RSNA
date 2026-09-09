#!/usr/bin/env bash
# Overnight decision: after the A3 queue finishes and no trainer runs, pick the next ~11-14 h GPU job.
# Gate: A3 planewise fold-0 best OOF >= 0.843 (224 baseline fold-0 0.8379 + 0.005) -> 384+plane folds 0-4;
# else plain 384 folds 1-4. Markers: DECISION | AUTO_NEXT LAUNCHED
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
while ! grep -qE 'GPU QUEUE DONE|QUEUE FAILED' logs/gpu_queue.log 2>/dev/null; do sleep 300; done
while ps -eo cmd | grep -qE '[p]ython[^ ]* -m src\.rsna\.train_(plane|hr)'; do sleep 120; done
bp=$(grep -oE 'OOF_macroAUC=[0-9.]+' logs/plane224_fold0.log 2>/dev/null | cut -d= -f2 | sort -g | tail -1)
bc=$(grep -oE 'OOF_macroAUC=[0-9.]+' logs/plane224_concat_fold0.log 2>/dev/null | cut -d= -f2 | sort -g | tail -1)
log "A3 planewise best OOF=${bp:-none} | concat best=${bc:-none} | 224 fold-0 baseline=0.8379 | gate>=0.843"
if [ -n "$bp" ] && awk -v b="$bp" 'BEGIN{exit !(b>=0.843)}'; then
  log "DECISION: plane lever WINS -> 384+plane folds 0-4 (dev/train_folds_384_plane.sh)"
  nohup bash dev/train_folds_384_plane.sh > logs/effv2s384_plane_folds.log 2>&1 &
else
  log "DECISION: plane lever below gate -> plain 384 folds 1-4 (dev/train_folds_384.sh)"
  nohup bash dev/train_folds_384.sh > logs/effv2s384_folds.log 2>&1 &
fi
log "AUTO_NEXT LAUNCHED PID $!"
