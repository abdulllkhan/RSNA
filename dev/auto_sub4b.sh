#!/usr/bin/env bash
# Wrapper: after the 384 driver finishes folds 2-4, RESUME the undertrained fold 1 (5 more epochs),
# then run the autonomous sub #4 chain (dev/auto_sub4.sh). Same log as auto_sub4 so queue_512 works.
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
while ! grep -q 'ALL 384 FOLDS DONE' logs/effv2s384_folds.log 2>/dev/null; do sleep 300; done
while ps -eo cmd | grep -qE '[p]ython[^ ]* -m src\.rsna\.train_'; do sleep 60; done
log "driver done; RESUMING fold 1 from epoch 5 (it was capped at 5 epochs @40 img/s)"
.venv/bin/python -m src.rsna.train_hr --fold 1 --size 384 --k 24 --batch 6 --grad-accum 3 --workers 6 \
    --epochs 10 --lr 3e-4 --cache-root data/cache384 --name effv2s_k24_384 --max-train-minutes 200 --resume \
    > logs/effv2s384_fold1_resume.log 2>&1
log "FOLD 1 RESUMED DONE: $(grep -E '^\[epoch' logs/effv2s384_fold1_resume.log | tail -1)"
exec bash dev/auto_sub4.sh
