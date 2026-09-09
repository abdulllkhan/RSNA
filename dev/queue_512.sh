#!/usr/bin/env bash
# After sub #4 is submitted (so the packager's no-trainer guard has already passed), run the 512px
# fold-0 headroom test on the idle GPU. Batch scaled for 512 (bag 75 MB): batch 4, accum 4, workers 4.
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
while ! grep -qE 'SUB4 submitted|AUTO_SUB4 FAILED' logs/auto_sub4.log 2>/dev/null; do sleep 600; done
while ps -eo cmd | grep -qE '[p]ython[^ ]* -m src\.rsna\.train_'; do sleep 120; done
log "launching 512 fold-0 headroom test"
nohup bash -c 'cd /home/abdulllkhan/Documents/RSNA; while ps -eo cmd | grep -qE "[p]ython[^ ]* -m src\.rsna\.train_hr"; do bash dev/evict_tree.sh data/cache512 >/dev/null 2>&1; sleep 90; done' > logs/evict_512.log 2>&1 &
sleep 20
.venv/bin/python -m src.rsna.train_hr --fold 0 --size 512 --k 24 --batch 4 --grad-accum 4 --workers 4 \
    --epochs 10 --lr 3e-4 --cache-root data/cache512 --name effv2s_k24_512 --max-train-minutes 300 \
    > logs/effv2s512_fold0.log 2>&1
log "512 FOLD0 DONE: $(grep -E '^\[epoch' logs/effv2s512_fold0.log | tail -1)"
