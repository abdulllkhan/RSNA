#!/usr/bin/env bash
# After the 512 fold-0 test exits: TTA OOF evaluation for the 384 arm (S=3), then the 224 arm.
# Markers: TTA384 DONE | TTA224 DONE | QUEUE_POST512 DONE
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
while ! grep -qE '512 FOLD0 DONE' logs/queue_512.log 2>/dev/null; do sleep 300; done
while ps -eo cmd | grep -qE '[p]ython[^ ]* -m src\.rsna\.train_'; do sleep 60; done
log "GPU free; TTA eval 384 (5 folds, S=3)"
.venv/bin/python -u dev/tta_eval.py --name effv2s_k24_384 --root data/cache384 --variants 3 --report-s 1,2,3 \
    --folds 0,1,2,3,4 --device cuda --batch 4 --workers 8 --out-dir data/oof/tta --check-combiner > logs/tta_384_s3.log 2>&1
log "TTA384 DONE: $(grep -iE 'macro|S=' logs/tta_384_s3.log | tail -3 | tr '\n' ' | ')"
.venv/bin/python -u dev/tta_eval.py --name effv2s_k24 --root data/processed_jpeg224 --variants 3 --report-s 1,2,3 \
    --folds 0,1,2,3,4 --device cuda --batch 12 --workers 8 --out-dir data/oof/tta --check-combiner > logs/tta_224_s3.log 2>&1
log "TTA224 DONE: $(grep -iE 'macro|S=' logs/tta_224_s3.log | tail -3 | tr '\n' ' | ')"
log "QUEUE_POST512 DONE"
