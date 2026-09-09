#!/usr/bin/env bash
# Serial GPU queue (one GPU): wait for the 384 fold-0 to finish, then run the plane arm at 224
# (fold-0 planewise, then the concat ablation). Markers: 384 FOLD0 FINISHED | A3 PLANEWISE DONE |
# A3 CONCAT DONE | GPU QUEUE DONE | QUEUE FAILED
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
until [ -f logs/effv2s384_fold0.log ]; do sleep 120; done
log "384 fold-0 log present; waiting for train_hr to exit"
sleep 120
while ps -eo cmd | grep -qE '[p]ython[^ ]* -m src\.rsna\.train_hr'; do sleep 300; done
log "384 FOLD0 FINISHED: $(grep -E '^\[epoch' logs/effv2s384_fold0.log | tail -1)"
.venv/bin/python dev/plane_gates.py --index-cache data/oof/study_index_train.pkl -n 300 \
  || { log "QUEUE FAILED: plane gates"; exit 1; }
COMMON="--fold 0 --size 224 --k-per-plane 8 8 8 --batch 16 --workers 6 --epochs 10 --lr 3e-4 --cache-root data/processed_jpeg224 --index-cache data/oof/study_index_train.pkl --max-train-minutes 58"
log "== A3 planewise 224 fold-0 =="
.venv/bin/python -m src.rsna.train_plane $COMMON --fusion planewise --name effv2s_plane_k8x3_224 > logs/plane224_fold0.log 2>&1
log "A3 PLANEWISE DONE: $(grep -E '^\[epoch' logs/plane224_fold0.log | tail -1)"
log "== A3 concat ablation 224 fold-0 =="
.venv/bin/python -m src.rsna.train_plane $COMMON --fusion concat --name effv2s_plane_concat_224 > logs/plane224_concat_fold0.log 2>&1
log "A3 CONCAT DONE: $(grep -E '^\[epoch' logs/plane224_concat_fold0.log | tail -1)"
log "GPU QUEUE DONE"
