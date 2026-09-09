#!/usr/bin/env bash
# Sequential GPU queue — launch ONLY when no CPU-heavy subagent work is active (dataloaders starve).
#   1) TTA OOF eval, 384 arm (S=3)   2) TTA OOF eval, 224 arm   3) 2.5D 384 fold-0 (cmd in dev/kaggle/next_25d_cmd)
#   4) 512 fold-0 headroom test (cap 330 min).  Markers: TTA384 DONE | TTA224 DONE | 25D FOLD0 DONE | 512 FOLD0 DONE | QUEUE_GPU_NEXT DONE
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
while ps -eo cmd | grep -qE '[p]ython[^ ]* -m src\.rsna\.train_'; do sleep 60; done
log "== 1) TTA eval 384 (5 folds, S=3) =="
.venv/bin/python -u dev/tta_eval.py --name effv2s_k24_384 --root data/cache384 --variants 3 --report-s 1,2,3 \
    --folds 0,1,2,3,4 --device cuda --batch 4 --workers 8 --out-dir data/oof/tta --check-combiner > logs/tta_384_s3.log 2>&1
log "TTA384 DONE: $(grep -iE 'macro|S=' logs/tta_384_s3.log | tail -4 | tr '\n' ' | ')"
log "== 2) TTA eval 224 =="
.venv/bin/python -u dev/tta_eval.py --name effv2s_k24 --root data/processed_jpeg224 --variants 3 --report-s 1,2,3 \
    --folds 0,1,2,3,4 --device cuda --batch 12 --workers 8 --out-dir data/oof/tta --check-combiner > logs/tta_224_s3.log 2>&1
log "TTA224 DONE: $(grep -iE 'macro|S=' logs/tta_224_s3.log | tail -4 | tr '\n' ' | ')"
if [ -s dev/kaggle/next_25d_cmd ]; then
  log "== 3) 2.5D 384 fold-0 =="; bash dev/kaggle/next_25d_cmd > logs/25d384_fold0.log 2>&1
  log "25D FOLD0 DONE: $(grep -E '^\[epoch' logs/25d384_fold0.log | tail -1)"
else log "no dev/kaggle/next_25d_cmd -> skipping 2.5D"; fi
log "== 4) 512 fold-0 headroom test (cap 330) =="
.venv/bin/python -m src.rsna.train_hr --fold 0 --size 512 --k 24 --batch 4 --grad-accum 4 --workers 6 \
    --epochs 10 --lr 3e-4 --cache-root data/cache512 --name effv2s_k24_512 --max-train-minutes 330 > logs/effv2s512_fold0.log 2>&1
log "512 FOLD0 DONE: $(grep -E '^\[epoch' logs/effv2s512_fold0.log | tail -1)"
log "QUEUE_GPU_NEXT DONE"
