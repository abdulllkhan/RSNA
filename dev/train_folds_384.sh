#!/usr/bin/env bash
# Plain 384 arm, folds 1-4 (fold 0 already trained by the runbook). ~2.8 h/fold.
cd /home/abdulllkhan/Documents/RSNA || exit 1
for f in 1 2 3 4; do
  echo "=== 384 FOLD $f START $(date) ==="
  .venv/bin/python -m src.rsna.train_hr --fold "$f" --size 384 --k 24 --batch 6 --grad-accum 3 \
      --workers 6 --epochs 10 --lr 3e-4 --cache-root data/cache384 --name effv2s_k24_384 --max-train-minutes 170
  echo "=== 384 FOLD $f DONE $(date) ==="
done
echo "=== ALL 384 FOLDS DONE $(date) ==="
