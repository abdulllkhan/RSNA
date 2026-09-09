#!/usr/bin/env bash
# Train folds 1-4 of the effv2s_k24 baseline sequentially (one GPU).
# workers=6 keeps dataloader memory low enough that harness background tasks survive.
cd /home/abdulllkhan/Documents/RSNA || exit 1
for f in 1 2 3 4; do
  echo "=== FOLD $f START $(date) ==="
  .venv/bin/python -m src.rsna.train --fold "$f" --epochs 10 --batch 16 \
      --workers 6 --lr 3e-4 --name effv2s_k24 --max-train-minutes 58
  echo "=== FOLD $f DONE $(date) ==="
done
echo "=== ALL FOLDS 1-4 DONE $(date) ==="
