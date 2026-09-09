#!/usr/bin/env bash
# 384 + plane-aware pooling, folds 0-4 (fold 0 included: the plain-384 fold-0 is a different model).
cd /home/abdulllkhan/Documents/RSNA || exit 1
for f in 0 1 2 3 4; do
  echo "=== 384-PLANE FOLD $f START $(date) ==="
  .venv/bin/python -m src.rsna.train_plane --fold "$f" --size 384 --k-per-plane 8 8 8 --fusion planewise \
      --batch 6 --grad-accum 3 --workers 6 --epochs 10 --lr 3e-4 --cache-root data/cache384 \
      --index-cache data/cache384/study_index_train.pkl --name effv2s_plane_k8x3_384 --max-train-minutes 170
  echo "=== 384-PLANE FOLD $f DONE $(date) ==="
done
echo "=== ALL 384-PLANE FOLDS DONE $(date) ==="
