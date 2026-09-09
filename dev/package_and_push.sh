#!/usr/bin/env bash
# Submission #2 (5-fold ensemble): slim -> dataset version -> wait ready -> rebuild -> T4 push
# -> record version -> self-gated submit. Run ONLY after "ALL FOLDS 1-4 DONE" (guarded).
set -euo pipefail
cd /home/abdulllkhan/Documents/RSNA
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
KG="../multi_step_tool_attack/.venv/bin/kaggle"
U=abdullllkhan; DS=rsna-knee-weights-v1; K=rsna-knee-baseline-infer
MSG="${1:-5-fold effv2s_k24 ensemble}"
if pgrep -f 'src.rsna.train' >/dev/null; then echo "REFUSING: training is still running (checkpoints being written)"; exit 1; fi
echo "== [1/6] slim any full-size best checkpoints (CPU) =="
mkdir -p models/full
for cp in models/*_fold[0-9].pt; do
  sz=$(stat -c %s "$cp"); if [ "$sz" -gt 120000000 ]; then
    PYTHONPATH=$PWD .venv/bin/python dev/slim_ckpt.py "$cp" 2>&1 | grep -E 'slim keys|verify'
    b=$(basename "$cp" .pt); mv "$cp" "models/full/${b}_full.pt"; mv "${cp%.pt}.slim.pt" "$cp"
  fi
done
ls -la models/*_fold[0-9].pt
echo "== [2/6] stage weights package =="
PKG=dev/kaggle/weights_pkg; rm -rf "$PKG"; mkdir -p "$PKG"
cp models/*_fold[0-9].pt "$PKG"/
sed "s/__KAGGLE_USERNAME__/$U/" dev/kaggle/dataset-metadata.json > "$PKG/dataset-metadata.json"
ls -la "$PKG"
echo "== [3/6] new dataset version =="
$KG datasets version -p "$PKG" -m "$MSG" --dir-mode zip 2>&1 | tail -3
echo "   waiting for dataset to finish processing..."
want=$(ls "$PKG"/*.pt | wc -l)
for i in $(seq 1 40); do
  have=$($KG datasets files "$U/$DS" 2>/dev/null | grep -c '\.pt' || true)
  [ "$have" -ge "$want" ] && { echo "   dataset ready: $have/$want files"; break; }
  sleep 15
done
echo "== [4/6] rebuild notebook =="
.venv/bin/python notebook/build_notebook.py 2>&1 | tail -1
echo "== [5/6] push pinned to T4 =="
out=$($KG kernels push -p notebook --accelerator NvidiaTeslaT4 2>&1); echo "$out" | tail -2
ver=$(echo "$out" | grep -oE 'Kernel version [0-9]+' | grep -oE '[0-9]+$')
[ -n "$ver" ] || { echo "could not parse kernel version"; exit 1; }
echo "$ver" > dev/kaggle/last_kernel_version; echo "$MSG" > dev/kaggle/last_submit_msg; echo "   kernel version = $ver"
echo "== [6/6] launch self-gated submit driver (polls, verifies non-0.5, submits -v $ver) =="
: > logs/submit1.log
nohup bash dev/submit1.sh > logs/submit1.log 2>&1 &
echo "submit driver PID $!  (watch logs/submit1.log)"
