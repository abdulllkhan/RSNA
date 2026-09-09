#!/usr/bin/env bash
# Package weights + push the inference notebook to Kaggle. DO NOT RUN AUTOMATICALLY —
# the orchestrator gates all Kaggle pushes/submissions. This script documents the exact
# flow and is safe to read; run it deliberately.
#
# Usage:
#   export KAGGLE_CONFIG_DIR=$HOME/.kaggle          # classic kaggle.json auth
#   KAGGLE_USERNAME=<your-username> bash dev/push_to_kaggle.sh
#
# Requires: kaggle CLI authenticated, competition rules accepted, and a trained
# checkpoint in models/ (e.g. models/effv2s_k24_fold0.pt).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${KAGGLE_USERNAME:?set KAGGLE_USERNAME=<your kaggle username>}"
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
KAGGLE="../multi_step_tool_attack/.venv/bin/kaggle"   # kaggle CLI v2.2.4
COMP="rsna-knee-abnormality-detection"

echo ">> [1/5] Rebuild the notebook from src/rsna via build_notebook.py ..."
.venv/bin/python notebook/build_notebook.py

echo ">> [2/5] Stage the weights package (best-by-fold checkpoints only, skip *_last.pt) ..."
PKG=dev/kaggle/weights_pkg
rm -rf "$PKG" && mkdir -p "$PKG"
cp models/*_fold[0-9].pt "$PKG"/            # e.g. effv2s_k24_fold0.pt
sed "s/__KAGGLE_USERNAME__/${KAGGLE_USERNAME}/" dev/kaggle/dataset-metadata.json \
  > "$PKG/dataset-metadata.json"
ls -la "$PKG"

echo ">> [3/5] Create (first time) OR version (subsequent) the private weights dataset ..."
# FIRST TIME:
#   "$KAGGLE" datasets create   -p "$PKG" --dir-mode zip
# SUBSEQUENT UPDATES:
#   "$KAGGLE" datasets version  -p "$PKG" -m "weights update" --dir-mode zip
echo "   (uncomment the create/version line appropriate for your situation)"

echo ">> [4/5] Fill kernel metadata with your username ..."
sed "s/__KAGGLE_USERNAME__/${KAGGLE_USERNAME}/g" notebook/kernel-metadata.json \
  > notebook/kernel-metadata.local.json
mv notebook/kernel-metadata.local.json notebook/kernel-metadata.json

echo ">> [5/5] Verify auth, then push the kernel (queues a GPU run, internet OFF) ..."
"$KAGGLE" competitions list -s "$COMP" >/dev/null && echo "   auth OK"
# "$KAGGLE" kernels push -p notebook

echo
echo ">> After pushing, track + submit:"
echo "   $KAGGLE kernels status ${KAGGLE_USERNAME}/rsna-knee-baseline-infer"
echo "   # when COMPLETE, submit from the notebook Output, or:"
echo "   $KAGGLE competitions submit -c $COMP -f submission.csv -m 'baseline_v1 effv2s fold0'"
