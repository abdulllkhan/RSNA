#!/usr/bin/env bash
# Submission #3: 5-fold 224 base + 384 fold-0 (hr arm), rank-averaged via the multi-arm kernel.
# Self-gated: refuses while train_hr runs; submits only if hr member detected AND real predictions.
set -uo pipefail
cd /home/abdulllkhan/Documents/RSNA || exit 1
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"; KG="../multi_step_tool_attack/.venv/bin/kaggle"
U=abdullllkhan; DS=rsna-knee-weights-v1; K=rsna-knee-multi-infer; COMP=rsna-knee-abnormality-detection
log(){ echo "[$(date '+%F %T')] $*"; }; fail(){ log "SUB3 FAILED: $*"; exit 1; }
ps -eo cmd | grep -qE '[p]ython[^ ]* -m src\.rsna\.train_hr' && fail "train_hr still running"
HR=models/effv2s_k24_384_fold0.pt; [ -f "$HR" ] || fail "no $HR"
if [ "$(stat -c %s "$HR")" -gt 120000000 ]; then
  PYTHONPATH=$PWD .venv/bin/python dev/slim_ckpt.py "$HR" | grep -E 'slim keys|verify' || fail "slim"
  mkdir -p models/full; mv "$HR" models/full/effv2s_k24_384_fold0_full.pt; mv "${HR%.pt}.slim.pt" "$HR"
fi
PKG=dev/kaggle/weights_pkg; rm -rf "$PKG"; mkdir -p "$PKG"
cp models/effv2s_k24_fold[0-4].pt "$HR" "$PKG"/; sed "s/__KAGGLE_USERNAME__/$U/" dev/kaggle/dataset-metadata.json > "$PKG/dataset-metadata.json"
ls -la "$PKG" | awk '{print $5,$9}'
$KG datasets version -p "$PKG" -m "5x224 base + 384 fold0 (hr)" --dir-mode zip 2>&1 | tail -2
want=$(ls "$PKG"/*.pt | wc -l)
for i in $(seq 1 40); do have=$($KG datasets files "$U/$DS" 2>/dev/null | grep -c '\.pt' || true); [ "$have" -ge "$want" ] && { log "dataset ready $have/$want"; break; }; sleep 15; done
.venv/bin/python notebook/build_notebook_multi.py 2>&1 | tail -1
ST=/tmp/rsna_multi_push; rm -rf "$ST"; mkdir -p "$ST"; cp notebook/submission_multi.ipynb "$ST"/; cp notebook/kernel-metadata-multi.json "$ST/kernel-metadata.json"
out=$($KG kernels push -p "$ST" --accelerator NvidiaTeslaT4 2>&1); echo "$out" | tail -2
ver=$(echo "$out" | grep -oE 'Kernel version [0-9]+' | grep -oE '[0-9]+$'); [ -n "$ver" ] || fail "no kernel version"; echo "$ver" > dev/kaggle/last_multi_kernel_version
KEY=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['key'])"); st=""
for i in $(seq 1 100); do st=$(curl -s -u "$U:$KEY" "https://www.kaggle.com/api/v1/kernels/status?userName=$U&kernelSlug=$K"); echo "$st" | grep -qiE 'complete|error|cancel' && break; sleep 45; done
log "kernel v$ver terminal: $st"; echo "$st" | grep -qi complete || fail "kernel not complete"
rm -rf /tmp/koutm; mkdir -p /tmp/koutm; $KG kernels output "$U/$K" -p /tmp/koutm >/dev/null 2>&1
nd=$(cut -d, -f2- /tmp/koutm/submission.csv | tail -n +2 | sort -u | wc -l)
hrok=$(grep -ac "arm=hr" /tmp/koutm/*.log 2>/dev/null || echo 0)
log "distinct rows=$nd hr members detected=$hrok"
[ "$nd" -gt 1 ] && [ "$hrok" -ge 1 ] || fail "gate: nd=$nd hr=$hrok"
$KG competitions submit "$COMP" -k "$U/$K" -v "$ver" -f submission.csv -m "sub3: 5-fold 224 base + 384 fold0 hr, rank-avg (multi kernel v$ver)" 2>&1 | tail -3
$KG competitions submissions "$COMP" 2>&1 | sed -n '3p'
log "SUB3 DONE"
