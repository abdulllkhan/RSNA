#!/usr/bin/env bash
# Submission #6: sub #5 members (5x224 base + 5x384 hr) + BAKED arm weights + BAKED RSNA_TTA=S (slot-phase TTA).
# Usage: dev/sub6.sh "base=1,hr=3.0" 3   — dataset is NOT re-versioned (weights unchanged since #5) unless REVERSION=1.
set -uo pipefail
cd /home/abdulllkhan/Documents/RSNA || exit 1
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"; KG="../multi_step_tool_attack/.venv/bin/kaggle"
U=abdullllkhan; DS=rsna-knee-weights-v1; K=rsna-knee-multi-infer; COMP=rsna-knee-abnormality-detection
W="${1:-base=1,hr=3.0}"; S="${2:-3}"; NAME=effv2s_k24_384; PLANE0=models/effv2s_plane_k8x3_384_fold0.pt
log(){ echo "[$(date '+%F %T')] $*"; }; fail(){ log "SUB6 FAILED: $*"; exit 1; }
ps -eo cmd | grep -qE '[p]ython[^ ]* -m src\.rsna\.train_(hr|plane)' && fail "a trainer is running"
grep -q 'ALL 384 FOLDS DONE' logs/effv2s384_folds.log 2>/dev/null || fail "plain-384 folds not done"
mkdir -p models/full
for f in 0 1 2 3 4; do cp=models/${NAME}_fold$f.pt; [ -f "$cp" ] || fail "missing $cp"
  if [ "$(stat -c %s "$cp")" -gt 120000000 ]; then PYTHONPATH=$PWD .venv/bin/python dev/slim_ckpt.py "$cp" | grep -E 'verify' || fail "slim $cp"
    mv "$cp" "models/full/${NAME}_fold${f}_full.pt"; mv "${cp%.pt}.slim.pt" "$cp"; fi; done
PKG=dev/kaggle/weights_pkg; rm -rf "$PKG"; mkdir -p "$PKG"
cp models/effv2s_k24_fold[0-4].pt models/${NAME}_fold[0-4].pt "$PKG"/
if [ -f "$PLANE0" ]; then if [ "$(stat -c %s "$PLANE0")" -gt 120000000 ]; then PYTHONPATH=$PWD .venv/bin/python dev/slim_ckpt.py "$PLANE0" | grep -E verify || fail "slim plane0"; mv "$PLANE0" models/full/effv2s_plane_k8x3_384_fold0_full.pt; mv "${PLANE0%.pt}.slim.pt" "$PLANE0"; fi; cp "$PLANE0" "$PKG"/; fi
sed "s/__KAGGLE_USERNAME__/$U/" dev/kaggle/dataset-metadata.json > "$PKG/dataset-metadata.json"; ls "$PKG" | wc -l
if [ "${REVERSION:-0}" = "1" ]; then $KG datasets version -p "$PKG" -m "multi-arm weights package" --dir-mode zip 2>&1 | tail -2; else log "dataset re-version skipped (REVERSION=0)"; fi
want=$(ls "$PKG"/*.pt | wc -l); for i in $(seq 1 60); do have=$($KG datasets files "$U/$DS" 2>/dev/null | grep -c '\.pt' || true); [ "$have" -ge "$want" ] && { log "dataset ready $have/$want"; break; }; sleep 15; done
.venv/bin/python notebook/build_notebook_multi.py 2>&1 | tail -1
# bake arm weights (Kaggle has no env): default the env lookup to $W in both generated files
sed -i "s|os.environ.get(\"RSNA_ARM_WEIGHTS\")|os.environ.get(\"RSNA_ARM_WEIGHTS\", \"$W\")|" notebook/submission_multi_standalone.py
sed -i "s|os.environ.get(\\\\\"RSNA_ARM_WEIGHTS\\\\\")|os.environ.get(\\\\\"RSNA_ARM_WEIGHTS\\\\\", \\\\\"$W\\\\\")|" notebook/submission_multi.ipynb
grep -c "RSNA_ARM_WEIGHTS\\\\\", \\\\\"$W" notebook/submission_multi.ipynb | sed 's/^/baked in ipynb (expect 1): /'
sed -i "s|os.environ.get(\"RSNA_TTA\", \"1\")|os.environ.get(\"RSNA_TTA\", \"$S\")|" notebook/submission_multi_standalone.py
sed -i "s|os.environ.get(\\\\\"RSNA_TTA\\\\\", \\\\\"1\\\\\")|os.environ.get(\\\\\"RSNA_TTA\\\\\", \\\\\"$S\\\\\")|" notebook/submission_multi.ipynb
nt=$(grep -c "RSNA_TTA\\\\\", \\\\\"$S" notebook/submission_multi.ipynb); echo "baked TTA S=$S in ipynb (expect 1): $nt"; [ "$nt" -eq 1 ] || fail "TTA bake count $nt"
ST=/tmp/rsna_multi_push; rm -rf "$ST"; mkdir -p "$ST"; cp notebook/submission_multi.ipynb "$ST"/; cp notebook/kernel-metadata-multi.json "$ST/kernel-metadata.json"
out=$($KG kernels push -p "$ST" --accelerator NvidiaTeslaT4 2>&1); echo "$out" | tail -2
ver=$(echo "$out" | grep -oE 'Kernel version [0-9]+' | grep -oE '[0-9]+$'); [ -n "$ver" ] || fail "no kernel version"; echo "$ver" > dev/kaggle/last_multi_kernel_version
KEY=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['key'])"); st=""
for i in $(seq 1 120); do st=$(curl -s -u "$U:$KEY" "https://www.kaggle.com/api/v1/kernels/status?userName=$U&kernelSlug=$K"); echo "$st" | grep -qiE 'complete|error|cancel' && break; sleep 45; done
log "kernel v$ver terminal: $st"; echo "$st" | grep -qi complete || fail "kernel not complete"
rm -rf /tmp/koutm; mkdir -p /tmp/koutm; $KG kernels output "$U/$K" -p /tmp/koutm >/dev/null 2>&1
nd=$(cut -d, -f2- /tmp/koutm/submission.csv | tail -n +2 | sort -u | wc -l); pl=$(grep -ac "arm=hr" /tmp/koutm/*.log 2>/dev/null || echo 0); tt=$(grep -ac "\[tta\] arm=" /tmp/koutm/*.log 2>/dev/null || echo 0)
log "distinct rows=$nd hr members detected=$pl tta lines=$tt weights=$W S=$S"; [ "$nd" -gt 1 ] && [ "$pl" -ge 5 ] && [ "$tt" -ge 2 ] || fail "gate nd=$nd hr=$pl tta=$tt"
$KG competitions submit "$COMP" -k "$U/$K" -v "$ver" -f submission.csv -m "sub6: multi-arm rank-avg $W + slot-phase TTA S=$S; members base=5 hr=5 (multi v$ver)" 2>&1 | tail -3
$KG competitions submissions "$COMP" 2>&1 | sed -n '3p'; log "SUB6 DONE"
