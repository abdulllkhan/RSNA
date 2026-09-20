#!/usr/bin/env bash
# Waits for the effv2-m folds (V2 FOLDS DONE v2_effv2s384_c110), fits the v2 arm (effs+effm averaged) on the stitched OOF, and submits
# sub #9 = 5x224 + 5x384 + 5x effs + 5x effm, TTA S=3 on all arms, if the ensemble beats sub #8's 0.8864 by >= 0.002.
# Markers: AUTO_SUB9B FIT | AUTO_SUB9B SKIP | AUTO_SUB9B LAUNCHED | AUTO_SUB9B SCORE | AUTO_SUB9B FAILED
cd /home/abdulllkhan/Documents/RSNA || exit 1
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
log(){ echo "[$(date '+%F %T')] $*"; }
until grep -q 'V2 FOLDS DONE v2_effv2s384_c110' logs/v2_c110_folds1-4.log 2>/dev/null; do sleep 300; done
n=$(ls data/oof/v2_effv2s384_c110_fold[0-4].parquet 2>/dev/null | wc -l); [ "$n" -eq 5 ] || { log "AUTO_SUB9B FAILED: only $n c110 OOF folds"; exit 1; }
nice -n 19 .venv/bin/python dev/fit_v2_stitched.py v2_effv2s384_lr v2_effv2s384_c110 2>&1 | grep -v Warning | tee logs/fit_v2_stitched_9b.log
W=$(grep -oE 'RECOMMENDED_W=[0-9]+' logs/fit_v2_stitched_9b.log | cut -d= -f2); [ -n "$W" ] || { log "AUTO_SUB9B FAILED: no weight"; exit 1; }
[ "$W" -lt 3 ] && W=3; [ "$W" -gt 10 ] && W=10
best=$(grep -E "^  w=$W " logs/fit_v2_stitched_9b.log | grep -oE 'steven [0-9.]+' | awk '{print $2}')
log "AUTO_SUB9B FIT w=$W ensemble_steven=$best (sub #8 was 0.8864)"
python3 -c "import sys; sys.exit(0 if float('$best') >= 0.8884 else 1)" || { log "AUTO_SUB9B SKIP: $best < 0.8884"; exit 0; }
V2_CKPTS="$(ls models/v2_effv2s384_lr_fold[0-4].pt models/v2_effv2s384_c110_fold[0-4].pt | tr '\n' ' ')" bash dev/sub7.sh "base=1,hr=3,v2=$W" 3 > logs/sub9b.log 2>&1
grep -q 'SUB7 DONE' logs/sub9b.log || { log "AUTO_SUB9B FAILED: see logs/sub9b.log"; exit 1; }
REF=$(../multi_step_tool_attack/.venv/bin/kaggle competitions submissions rsna-knee-abnormality-detection 2>/dev/null | sed -n '3p' | awk '{print $1}')
log "AUTO_SUB9B LAUNCHED ref=$REF w=$W"; bash dev/watch_score.sh "$REF" > logs/score_watch9b.log 2>&1
log "AUTO_SUB9B SCORE $(grep '^SCORE' logs/score_watch9b.log | tail -1)"
