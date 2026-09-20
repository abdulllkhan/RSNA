#!/usr/bin/env bash
# Waits for the v2 folds runner marker, fits the v2 weight on the stitched 5-fold OOF, and submits sub #8
# (10 members + TTA3 + 5 v2 folds). Markers: AUTO_SUB8 FIT | AUTO_SUB8 LAUNCHED | AUTO_SUB8 SCORE | AUTO_SUB8 FAILED
cd /home/abdulllkhan/Documents/RSNA || exit 1
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
log(){ echo "[$(date '+%F %T')] $*"; }
until grep -q 'V2 FOLDS DONE' logs/v2_lr_folds1-4.log 2>/dev/null; do sleep 120; done
n=$(ls data/oof/v2_effv2s384_lr_fold[0-4].parquet 2>/dev/null | wc -l); [ "$n" -eq 5 ] || { log "AUTO_SUB8 FAILED: only $n v2 OOF folds"; exit 1; }
nice -n 19 .venv/bin/python dev/fit_v2_stitched.py v2_effv2s384_lr 2>&1 | grep -v Warning | tee logs/fit_v2_stitched.log
W=$(grep -oE 'RECOMMENDED_W=[0-9]+' logs/fit_v2_stitched.log | cut -d= -f2); [ -n "$W" ] || { log "AUTO_SUB8 FAILED: no weight"; exit 1; }
[ "$W" -lt 3 ] && W=3; [ "$W" -gt 8 ] && W=8; log "AUTO_SUB8 FIT w=$W"
V2_CKPTS="models/v2_effv2s384_lr_fold0.pt models/v2_effv2s384_lr_fold1.pt models/v2_effv2s384_lr_fold2.pt models/v2_effv2s384_lr_fold3.pt models/v2_effv2s384_lr_fold4.pt" \
  bash dev/sub7.sh "base=1,hr=3,v2=$W" 3 > logs/sub8.log 2>&1
grep -q 'SUB7 DONE' logs/sub8.log || { log "AUTO_SUB8 FAILED: see logs/sub8.log"; exit 1; }
REF=$(../multi_step_tool_attack/.venv/bin/kaggle competitions submissions rsna-knee-abnormality-detection 2>/dev/null | sed -n '3p' | awk '{print $1}')
log "AUTO_SUB8 LAUNCHED ref=$REF w=$W"; bash dev/watch_score.sh "$REF" > logs/score_watch8.log 2>&1
log "AUTO_SUB8 SCORE $(grep '^SCORE' logs/score_watch8.log | tail -1)"
