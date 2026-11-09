#!/usr/bin/env bash
# Waits for the distilled c110 folds (QUEUE_V2K DONE), fits the base-free stitched ensemble for three v2 compositions
# (dev/fit_v2_stitched2.py), and submits sub #11 = hr_tta3 + best v2 composition (TTA "3") if the stitched steven
# gain over sub #10b's 0.8899 is >= 0.005. Between +0.003 and +0.005 it HOLDS for a manual decision.
# Markers: AUTO_SUB11 FIT | AUTO_SUB11 HOLD | AUTO_SUB11 SKIP | AUTO_SUB11 LAUNCHED | AUTO_SUB11 SCORE | AUTO_SUB11 FAILED
cd /home/abdulllkhan/Documents/RSNA || exit 1
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
log(){ echo "[$(date '+%F %T')] $*"; }
REF=0.8899; BAR=0.005; HOLD=0.003
until grep -q 'QUEUE_V2K DONE' logs/queue_v2.log 2>/dev/null; do sleep 300; done
n=$(ls data/oof/v2_effv2s384_c110_distill_fold[0-4].parquet 2>/dev/null | wc -l); [ "$n" -eq 5 ] || { log "AUTO_SUB11 FAILED: only $n distill OOF folds"; exit 1; }
nice -n 19 .venv/bin/python dev/fit_v2_stitched2.py "lr+c110" "lr+c110_distill" "lr+c110+c110_distill" 2>&1 | grep -v Warning | tee logs/fit_v2_stitched_11.log
CFG=$(grep -oE '^RECOMMENDED_CFG=.*' logs/fit_v2_stitched_11.log | cut -d= -f2); W=$(grep -oE '^RECOMMENDED_W=[0-9]+' logs/fit_v2_stitched_11.log | cut -d= -f2)
S=$(grep -oE '^RECOMMENDED_STEVEN=[0-9.]+' logs/fit_v2_stitched_11.log | cut -d= -f2); [ -n "$CFG" ] && [ -n "$W" ] && [ -n "$S" ] || { log "AUTO_SUB11 FAILED: fit parse"; exit 1; }
log "AUTO_SUB11 FIT cfg=$CFG w=$W stitched_steven=$S (sub #10b was $REF; bar +$BAR)"
[ "$CFG" != "lr+c110" ] || { log "AUTO_SUB11 SKIP: the distilled arm does not improve the stitched ensemble"; exit 0; }
if python3 -c "import sys; sys.exit(0 if float('$S') >= $REF + $BAR else 1)"; then :
elif python3 -c "import sys; sys.exit(0 if float('$S') >= $REF + $HOLD else 1)"; then log "AUTO_SUB11 HOLD: +$(python3 -c "print(round(float('$S')-$REF,4))") is below the 0.005 bar -> manual decision (rerun with FORCE=1)"; [ "${FORCE:-0}" = "1" ] || exit 0
else log "AUTO_SUB11 SKIP: $S < $(python3 -c "print($REF+$HOLD)")"; exit 0; fi
CK=""; for m in $(echo "$CFG" | tr '+' ' '); do CK="$CK $(ls models/v2_effv2s384_${m}_fold[0-4].pt | tr '\n' ' ')"; done
n110=$(echo "$CK" | tr ' ' '\n' | grep -cE 'c110'); RE="2 decode group\\(s\\): \\[\\(110\\.0, 384, ${n110}\\), \\(140\\.0, 384, 5\\)\\]"
log "AUTO_SUB11 packaging: cfg=$CFG w=$W members=$(echo $CK | wc -w) expect=$RE"
V2_CKPTS="$CK" EXPECT_LOG_RE="$RE" bash dev/sub7.sh "base=0,hr=3,v2=$W" "3" > logs/sub11.log 2>&1
grep -q 'SUB7 DONE' logs/sub11.log || { log "AUTO_SUB11 FAILED: see logs/sub11.log"; exit 1; }
REFN=$(../multi_step_tool_attack/.venv/bin/kaggle competitions submissions rsna-knee-abnormality-detection 2>/dev/null | sed -n '3p' | awk '{print $1}')
log "AUTO_SUB11 LAUNCHED ref=$REFN cfg=$CFG w=$W"; bash dev/watch_score.sh "$REFN" > logs/score_watch11.log 2>&1
log "AUTO_SUB11 SCORE $(grep '^SCORE' logs/score_watch11.log | tail -1)"
