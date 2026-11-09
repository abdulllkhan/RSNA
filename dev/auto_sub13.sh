#!/usr/bin/env bash
# Waits for the seed-2 folds (SEED2 DONE), fits the base-free stitched ensemble for three honest v2 compositions
# (dev/fit_v2_stitched2.py), and submits sub #13 = hr_tta3 + best v2 composition (TTA "3") if the stitched steven
# gain over sub #10b's 0.8899 is >= 0.005. Between +0.003 and +0.005 it HOLDS for a manual decision.
# Markers: AUTO_SUB13 FIT | AUTO_SUB13 HOLD | AUTO_SUB13 SKIP | AUTO_SUB13 LAUNCHED | AUTO_SUB13 SCORE | AUTO_SUB13 FAILED
cd /home/abdulllkhan/Documents/RSNA || exit 1
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
log(){ echo "[$(date '+%F %T')] $*"; }
REF=0.8899; BAR=0.005; HOLD=0.003
until grep -q 'SEED2 DONE' logs/queue_v2.log 2>/dev/null; do sleep 300; done
n=$(ls data/oof/v2_effv2s384_lr_r2_fold[0-4].parquet 2>/dev/null | wc -l); [ "$n" -eq 5 ] || { log "AUTO_SUB13 FAILED: only $n distill OOF folds"; exit 1; }
nice -n 19 .venv/bin/python dev/fit_v2_stitched2.py "lr+c110" "lr+lr_r2+c110+c110_r2" "lr+lr_r2+c110+c110_r2+v2_effv2m384" 2>&1 | grep -v Warning | tee logs/fit_v2_stitched_13.log
CFG=$(grep -oE '^RECOMMENDED_CFG=.*' logs/fit_v2_stitched_13.log | cut -d= -f2); W=$(grep -oE '^RECOMMENDED_W=[0-9]+' logs/fit_v2_stitched_13.log | cut -d= -f2)
S=$(grep -oE '^RECOMMENDED_STEVEN=[0-9.]+' logs/fit_v2_stitched_13.log | cut -d= -f2); [ -n "$CFG" ] && [ -n "$W" ] && [ -n "$S" ] || { log "AUTO_SUB13 FAILED: fit parse"; exit 1; }
log "AUTO_SUB13 FIT cfg=$CFG w=$W stitched_steven=$S (sub #10b was $REF; bar +$BAR)"
[ "$CFG" != "lr+c110" ] || { log "AUTO_SUB13 SKIP: seed-2 members do not improve the stitched ensemble"; exit 0; }
if python3 -c "import sys; sys.exit(0 if float('$S') >= $REF + $BAR else 1)"; then :
elif python3 -c "import sys; sys.exit(0 if float('$S') >= $REF + $HOLD else 1)"; then log "AUTO_SUB13 HOLD: +$(python3 -c "print(round(float('$S')-$REF,4))") is below the 0.005 bar -> manual decision (rerun with FORCE=1)"; [ "${FORCE:-0}" = "1" ] || exit 0
else log "AUTO_SUB13 SKIP: $S < $(python3 -c "print($REF+$HOLD)")"; exit 0; fi
CK=""; for m in $(echo "$CFG" | tr '+' ' '); do case "$m" in v2_*) pat="models/${m}_fold[0-4].pt";; *) pat="models/v2_effv2s384_${m}_fold[0-4].pt";; esac; CK="$CK $(ls $pat | tr '\n' ' ')"; done
n110=$(echo "$CK" | tr ' ' '\n' | grep -cE 'c110'); n140=$(echo "$CK" | tr ' ' '\n' | grep -cE '_lr|effv2m'); RE="2 decode group\\(s\\): \\[\\(110\\.0, 384, ${n110}\\), \\(140\\.0, 384, ${n140}\\)\\]"
log "AUTO_SUB13 packaging: cfg=$CFG w=$W members=$(echo $CK | wc -w) expect=$RE"
V2_CKPTS="$CK" EXPECT_LOG_RE="$RE" bash dev/sub7.sh "base=0,hr=3,v2=$W" "3" > logs/sub13.log 2>&1
grep -q 'SUB7 DONE' logs/sub13.log || { log "AUTO_SUB13 FAILED: see logs/sub13.log"; exit 1; }
REFN=$(../multi_step_tool_attack/.venv/bin/kaggle competitions submissions rsna-knee-abnormality-detection 2>/dev/null | sed -n '3p' | awk '{print $1}')
log "AUTO_SUB13 LAUNCHED ref=$REFN cfg=$CFG w=$W"; bash dev/watch_score.sh "$REFN" > logs/score_watch13.log 2>&1
log "AUTO_SUB13 SCORE $(grep '^SCORE' logs/score_watch13.log | tail -1)"
