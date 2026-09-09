#!/usr/bin/env bash
# Fully autonomous sub #4: wait for plain-384 folds 1-4 -> fit hr weight on stitched OOF -> sub4.sh -> watch score.
# Markers: FOLDS DONE | WEIGHTS | SUB4 submitted | AUTO_SUB4 FAILED | AUTO_SUB4 SCORE
cd /home/abdulllkhan/Documents/RSNA || exit 1
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"; KG="../multi_step_tool_attack/.venv/bin/kaggle"
log(){ echo "[$(date '+%F %T')] $*"; }
while ! grep -q 'ALL 384 FOLDS DONE' logs/effv2s384_folds.log 2>/dev/null; do sleep 600; done
while ps -eo cmd | grep -qE '[p]ython[^ ]* -m src\.rsna\.train_'; do sleep 60; done
log "FOLDS DONE: $(grep -E 'FOLD [0-9] DONE' logs/effv2s384_folds.log | wc -l) folds; stitched 384 OOF:"
.venv/bin/python dev/gold_oof.py effv2s_k24_384 2>&1 | grep -E 'GOLD macro|OOF macro'
.venv/bin/python dev/fit_arm_weights.py effv2s_k24 effv2s_k24_384 hr > logs/fit_arm_weights.log 2>&1
cat logs/fit_arm_weights.log
W=$(grep -oE 'RSNA_ARM_WEIGHTS="[^"]+"' logs/fit_arm_weights.log | head -1 | cut -d'"' -f2); [ -n "$W" ] || W="base=1,hr=1"
W="$W,plane=0.5"; log "WEIGHTS: $W"
bash dev/sub4.sh "$W" > logs/sub4.log 2>&1
if grep -q 'SUB4 DONE' logs/sub4.log; then log "SUB4 submitted"; else log "AUTO_SUB4 FAILED (logs/sub4.log): $(grep -E 'FAILED' logs/sub4.log | tail -1)"; exit 1; fi
REF=$($KG competitions submissions rsna-knee-abnormality-detection --csv 2>/dev/null | python3 -c "import csv,sys; r=list(csv.DictReader(sys.stdin)); print(r[0]['ref'])")
log "watching score for ref $REF"
bash dev/watch_score.sh "$REF" > logs/score_watch4.log 2>&1
log "AUTO_SUB4 SCORE: $(grep -E '^SCORE' logs/score_watch4.log | tail -1)"
