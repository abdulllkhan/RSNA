#!/usr/bin/env bash
# Detached score watcher: polls the submission until a public score appears, logs it, exits.
cd /home/abdulllkhan/Documents/RSNA || exit 1
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
KG="../multi_step_tool_attack/.venv/bin/kaggle"
COMP=rsna-knee-abnormality-detection
REF="${1:-56082798}"
echo "watch_score start $(date) ref=$REF"
for i in $(seq 1 150); do
  row=$($KG competitions submissions "$COMP" --csv 2>/dev/null | python3 -c "
import csv,sys
for r in csv.DictReader(sys.stdin):
    if r.get('ref')=='$REF': print(r.get('status',''), '|', r.get('publicScore','')); break
")
  status="${row%%|*}"; score="${row##*|}"; status="${status// /}"; score="${score// /}"
  echo "$(date +%H:%M) status=$status public=$score"
  if [ -n "$score" ] && [ "$score" != "None" ]; then echo "SCORE $score (ref $REF) $(date)"; break; fi
  echo "$status" | grep -qiE 'error|fail' && { echo "SCORE ERROR status=$status"; break; }
  sleep 600
done
echo "watch_score end $(date)"
