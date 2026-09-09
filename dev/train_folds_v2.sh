#!/usr/bin/env bash
# Train folds 1-4 (default) of one v2 config: dev/train_folds_v2.sh <line-no in dev/kaggle/next_v2_cmd> [folds "1 2 3 4"]
# Waits for a free GPU (no trainer). Logs logs/<name>_fold<k>.log. Markers: V2 FOLD k DONE | V2 FOLDS DONE <name> | V2 FOLDS FAILED
cd /home/abdulllkhan/Documents/RSNA || exit 1
LINE="${1:?line number}"; FOLDS="${2:-1 2 3 4}"; CMD_FILE="${CMD_FILE:-dev/kaggle/next_v2_cmd}"
log(){ echo "[$(date '+%F %T')] $*"; }
cmd=$(sed -n "${LINE}p" "$CMD_FILE"); [ -n "$cmd" ] || { log "V2 FOLDS FAILED: no line $LINE"; exit 1; }
name=$(echo "$cmd" | grep -oE -- '--name [^ ]+' | awk '{print $2}')
while ps -eo cmd | grep -qE '[p]ython[^ ]* -m src\.rsna\.train_'; do sleep 60; done
for k in $FOLDS; do
  c=$(echo "$cmd" | sed "s/--fold 0/--fold $k/"); log "== $name fold $k =="
  bash -c "$c" > "logs/${name}_fold${k}.log" 2>&1; rc=$?
  log "V2 FOLD $k DONE rc=$rc: $(grep -E '^\[epoch' logs/${name}_fold${k}.log | tail -1)"
  [ "$rc" -eq 0 ] || log "V2 FOLDS FAILED: $name fold $k rc=$rc"
done
log "V2 FOLDS DONE $name"
