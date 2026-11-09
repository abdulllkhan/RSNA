#!/usr/bin/env bash
# Wave-4 GPU queue PART 9 (waits for the 110mm/512 cache, then reads dev/kaggle/next_v2_cmd7). Original header: fold-0 of the v2 arm with three backbones, in the order of dev/kaggle/next_v2_cmd:
#   1) v2_effv2s384 (contract-only test vs the 384 HR arm)  2) v2_coat384 (public best backbone)  3) v2_dino336
# Each step: launch, log the img/s after 10 min ("[rate]"), wait. Markers: V2 <NAME> FOLD0 DONE rc=N | [rate] | QUEUE_V2 FAILED | QUEUE_V2 DONE
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
source dev/mem_guard.sh   # MemAvailable gate + watchdog (2026-09-19 box hang post-mortem)
MIN_GB=${MIN_GB:-60}; FLOOR_GB=${FLOOR_GB:-6}
until grep -q 'C110_512 CACHE DONE' logs/decode_v2_c110_512.log 2>/dev/null; do sleep 120; done
while ps -eo cmd | grep -qE '[p]ython[^ ]* -m src\.rsna\.train_'; do sleep 60; done
i=0
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue; i=$((i+1))
  name=$(echo "$cmd" | grep -oE -- '--name [^ ]+' | awk '{print $2}')
  log "== step $i: $name fold 0 =="
  if ! require_mem_gb "$MIN_GB" 30; then log "QUEUE_V2 FAILED: MemAvailable < ${MIN_GB} GB for 30 min; NOT launching $name"; exit 1; fi
  bash -c "$cmd" > "logs/${name}_fold0.log" 2>&1 &
  pid=$!
  mem_watchdog "$pid" "$FLOOR_GB" logs/queue_v2.log &
  wd=$!
  sleep 600; log "[rate] $name: $(grep -E 'img/s' logs/${name}_fold0.log | tail -1 | sed 's/^ *//')"
  wait $pid; rc=$?
  kill $wd 2>/dev/null; wait $wd 2>/dev/null
  log "V2 $name FOLD0 DONE rc=$rc: $(grep -E '^\[epoch' logs/${name}_fold0.log | tail -1)"
  [ "$rc" -eq 0 ] || log "QUEUE_V2 FAILED: $name rc=$rc (continuing)"
done < dev/kaggle/next_v2_cmd7
log "QUEUE_V2I DONE"
