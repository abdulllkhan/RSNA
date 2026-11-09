#!/usr/bin/env bash
# Memory guard helpers for GPU launchers on the DGX Spark GB10 (128 GB UNIFIED memory: host and
# GPU share one pool, and `nvidia-smi --query-gpu=memory.used` prints [N/A], so /proc/meminfo
# MemAvailable is the only usable signal). Written after the 2026-09-19 box hang: a 512 px run
# was launched with ~30 GB available (qwen-server held 85 GB) -> NVRM out-of-memory -> thrash ->
# Xorg blocked on the driver lock -> hard reset. Usage:  source dev/mem_guard.sh
#   mem_avail_gb                       -> prints integer GB available
#   require_mem_gb MIN [WAIT_MIN]      -> waits up to WAIT_MIN minutes (default 0) for >= MIN GB; rc 1 if not
#   mem_watchdog PID FLOOR_GB [LOG]    -> background loop: kills PID if MemAvailable < FLOOR twice in a row
mem_avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
_mg_log() { echo "[$(date '+%F %T')] [mem_guard] $*"; }
require_mem_gb() {
  local need="$1" wait_min="${2:-0}" t=0 avail
  while :; do
    avail=$(mem_avail_gb)
    if [ "$avail" -ge "$need" ]; then _mg_log "OK: MemAvailable ${avail} GB >= ${need} GB"; return 0; fi
    if systemctl --user is-active --quiet qwen-server.service 2>/dev/null; then
      _mg_log "NOTE: qwen-server.service is ACTIVE (llama-server holds ~85 GB); stop it: systemctl --user stop qwen-server"
    fi
    if [ "$t" -ge "$wait_min" ]; then _mg_log "FAIL: MemAvailable ${avail} GB < ${need} GB after ${t} min"; return 1; fi
    _mg_log "waiting: MemAvailable ${avail} GB < ${need} GB (${t}/${wait_min} min)"; sleep 60; t=$((t+1))
  done
}
mem_watchdog() {
  local pid="$1" floor="$2" log="${3:-/dev/stdout}" strikes=0 avail
  while kill -0 "$pid" 2>/dev/null; do
    avail=$(mem_avail_gb)
    if [ "$avail" -lt "$floor" ]; then
      strikes=$((strikes+1))
      if [ "$strikes" -ge 2 ]; then
        _mg_log "WATCHDOG: MemAvailable ${avail} GB < ${floor} GB twice -> killing PID ${pid} to prevent a box hang" >> "$log"
        kill -TERM "$pid" 2>/dev/null; sleep 20; kill -KILL "$pid" 2>/dev/null
        return 0
      fi
    else strikes=0; fi
    sleep 10
  done
}
