#!/usr/bin/env bash
# Queue PART 15 (Oct 12): CoAtNet-2 v2 member trained with the public "Raptor" recipe (reports/coat2_raptor.md).
# Every GPU job runs under the shared box lock (flock -w LOCK_WAIT logs/box.lock; dev/train_folds_v2.sh inside it
# waits for MemAvailable >= MIN_GB=60 and runs the MemAvailable watchdog). Fold 0 (dev/kaggle/next_v2_cmd14) ->
# fold-0 ADD gate (dev/fold0_add_gate.py: baseline rank-avg hr=3 + lr=5 + c110=5 vs + X at w in {2,3,5}, w chosen
# on the steven OOF only; PASS iff d(steven) >= +0.003 AND d(gold-in-fold) >= 0 at that w) -> PASS: folds 1-4,
# the lock re-taken per fold so other GPU jobs can interleave between folds.
# Launch:  nohup bash dev/queue_v2o.sh >> logs/queue_v2.log 2>&1 &
# Markers: QUEUE_V2O GATE | QUEUE_V2O PASS | QUEUE_V2O FAIL | QUEUE_V2O DONE
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
NAME=v2_coat2_c110; CMD_FILE=dev/kaggle/next_v2_cmd14; LOCK=logs/box.lock; LOCK_WAIT=${LOCK_WAIT:-14400}
export MIN_GB=${MIN_GB:-60}
T_START=$(date +%s)
[ -f "$CMD_FILE" ] || { log "QUEUE_V2O FAIL: missing $CMD_FILE"; exit 1; }
grep -q -- "--name $NAME" "$CMD_FILE" || { log "QUEUE_V2O FAIL: $CMD_FILE does not train $NAME"; exit 1; }

log "QUEUE_V2O: waiting for $LOCK (<= ${LOCK_WAIT} s) -> fold 0 of $NAME"
CMD_FILE=$CMD_FILE flock -w "$LOCK_WAIT" "$LOCK" bash dev/train_folds_v2.sh 1 "0" > logs/${NAME}_fold0_queue.log 2>&1
rc=$?
log "QUEUE_V2O: fold 0 rc=$rc: $(tail -1 logs/${NAME}_fold0_queue.log 2>/dev/null)"
OOF=data/oof/${NAME}_fold0.parquet
best=$(grep -oE '^\[done\].*best OOF_macroAUC=[0-9.]+' logs/${NAME}_fold0.log 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)
if [ "$rc" -ne 0 ] || [ -z "$best" ] || [ ! -f "$OOF" ] || [ "$(stat -c %Y "$OOF")" -lt "$T_START" ]; then
  log "QUEUE_V2O FAIL: no fresh fold-0 result (rc=$rc best=${best:-none}; see logs/${NAME}_fold0.log)"; exit 1
fi

PYTHONPATH=. .venv/bin/python dev/fold0_add_gate.py "$OOF" > logs/${NAME}_gate.txt 2>&1
grep -v -i warning logs/${NAME}_gate.txt | sed 's/^/[gate] /'
verdict=$(grep -oE '^GATE=(PASS|FAIL)' logs/${NAME}_gate.txt | cut -d= -f2)
w=$(grep -oE '^GATE_W=[0-9.]+' logs/${NAME}_gate.txt | cut -d= -f2)
doof=$(grep -oE '^GATE_DOOF=[-+0-9.]+' logs/${NAME}_gate.txt | cut -d= -f2)
dgold=$(grep -oE '^GATE_DGOLD=[-+0-9.]+' logs/${NAME}_gate.txt | cut -d= -f2)
log "QUEUE_V2O GATE: fold-0 best OOF=$best; ADD at w=${w:-?} (chosen on steven): d(steven)=${doof:-?} d(gold-in-fold)=${dgold:-?} (bars >= +0.003 / >= 0) -> ${verdict:-ERROR}"
if [ "$verdict" != "PASS" ]; then
  log "QUEUE_V2O FAIL: gate not met (${verdict:-gate script error, see logs/${NAME}_gate.txt}); GPU left idle"; exit 0
fi

log "QUEUE_V2O PASS -> folds 1-4 of $NAME (lock re-taken per fold)"
for k in 1 2 3 4; do
  t_k=$(date +%s)
  CMD_FILE=$CMD_FILE flock -w "$LOCK_WAIT" "$LOCK" bash dev/train_folds_v2.sh 1 "$k" >> logs/${NAME}_folds1-4.log 2>&1
  rc=$?
  done_k=$(grep -E '^\[done\]' logs/${NAME}_fold${k}.log 2>/dev/null | tail -1)
  if [ "$rc" -ne 0 ] || [ -z "$done_k" ] || [ "$(stat -c %Y logs/${NAME}_fold${k}.log)" -lt "$t_k" ]; then
    log "QUEUE_V2O FAIL: fold $k (rc=$rc; see logs/${NAME}_fold${k}.log)"; exit 1
  fi
  log "QUEUE_V2O: fold $k $done_k"
done
log "QUEUE_V2O DONE"
