#!/usr/bin/env bash
# Queue PART 7: after the 110 mm fold-0 test (QUEUE_V2F DONE), train folds 1-4 of effv2-m (next_v2_cmd5 line 3).
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
until grep -q 'QUEUE_V2F DONE' logs/queue_v2.log 2>/dev/null; do sleep 120; done
log "== part 7: effv2m folds 1-4 =="
CMD_FILE=dev/kaggle/next_v2_cmd5 bash dev/train_folds_v2.sh 3 "1 2 3 4" > logs/v2_effv2m_folds1-4.log 2>&1
log "QUEUE_V2G DONE: $(tail -1 logs/v2_effv2m_folds1-4.log)"
