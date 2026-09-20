#!/usr/bin/env bash
# Queue PART 8: after the 110 mm folds (V2 FOLDS DONE v2_effv2s384_c110), train effv2-m folds 1-4 (next_v2_cmd5 line 3).
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
until grep -q 'V2 FOLDS DONE v2_effv2s384_c110' logs/v2_c110_folds1-4.log 2>/dev/null; do sleep 300; done
log "== part 8: effv2m folds 1-4 =="
CMD_FILE=dev/kaggle/next_v2_cmd5 bash dev/train_folds_v2.sh 3 "1 2 3 4" > logs/v2_effv2m_folds1-4.log 2>&1
log "QUEUE_V2H DONE: $(tail -1 logs/v2_effv2m_folds1-4.log)"
