#!/usr/bin/env bash
# Robustness seeds (Sep 22): after TTA5B DONE, retrain the two honest v2 arms once more (the trainer is unseeded, so a
# rerun is an independent seed): c110_r2 folds 0-4, then lr_r2 folds 0-4 (~4.3 h each, ~43 h total). Purpose:
# variance reduction for the private LB (same recipe, same labels, no OOF-derived targets). Markers: SEED2 START |
# SEED2 <arm> DONE | SEED2 DONE
cd /home/abdulllkhan/Documents/RSNA || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
until grep -q 'TTA5B DONE' logs/queue_v2.log 2>/dev/null; do sleep 300; done
log "SEED2 START"
CMD_FILE=dev/kaggle/next_v2_cmd11 bash dev/train_folds_v2.sh 1 "0 1 2 3 4" > logs/v2_effv2s384_c110_r2_folds.log 2>&1
log "SEED2 c110_r2 DONE: $(tail -1 logs/v2_effv2s384_c110_r2_folds.log)"
CMD_FILE=dev/kaggle/next_v2_cmd12 bash dev/train_folds_v2.sh 1 "0 1 2 3 4" > logs/v2_effv2s384_lr_r2_folds.log 2>&1
log "SEED2 lr_r2 DONE: $(tail -1 logs/v2_effv2s384_lr_r2_folds.log)"
log "SEED2 DONE"
