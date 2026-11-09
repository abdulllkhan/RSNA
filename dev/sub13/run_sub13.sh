#!/usr/bin/env bash
# Sub #13 (EXPERIMENTS A18, pre-registered): sub #10b UNCHANGED (base=0, hr=3, v2=10 with 5 lr + 5 c110 members, TTA S=3)
# + the public goodpjw2008 reader arm (3 checkpoints, k=12, probability mean, S=1) at rank weight 6.
# Bundle: every pre-existing module from git HEAD (dev/sub13/export_head_pkg.py), the two new modules from the tree.
# Gates (dev/sub7.sh): its own + READER_N=3 member lines + EVERY line of EXPECT_LOG_RE below, else NO submission.
# Timing gates = the visible-test commit run's per-study stages within 1.5x of the v14 baseline (predict[hr] 14.8 s,
# v2_decode 12.7 s, v2_forward 27.4 s): at the bounds the hidden run is ~1.5 x 5.2 h + 0.45 h (reader) = 8.25 h <= 8.5 h.
# Re-run after a stopped attempt: REVERSION=0 STAMP_OVERRIDE=<PKG_STAMP_*.txt already on Kaggle> (sub #10b pattern).
set -uo pipefail
cd /home/abdulllkhan/Documents/RSNA || exit 1
PKGDIR="${PKGDIR:?set PKGDIR to a scratch dir for the HEAD export}"
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); echo "MemAvailable ${avail} GB"
[ "$avail" -ge 30 ] || { echo "MemAvailable ${avail} GB < 30 GB -> abort"; exit 1; }
.venv/bin/python -P dev/sub13/export_head_pkg.py "$PKGDIR" || exit 1
export BUNDLE_PKG="$PKGDIR" READER_N=3 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
export V2_CKPTS="$(ls models/v2_effv2s384_lr_fold[0-4].pt models/v2_effv2s384_c110_fold[0-4].pt | tr '\n' ' ')"
export EXPECT_LOG_RE='\[plan\] arm reader +: 3 member\(s\), weight 6\.0
\[arm reader\] 3 member\(s\) loaded \(folds \[0, 1, 2\]\): k=12 res=256 fp16-autocast=True probability mean, S=1,
\[reader\] studies=3 valid=3 slots_mean=[0-9.]+ series=[0-9]+ series_err=0 guard=ok
\[arm reader\] OK: 3 studies
\[combine\] arm reader covers 3/3 studies
\[combine\] per-column rank average over arms \[.hr., .v2., .reader.\] with weights \[3\.0, 10\.0, 6\.0\]
\[driver\] pre-reader submission written \(arms \[.hr., .v2.\]\)
\[plan\] arm base +: 5 member\(s\) but weight 0\.0 
\[tta\] arm=hr S=3
10 v2 member\(s\) in 2 decode group\(s\): \[\(110\.0, 384, 5\), \(140\.0, 384, 5\)\]
\[tta\] v2: S=3 slot phase\(s\)
\[time\] stage=predict\[hr\] elapsed=([0-9]|1[0-9]|2[0-2])\.[0-9]s
\[time\] stage=v2_decode elapsed=1?[0-9]\.[0-9]s
\[time\] stage=v2_forward elapsed=([0-9]|[1-3][0-9]|4[01])\.[0-9]s
\[time\] stage=arm\[reader\] elapsed=[1-5]?[0-9]\.[0-9]s
\[time\] stage=total elapsed=1?[0-9]{1,2}\.[0-9]s
\[driver\] submission OK: rows=3 cols=13'
nice -n 19 bash dev/sub7.sh "base=0,hr=3,v2=10,reader=6" "3"
