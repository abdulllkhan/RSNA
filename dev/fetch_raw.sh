#!/usr/bin/env bash
# Guarded raw-DICOM download for wave-3 (247 GB). DO NOT run while GPU training is up.
#
# The last unguarded download OOM'd the box: the zip + page cache fought the training
# dataloaders. So this script REFUSES to start unless the box is actually idle:
#   1. no `src.rsna.train` process is running,
#   2. `free -g` reports >= MIN_FREE_GB (default 80) of FREE RAM,
#   3. enough disk for the zip + the extracted tree,
#   4. the kaggle CLI and credentials are present.
# On success it downloads to data/raw, unzips, and prints the exact decode command.
#
# Usage:   bash dev/fetch_raw.sh
# Env:     MIN_FREE_GB=80  MIN_FREE_DISK_GB=560  RSNA_FETCH_FORCE_RAM=1 (skip check 2 only)
set -uo pipefail

REPO=/home/abdulllkhan/Documents/RSNA
COMP=rsna-knee-abnormality-detection
RAW="$REPO/data/raw"
KG=/home/abdulllkhan/Documents/multi_step_tool_attack/.venv/bin/kaggle
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
MIN_FREE_GB="${MIN_FREE_GB:-80}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-560}"   # ~247 GB zip + ~247 GB extracted + headroom

die() { echo "REFUSING: $*" >&2; exit 1; }

cd "$REPO" || die "repo not found: $REPO"
echo "=== fetch_raw $(date) ==="

# --- guard 1: no training running (non-overridable) ------------------------------------
if pgrep -af "src[.]rsna[.]train" >/dev/null 2>&1; then
  echo "--- running trainers ---"
  pgrep -af "src[.]rsna[.]train" | head -5
  die "src.rsna.train is running. Wait for dev/train_folds.sh to finish (folds 1-4), then re-run."
fi
if pgrep -af "decode_raw[.]py" >/dev/null 2>&1; then
  die "a decode_raw.py run is already in progress."
fi
echo "[guard] no src.rsna.train process        OK"

# --- guard 2: free RAM ------------------------------------------------------------------
FREE_GB=$(free -g | awk '/^Mem:/{print $4}')
AVAIL_GB=$(free -g | awk '/^Mem:/{print $7}')
if [ "${AVAIL_GB:-0}" -lt "$MIN_FREE_GB" ]; then   # MemAvailable (page cache is reclaimable)
  if [ "${RSNA_FETCH_FORCE_RAM:-0}" = "1" ]; then
    echo "[guard] free RAM ${FREE_GB} GB < ${MIN_FREE_GB} GB  -- FORCED past by RSNA_FETCH_FORCE_RAM=1"
  else
    free -g
    die "only ${FREE_GB} GB free RAM (available ${AVAIL_GB} GB), need >= ${MIN_FREE_GB} GB. Free the box first (the last download OOM'd it)."
  fi
else
  echo "[guard] free RAM ${FREE_GB} GB (available ${AVAIL_GB} GB) OK"
fi

# --- guard 3: disk ----------------------------------------------------------------------
mkdir -p "$RAW"
DISK_GB=$(df -BG --output=avail "$RAW" | tail -1 | tr -dc '0-9')
[ "${DISK_GB:-0}" -ge "$MIN_FREE_DISK_GB" ] || \
  die "only ${DISK_GB} GB free on $(df -h "$RAW" | tail -1 | awk '{print $6}'), need >= ${MIN_FREE_DISK_GB} GB (247 GB zip + 247 GB extracted)."
echo "[guard] disk ${DISK_GB} GB free          OK"

# --- guard 4: credentials ---------------------------------------------------------------
[ -x "$KG" ] || die "kaggle CLI not found/executable at $KG"
[ -f "$KAGGLE_CONFIG_DIR/kaggle.json" ] || die "no $KAGGLE_CONFIG_DIR/kaggle.json"
echo "[guard] kaggle CLI + credentials         OK"
echo "[guard] ALL CHECKS PASSED -> starting download (~247 GB, expect 2-3 h)"

# --- download ---------------------------------------------------------------------------
ionice -c3 nice -n 10 "$KG" competitions download -c "$COMP" -p "$RAW" || die "kaggle download failed"

ZIP="$RAW/${COMP}.zip"
[ -f "$ZIP" ] || ZIP=$(ls -S "$RAW"/*.zip 2>/dev/null | head -1)
[ -n "${ZIP:-}" ] && [ -f "$ZIP" ] || die "no zip in $RAW after download"
echo "[zip] $ZIP  ($(du -h "$ZIP" | cut -f1))"

# --- unzip ------------------------------------------------------------------------------
echo "[unzip] extracting into $RAW (expect 1-2 h) ..."
if command -v unzip >/dev/null 2>&1; then
  ionice -c3 nice -n 10 unzip -q -o "$ZIP" -d "$RAW" || die "unzip failed"
else
  ionice -c3 nice -n 10 "$REPO/.venv/bin/python" -m zipfile -e "$ZIP" "$RAW" || die "python unzip failed"
fi

echo "--- extracted ---"
for s in train_series test_series; do
  if [ -d "$RAW/$s" ]; then
    echo "  $s: $(ls "$RAW/$s" | wc -l) studies"
  fi
done
ls "$RAW"/*.csv 2>/dev/null

cat <<EOF

=== fetch_raw DONE $(date) ===
Zip kept at $ZIP  -- delete it once the caches are built to reclaim ~247 GB:
    rm -f "$ZIP"

NEXT (CPU only; still must not overlap GPU training):

    cd $REPO
    nohup .venv/bin/python dev/decode_raw.py data/raw --out data \\
        --sizes 384 512 --workers 6 > dev/decode_raw.out 2>&1 &
    tail -f dev/decode_raw.out

  -> data/cache384/{train,test}_series/<StudyUID>/<SeriesUID>/<InstanceUID>.jpg  (~27 GB)
  -> data/cache512/...                                                            (~42 GB)
  Resumable: re-run the same command after any interruption.
EOF
