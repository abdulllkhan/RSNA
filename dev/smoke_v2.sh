#!/usr/bin/env bash
# CPU smokes for the Wave-4 "v2" arm (slot sampler + laterality + per-finding slot-prior MIL).
#
# EVERYTHING here runs on the CPU with CUDA_VISIBLE_DEVICES="" and `nice -n 19 ionice -c3`:
# a GPU training is expected to be running concurrently and must not be disturbed. Nothing
# scans the raw tree; the study index comes from the cached pickle, and the DICOM smoke
# touches only the 3-study visible-test mount.
#
#   1  dataset      6 studies -> per-slot chosen series (plane, fluid, n), ranks, valid mask,
#                   and a pixel-level PROOF that a left knee gets flipped + order-reversed
#   2  train_v2     2 iterations, effv2s @224, k_scale 0.25 -> ckpt + OOF parquet,
#                   is_v2_meta() True, dev/slim_ckpt.py loads the result
#   3  model build  coatnet_rmlp_2_rw_384 @384 and vit_small_patch14_dinov2 @336:
#                   forward on random tensors, param counts, peak CPU RSS at batch 1 x K=64
#   4  DICOM        /tmp/dicom_mount (3 test studies) end to end, if decode.read_series_v2
#                   exists: 3 distinct non-0.5 rows, side per study, [time] stages
#   5  idempotence  the same study twice -> bit-identical logits; stored vs recomputed
#                   checkpoint fingerprint
#
# Usage:  bash dev/smoke_v2.sh            [all sections]
#         bash dev/smoke_v2.sh 1 3        [only those sections]
#         V2_CACHE_ROOT=data/cache384 bash dev/smoke_v2.sh
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="$ROOT/.venv/bin/python"
RUN=(nice -n 19 ionice -c3 "$PY")
OUT="${V2_SMOKE_DIR:-/tmp/v2_smoke}"
MOUNT="${V2_DICOM_MOUNT:-/tmp/dicom_mount}"
mkdir -p "$OUT/models" "$OUT/oof"

export CUDA_VISIBLE_DEVICES=""      # the GPU belongs to the trainer
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="$ROOT"
# Few threads AND a passive wait policy. On a box already at load ~25 (the GPU trainer plus
# a cache build), the default OpenMP spin-wait turns a 30 s CPU smoke into minutes of pure
# system time — measured: utime +0.03 s per 20 s of wall clock, stime +7.1 s. PASSIVE makes
# an idle worker sleep instead of spinning, which is what a nice-19 job must do here.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0

# Prefer the v2 cache once it exists; fall back to data/cache384 (identical LAYOUT, only the
# pixels differ: no 140 mm crop, no per-series percentile). Slot logic is layout-only, so the
# stand-in exercises every code path.
# The v2 cache counts as usable only once it ships its own study_index_train.pkl — that
# pickle is the builder's "finished" marker, and scanning a half-built tree would give a
# partial index (which is why dataset_v2 refuses to cache one).
if [ -z "${V2_CACHE_ROOT:-}" ]; then
  if [ -f "$ROOT/data/cache_c140_384/study_index_train.pkl" ]; then V2_CACHE_ROOT=data/cache_c140_384
  else V2_CACHE_ROOT=data/cache384; fi
fi
LABELS="${V2_LABELS:-data/labels/consensus_v1.parquet}"
SIDES="${V2_SIDE_PARQUET:-data/study_side.parquet}"
ORDERP="${V2_ORDER_PARQUET:-data/series_order.parquet}"

WANT="${*:-1 2 3 4 5}"
want() { case " $WANT " in *" $1 "*) return 0;; *) return 1;; esac }

echo "=================================================================="
echo "v2 smokes | cache_root=$V2_CACHE_ROOT labels=$LABELS sides=$SIDES"
echo "          | out=$OUT mount=$MOUNT device=cpu (CUDA hidden)"
echo "=================================================================="

# ------------------------------------------------------------------------------------
if want 1; then
echo; echo "### 1. dataset: slots, ranks, valid mask, laterality proof #############"
V2_CACHE_ROOT="$V2_CACHE_ROOT" V2_LABELS="$LABELS" V2_SIDES="$SIDES" V2_ORDERP="$ORDERP" \
"${RUN[@]}" - <<'PY'
import os
import numpy as np, pandas as pd
from src.rsna.dataset_v2 import (SLOT_NAMES, StudyBagDatasetV2, build_index_v2, fallback_report,
                                 load_labels_v2, load_side_map, plan_study, scaled_slot_counts,
                                 slot_total)
from src.rsna.order import SliceOrderResolver

root, labels_p = os.environ["V2_CACHE_ROOT"], os.environ["V2_LABELS"]
ser = pd.read_csv(os.path.join(root, "train_series.csv")
                  if os.path.exists(os.path.join(root, "train_series.csv"))
                  else "data/comp_csv/train_series.csv")
ids = sorted(ser["StudyInstanceUID"].astype(str).unique())[:6]
res = SliceOrderResolver.from_parquet(os.environ["V2_ORDERP"], split="train")
idx = build_index_v2(root, "train", order=res, only=ids)          # filtered -> not cached
lab = load_labels_v2(labels_p)
sides = load_side_map(os.environ["V2_SIDES"])
print(f"[smoke1] slot counts k_scale=1.0 -> {scaled_slot_counts(1.0)} K={slot_total(1.0)}; "
      f"k_scale=0.75 -> {scaled_slot_counts(0.75)} K={slot_total(0.75)}")

for n, sid in enumerate(list(idx)[:6]):
    side = sides.get(sid, "R")
    sels, picks = plan_study(idx[sid], side=side, k_scale=1.0, train=False)
    print(f"\n[study {n+1}] {sid[-16:]} side={side} series={len(idx[sid])}")
    off = 0
    for s in sels:
        r = s.rec
        pk = [p for p in picks if p.slot == s.slot]
        tag = f"{r.series[-10:]} plane={r.plane} fluid={r.fluid} n={r.n}" if r else "NONE"
        print(f"   slot{s.slot} {SLOT_NAMES[s.slot]:<13} want={s.n_want:<3} {tag:<48}"
              f" fb={s.fallback or '-':<13} ranks={[p.idx for p in pk]}")
        off += s.n_want
    print(f"   valid mask ({sum(p.valid for p in picks)}/{len(picks)}): "
          f"{''.join('1' if p.valid else '0' for p in picks)}")
print(); print(" ", fallback_report())

# ---- laterality proof: the SAME study, side R vs side L ----------------------------
sid = list(idx)[0]
ds_r = StudyBagDatasetV2([sid], idx, lab, size=224, train=False, k_scale=1.0,
                         side_map={sid: "R"})
ds_l = StudyBagDatasetV2([sid], idx, lab, size=224, train=False, k_scale=1.0,
                         side_map={sid: "L"})
xr = ds_r[0][0].numpy(); xl = ds_l[0][0].numpy()
slot = ds_r[0][1].numpy()
print(f"\n[smoke1] laterality proof on {sid[-16:]} (bag {xr.shape}, no augmentation)")
ok = True
for s in range(5):
    m = slot == s
    a, b = xr[m], xl[m]
    name = SLOT_NAMES[s]
    if name.startswith("Sag"):
        # sagittal: the slice ORDER is reversed, the pixels are untouched
        d_rev = float(np.abs(a[::-1] - b).mean())
        d_same = float(np.abs(a - b).mean())
        print(f"   {name:<13} mean|L - reverse(R)| = {d_rev:.2e}   "
              f"mean|L - R| = {d_same:.2e}  -> {'REVERSED' if d_rev < 1e-6 else 'NOT reversed'}")
        ok &= d_rev < 1e-6
    else:
        # coronal / axial: every slice is mirrored left-right, the order is untouched
        d_flip = float(np.abs(a[:, :, :, ::-1] - b).mean())
        d_same = float(np.abs(a - b).mean())
        print(f"   {name:<13} mean|L - fliplr(R)|  = {d_flip:.2e}   "
              f"mean|L - R| = {d_same:.2e}  -> {'FLIPPED' if d_flip < 1e-6 else 'NOT flipped'}")
        ok &= d_flip < 1e-6
print(f"[smoke1] laterality canonicalisation: {'PASS' if ok else 'FAIL'}")
PY
fi

# ------------------------------------------------------------------------------------
if want 2; then
echo; echo "### 2. train_v2: 2 iterations -> ckpt + OOF + slim_ckpt ################"
"${RUN[@]}" -m src.rsna.train_v2 \
    --device cpu --backbone tf_efficientnetv2_s --size 224 --k-scale 0.25 \
    --batch 2 --grad-accum 1 --epochs 1 --workers 0 --lr 1e-4 \
    --cpu-threads "$OMP_NUM_THREADS" \
    --limit-studies 4 --limit-val 2 --pretrained 0 \
    --cache-root "$V2_CACHE_ROOT" --labels "$LABELS" --side-parquet "$SIDES" \
    --order-parquet "$ORDERP" --models-dir "$OUT/models" --oof-dir "$OUT/oof" \
    --name smoke_v2 --fold 0

echo "--- checkpoint metadata / routing / slim ---"
V2_OUT="$OUT" "${RUN[@]}" - <<'PY'
import os, torch, pandas as pd
from src.rsna.infer_v2 import is_v2_meta
out = os.environ["V2_OUT"]
cp = os.path.join(out, "models", "smoke_v2_fold0.pt")
st = torch.load(cp, map_location="cpu", weights_only=False)
a = st["args"]
print("ckpt keys :", sorted(st.keys()))
print("args      :", {k: a.get(k) for k in ("mode", "size", "backbone", "k_scale", "k",
                                            "k_eval", "slot_prior", "crop_mm", "pct")})
print("slots     :", a.get("slots"))
print("is_v2_meta:", is_v2_meta(a), "| fingerprint len:", len(st["fingerprint"]))
oof = os.path.join(out, "oof", "smoke_v2_fold0.parquet")
d = pd.read_parquet(oof)
print("OOF       :", oof, d.shape, list(d.columns)[:3], "...")
PY
"${RUN[@]}" dev/slim_ckpt.py "$OUT/models/smoke_v2_fold0.pt"
fi

# ------------------------------------------------------------------------------------
if want 3; then
echo; echo "### 3. model build: coatnet-384 + dinov2-small-336, K=64 ###############"
"${RUN[@]}" - <<'PY'
import time, torch
torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", 4)))
from src.rsna.model_v2 import ModelV2Config, build_model_v2, format_slot_prior

def hwm():
    for l in open("/proc/self/status"):
        if l.startswith("VmHWM"):
            return int(l.split()[1]) / 1e6
def reset():
    try: open("/proc/self/clear_refs", "w").write("5\n")
    except Exception: pass

print(format_slot_prior(0.55))
for name, size in [("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", 384),
                   ("vit_small_patch14_dinov2.lvd142m", 336)]:
    m = build_model_v2(ModelV2Config(backbone=name, size=size, pretrained=True), verbose=True)
    m.eval()
    p = m.n_params()
    x = torch.randn(1, 64, 3, size, size)
    reset(); base = hwm(); t = time.time()
    with torch.no_grad():
        y = m(x)
    dt, peak = time.time() - t, hwm()
    print(f"[smoke3] {name} @{size}: out={tuple(y.shape)} finite={bool(torch.isfinite(y).all())} "
          f"params={p['total']/1e6:.1f}M (bb {p['backbone']/1e6:.1f}M + head {p['head']/1e6:.2f}M) "
          f"batch1xK64 fwd {dt:.1f}s peak_RSS={peak:.2f} GB (base {base:.2f} GB, "
          f"activations {peak-base:.2f} GB)", flush=True)
    del m, x, y
PY
fi

# ------------------------------------------------------------------------------------
if want 4; then
echo; echo "### 4. DICOM end-to-end on $MOUNT ######################################"
if [ ! -d "$MOUNT/test_series" ]; then
  echo "[smoke4] SKIP: no $MOUNT/test_series"
elif ! "${RUN[@]}" -c "from src.rsna.infer_v2 import has_read_series_v2; raise SystemExit(0 if has_read_series_v2() else 1)"; then
  echo "[smoke4] SKIP: decode.read_series_v2 is not importable yet (another package owns it)."
  echo "         Re-run section 4 once it lands: bash dev/smoke_v2.sh 4"
else
  "${RUN[@]}" -m src.rsna.infer_v2 --input-root "$MOUNT" --split test \
      --ckpt "$OUT/models/smoke_v2_fold0.pt" --device cpu --batch 1 --workers 0 \
      --out "$OUT/sub_v2.csv"
fi
fi

# ------------------------------------------------------------------------------------
if want 5; then
echo; echo "### 5. idempotence: same study twice -> identical logits ###############"
V2_OUT="$OUT" V2_MOUNT="$MOUNT" V2_CACHE_ROOT="$V2_CACHE_ROOT" "${RUN[@]}" - <<'PY'
import os
import numpy as np, pandas as pd, torch
from src.rsna.infer_v2 import (check_fingerprint_v2, has_read_series_v2, load_members_v2,
                               predict_studies_v2)
from src.rsna.model_v2 import compute_fingerprint_v2

out, mount = os.environ["V2_OUT"], os.environ["V2_MOUNT"]
ms = load_members_v2([os.path.join(out, "models", "smoke_v2_fold0.pt")], device="cpu")
print("[smoke5] stored-vs-recomputed fingerprint:")
check_fingerprint_v2(ms[0], "cpu")
f1 = compute_fingerprint_v2(ms[0].model, "cpu", ms[0].k_eval, ms[0].size)
f2 = compute_fingerprint_v2(ms[0].model, "cpu", ms[0].k_eval, ms[0].size)
print(f"[smoke5] fingerprint recomputed twice: max|diff|={float(np.abs(f1-f2).max()):.3e} "
      f"-> {'IDENTICAL' if np.array_equal(f1, f2) else 'DIFFERS'}")

if not (has_read_series_v2() and os.path.isdir(os.path.join(mount, "test_series"))):
    print("[smoke5] SKIP the study half: no read_series_v2 / no DICOM mount")
    raise SystemExit(0)
ids = sorted(os.listdir(os.path.join(mount, "test_series")))[:1]
a = predict_studies_v2(ms, mount, "test", batch=1, device="cpu", study_ids=ids)
b = predict_studies_v2(ms, mount, "test", batch=1, device="cpu", study_ids=ids)
from src.rsna import TARGETS
d = float(np.abs(a[TARGETS].values - b[TARGETS].values).max())
print(f"[smoke5] same study predicted twice: max|diff|={d:.3e} -> "
      f"{'IDENTICAL' if d == 0.0 else 'DIFFERS'}")
print(a.round(6).to_string())
PY
fi

echo; echo "=================== v2 smokes done ==================="
