#!/usr/bin/env bash
# CPU smokes for SLOT-PHASE TTA on the Wave-4 "v2" arm (reports/tta_v2.md).
#
# EVERYTHING here runs with CUDA_VISIBLE_DEVICES="" and `nice -n 19 ionice -c3`: a GPU
# training queue is expected to be running and must not be disturbed. Nothing scans the raw
# tree (the v2 index comes from the cached pickle) and the notebook sections touch only the
# 3-study visible-test mount.
#
#   1  planner   phase=0 picks the SAME InstanceUIDs as the COMMITTED pre-edit dataset_v2.py
#                on >= 6 real studies (bit-identity), plus how far v1/v2 actually move and
#                what the grid does at the stack edges
#   2  predictor infer_v2 tta=1 == the COMMITTED pre-edit predict_studies_v2 (same members,
#                same process, cache path) to 0.0; and the v0 frame of a tta=3 run == tta=1
#   3  eval      dev/tta_eval_v2.py, fold 0, 4 studies, S=3, on models/v2_effv2s384_lr_fold0.pt
#                (writes data/oof/tta_v2/..._n4_fold0.parquet; reports v0 vs the trainer's OOF)
#   4  notebook  RSNA_TTA=1 md5 == the pre-patch generator's, base+hr+v2 on /tmp/dicom_mount;
#                then RSNA_TTA=2: distinct rows, v2_forward ~2x, v2_decode FLAT
#   5  cmd       writes dev/kaggle/next_tta_v2_cmd (the full 5-fold GPU eval; NOT run here)
#
# Usage:  bash dev/smoke_v2_tta.sh            [all sections]
#         bash dev/smoke_v2_tta.sh 1 2        [only those sections]
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="$ROOT/.venv/bin/python"
RUN=(nice -n 19 ionice -c3 "$PY")
OUT="${TTA_V2_SMOKE_DIR:-/tmp/v2_tta_smoke}"
MOUNT="${V2_DICOM_MOUNT:-/tmp/dicom_mount}"
CACHE="${V2_CACHE_ROOT:-data/cache_c140_384}"
CKPT="${V2_CKPT:-models/v2_effv2s384_lr_fold0.pt}"
NAME="${V2_NAME:-v2_effv2s384_lr}"
BASE0="${BASE_CKPT:-models/effv2s_k24_fold0.pt}"
HR0="${HR_CKPT:-models/effv2s_k24_384_fold0.pt}"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=""      # the GPU belongs to the trainer queue
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="$ROOT"
# Few threads AND a passive wait policy: on a loaded box the default OpenMP spin-wait turns a
# nice-19 CPU smoke into minutes of system time (see dev/smoke_v2.sh header).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0

WANT="${*:-1 2 3 4 5}"
want() { case " $WANT " in *" $1 "*) return 0;; *) return 1;; esac }
hdr() { echo; echo "=================== $* ==================="; }

# The pre-edit tree: src/rsna + the notebook generator exactly as committed. Every identity
# proof below diffs against THIS, not against a re-implementation of the old arithmetic.
PRE="$OUT/pre"
ensure_pre() {
  [ -f "$PRE/src/rsna/dataset_v2.py" ] && return 0
  rm -rf "$PRE"; mkdir -p "$PRE"
  git -C "$ROOT" archive HEAD src/rsna notebook/build_notebook_multi.py | tar -x -C "$PRE" || return 1
  echo "[pre] committed HEAD tree -> $PRE"
  for f in dataset_v2.py infer_v2.py; do
    if diff -q "$PRE/src/rsna/$f" "$ROOT/src/rsna/$f" >/dev/null; then
      echo "[pre] WARNING $f is IDENTICAL to HEAD -- the edits are committed, so section 1/2"
      echo "      would be comparing a file with itself. Check out the pre-TTA revision first."
    fi
  done
}

# --------------------------------------------------------------------------------------
if want 1; then
hdr "1  planner: phase=0 is bit-identical to the committed pre-edit dataset_v2"
ensure_pre
PRE_DIR="$PRE" CACHE_ROOT="$CACHE" "${RUN[@]}" - <<'PYEOF'
import importlib.util, os, sys
import numpy as np

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
import src.rsna                                   # register the package for relative imports
from src.rsna.dataset_v2 import (SPAN_HI, SPAN_LO, K_FULL, build_index_v2, load_side_map,
                                 plan_study, scaled_slot_counts, slot_instance_uids_v2,
                                 variant_label, variant_phases, _slot_fracs)
from src.rsna.order import SliceOrderResolver

# ---- load the COMMITTED pre-edit module under its own name, inside the real package ----
pre_path = os.path.join(os.environ["PRE_DIR"], "src", "rsna", "dataset_v2.py")
spec = importlib.util.spec_from_file_location("src.rsna._dataset_v2_pre", pre_path)
pre = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pre
spec.loader.exec_module(pre)
print(f"[pre] loaded {pre_path}")
print("[pre] has a `phase` parameter:",
      "phase" in pre.plan_slots.__code__.co_varnames[:pre.plan_slots.__code__.co_argcount],
      "(must be False)")

cache = os.environ["CACHE_ROOT"]
resolver = SliceOrderResolver.from_parquet(os.path.join(ROOT, "data", "series_order.parquet"),
                                           split="train")
import pandas as pd
fo = pd.read_parquet(os.path.join(ROOT, "data", "labels", "consensus_v1.parquet"),
                     columns=["fold"])
ids = [str(s) for s in fo.index[:8]]
idx = build_index_v2(cache, split="train", order=resolver, only=ids)
side_map = load_side_map(os.path.join(ROOT, "data", "study_side.parquet"))
studies = sorted(idx)
print(f"[data] {len(studies)} studies from {cache} (need >= 6)")
assert len(studies) >= 6, "need >= 6 studies for the identity proof"


def pre_uids(recs, side):
    """The pre-edit planner reduced to InstanceUIDs (it has no `slot_instance_uids_v2`)."""
    by_uid = {r.series: r for r in recs}
    _s, picks = pre.plan_study(recs, side=side, k_scale=1.0, train=False)
    out = []
    for p in picks:
        rec = by_uid.get(p.series) if p.series is not None else None
        if not p.valid or rec is None or not rec.paths:
            out.append("")
        else:
            out.append(os.path.splitext(os.path.basename(
                rec.paths[min(p.idx, len(rec.paths) - 1)]))[0])
    return out


phases = variant_phases(3)
ok = True
moved = {v: [0, 0] for v in (1, 2)}
for sid in studies:
    recs, side = idx[sid], side_map.get(sid, "R")
    a = pre_uids(recs, side)
    b = slot_instance_uids_v2(recs, side=side, k_scale=1.0, phase=0.0)
    same = (a == b)
    ok &= same
    for v in (1, 2):
        g = slot_instance_uids_v2(recs, side=side, k_scale=1.0, phase=phases[v])
        moved[v][0] += sum(1 for x, y in zip(g, b) if x != y)
        moved[v][1] += len(b)
    print(f"  {sid[-12:]} side={side} K={len(b)} phase0==pre: {same}")
print(f"[proof] phase=0 InstanceUIDs == the committed pre-edit planner on "
      f"{len(studies)}/{len(studies)} studies: {ok}")
for v in (1, 2):
    c, t = moved[v]
    print(f"[proof] {variant_label(v, phases[v])}: {c}/{t} slices move "
          f"({100.0 * c / max(1, t):.1f}%)")

# ---- the arithmetic identity, over every slot size any k_scale can produce ----
def baseline(n):
    span = SPAN_HI - SPAN_LO
    return [min(SPAN_HI, max(0.0, SPAN_LO + span * (j + 0.5) / n)) for j in range(n)]

sizes = sorted(set(range(1, 33)) | {n for ks in (1.0, 0.75, 0.6, 0.25)
                                    for n in scaled_slot_counts(ks)})
prim = all([float(x) for x in _slot_fracs(n, False, None, 1.0)] == baseline(n) for n in sizes)
print(f"[proof] _slot_fracs(phase=0) == the pre-TTA expression for n in "
      f"{min(sizes)}..{max(sizes)}: {prim}")

# ---- train-mode determinism is untouched (the rng is drawn in the same order) ----
r1 = np.random.default_rng(7); r2 = np.random.default_rng(7)
a = pre._slot_fracs(18, True, r1, 1.0)
b = _slot_fracs(18, True, r2, 1.0)
print(f"[proof] train-mode jitter identical to the pre-edit code: {a == b}")

# ---- edges ----
print("[edges] one 12-slice slot, fractions of the stack:")
for ph, tag in ((0.0, "v0"), (+1/3, "v1"), (-1/3, "v2"), (+0.5, "v3"), (+2.0, "phase=+2 (saturates)")):
    f = _slot_fracs(12, False, None, 1.0, ph)
    print(f"   {tag:22s} first={f[0]:.4f} last={f[-1]:.4f}  (span {SPAN_LO}-{SPAN_HI})")
short = tot = 0
for sid in studies:
    sels, _p = plan_study(idx[sid], side=side_map.get(sid, "R"), k_scale=1.0)
    for s in sels:
        tot += 1
        short += int(s.rec is not None and s.rec.n < s.n_want)
print(f"[edges] slots whose chosen series is SHORTER than the slot (indices repeat, so a "
      f"phase moves fewer of them): {short}/{tot}; over the 8 studies v1 still moves "
      f"{100.0 * moved[1][0] / max(1, moved[1][1]):.1f}% of all {moved[1][1]} slices")
raise SystemExit(0 if (ok and prim) else 1)
PYEOF
echo "[section 1] rc=$?"
fi

# --------------------------------------------------------------------------------------
if want 2; then
hdr "2  predictor: infer_v2 tta=1 == the committed pre-edit predict_studies_v2"
ensure_pre
PRE_DIR="$PRE" CACHE_ROOT="$CACHE" CKPT="$CKPT" "${RUN[@]}" - <<'PYEOF'
import importlib.util, os, sys
import numpy as np
import torch

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
import src.rsna
from src.rsna import TARGETS, ID_COL
from src.rsna.dataset_v2 import build_index_v2, variant_phases
from src.rsna.infer_v2 import load_members_v2, predict_studies_v2
from src.rsna.order import SliceOrderResolver
import pandas as pd

pre_path = os.path.join(os.environ["PRE_DIR"], "src", "rsna", "infer_v2.py")
spec = importlib.util.spec_from_file_location("src.rsna._infer_v2_pre", pre_path)
pre = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pre
spec.loader.exec_module(pre)
print(f"[pre] loaded {pre_path} (tta kwarg present: "
      f"{'tta' in pre.predict_studies_v2.__wrapped__.__code__.co_varnames})")

cache = os.environ["CACHE_ROOT"]
fo = pd.read_parquet(os.path.join(ROOT, "data", "labels", "consensus_v1.parquet"),
                     columns=["fold"])
ids = [str(s) for s in fo.index[fo["fold"].astype(int) == 0][:2]]
resolver = SliceOrderResolver.from_parquet(os.path.join(ROOT, "data", "series_order.parquet"),
                                           split="train")
index = build_index_v2(cache, split="train", order=resolver, only=ids)
ids = [s for s in ids if s in index]
side = os.path.join(ROOT, "data", "study_side.parquet")
# ONE model load, three prediction calls -- the members are plain attribute bags, so the
# pre-edit predictor consumes them unchanged.
members = load_members_v2([os.environ["CKPT"]], device="cpu")

# The pre-edit predictor has no `index=`; feed it the SAME manifest by rebinding the builder in
# its own namespace (the `infer_tta._tta_index_stub` trick), so the ONLY difference under test
# is the predictor body -- not which index each side happened to build.
pre.build_index_v2 = lambda *a, **k: index
kw = dict(split="train", batch=1, workers=2, device="cpu", study_ids=ids, mode="prob",
          side_parquet=side, order_parquet=os.path.join(ROOT, "data", "series_order.parquet"))
ref = pre.predict_studies_v2(members, cache, **kw)
got = predict_studies_v2(members, cache, tta=1, index=index, **kw)
comb, frames = predict_studies_v2(members, cache, tta=3, tta_combine="rank",
                                  return_frames=True, index=index, **kw)

def dmax(a, b):
    a = a.set_index(ID_COL).loc[ids][list(TARGETS)].values.astype(np.float64)
    b = b.set_index(ID_COL).loc[ids][list(TARGETS)].values.astype(np.float64)
    return float(np.abs(a - b).max())

d1 = dmax(ref, got)
d0 = dmax(got, frames[0])
print(f"[proof] tta=1 vs the committed pre-edit predictor : max|diff| = {d1:.3e} "
      f"-> {'EXACT' if d1 == 0.0 else 'NOT EXACT'}")
print(f"[proof] tta=3 variant-0 frame vs tta=1            : max|diff| = {d0:.3e} "
      f"-> {'EXACT' if d0 == 0.0 else 'NOT EXACT'}")
for v in (1, 2):
    print(f"[proof] tta=3 v{v} vs v0 (must be NON-zero)         : max|diff| = "
          f"{dmax(frames[0], frames[v]):.3e}")
print(f"[proof] combined S=3 frame rows={len(comb)} cols={len(comb.columns)}")
raise SystemExit(0 if (d1 == 0.0 and d0 == 0.0) else 1)
PYEOF
echo "[section 2] rc=$?"
fi

# --------------------------------------------------------------------------------------
if want 3; then
hdr "3  dev/tta_eval_v2.py: fold 0, 4 studies, S=3"
"${RUN[@]}" dev/tta_eval_v2.py --name "$NAME" --cache-root "$CACHE" --device cpu --folds 0 \
    --limit-studies 4 --variants 3 --batch 1 --workers 2 --threads "$OMP_NUM_THREADS" \
    --check-combiner 2>&1 | grep -vE "^\[v2\] index built|^ *\[order\]"
echo "[section 3] rc=$?"
ls -1 data/oof/tta_v2/${NAME}_v*_n4_fold0.parquet 2>/dev/null
fi

# --------------------------------------------------------------------------------------
if want 4; then
hdr "4  notebook: RSNA_TTA=1 md5 unchanged, RSNA_TTA=2 doubles only the v2 forward"
ensure_pre
W="$OUT/weights"; rm -rf "$W"; mkdir -p "$W"
for cp in "$BASE0" "$HR0" "$CKPT"; do ln -sf "$ROOT/$cp" "$W/$(basename "$cp")"; done
ls -l "$W" | tail -n +2 | awk '{print "  [weights]", $9}'

"${RUN[@]}" "$PRE/notebook/build_notebook_multi.py" | tail -1
"${RUN[@]}" notebook/build_notebook_multi.py | tail -1

run_nb() {   # $1 standalone  $2 out.csv  $3 RSNA_TTA  $4 logfile
  RSNA_INPUT_ROOT="$MOUNT" RSNA_WEIGHTS_DIR="$W" RSNA_OUT="$2" RSNA_TTA="$3" \
  RSNA_DEVICE=cpu RSNA_WORKERS=2 RSNA_BATCH=8 RSNA_BATCH_V2=1 \
  RSNA_ARM_WEIGHTS="base=1,hr=3,v2=8" \
    "${RUN[@]}" "$1" > "$4" 2>&1
  echo "  rc=$? -> $4"
}
run_nb "$PRE/notebook/submission_multi_standalone.py" "$OUT/sub_pre_tta1.csv" 1 "$OUT/pre_tta1.log"
run_nb "$ROOT/notebook/submission_multi_standalone.py" "$OUT/sub_post_tta1.csv" 1 "$OUT/post_tta1.log"
run_nb "$ROOT/notebook/submission_multi_standalone.py" "$OUT/sub_post_tta2.csv" 2 "$OUT/post_tta2.log"

echo "-- md5 (RSNA_TTA=1: pre-patch vs post-patch) --"
md5sum "$OUT/sub_pre_tta1.csv" "$OUT/sub_post_tta1.csv"
cmp -s "$OUT/sub_pre_tta1.csv" "$OUT/sub_post_tta1.csv" \
  && echo "RSNA_TTA=1 BYTE-IDENTICAL" || echo "RSNA_TTA=1 DIFFERS  <-- FAIL"
echo "-- RSNA_TTA=2 --"
grep -aE "^\[tta\] arm=" "$OUT/post_tta2.log"
md5sum "$OUT/sub_post_tta2.csv"
cmp -s "$OUT/sub_post_tta1.csv" "$OUT/sub_post_tta2.csv" \
  && echo "S=2 output == S=1  <-- FAIL (TTA is a no-op)" || echo "S=2 output DIFFERS from S=1 (expected)"
nd=$(cut -d, -f2- "$OUT/sub_post_tta2.csv" | tail -n +2 | sort -u | wc -l)
echo "distinct prediction rows at S=2: $nd (rows: $(( $(wc -l < "$OUT/sub_post_tta2.csv") - 1 )))"
echo "-- v2 stage timers: decode must stay FLAT, forward must ~double --"
for f in "$OUT/post_tta1.log" "$OUT/post_tta2.log"; do
  echo "  $(basename "$f"): $(grep -aoE 'stage=v2_(decode|forward|assemble) elapsed=[0-9.]+s' "$f" | tr '\n' ' ')"
done
"$PY" - "$OUT/post_tta1.log" "$OUT/post_tta2.log" <<'PYEOF'
import re, sys
def st(p):
    t = dict(re.findall(r"stage=v2_(\w+) elapsed=([0-9.]+)s", open(p, errors="ignore").read()))
    return {k: float(v) for k, v in t.items()}
a, b = st(sys.argv[1]), st(sys.argv[2])
for k in ("decode", "forward", "assemble"):
    if k in a and k in b:
        r = b[k] / max(1e-9, a[k])
        print(f"  v2_{k:9s} S=1 {a[k]:7.1f}s  S=2 {b[k]:7.1f}s  ratio {r:.2f}x")
# The forward ratio is BELOW 2 on a 3-study mount because the one-time oneDNN/allocator
# warm-up inside the first forward is amortised over 6 batches at S=2 instead of 3 at S=1
# (and this box is running GPU trainers, so CPU timings are noisy). What must hold is the
# SHAPE: forward grows with S, decode does not.
fr = b["forward"] / max(1e-9, a["forward"])
dr = b["decode"] / max(1e-9, a["decode"])
ok = 1.35 <= fr <= 2.6 and dr <= 1.35
print(f"VERDICT: {'PASS' if ok else 'CHECK'} -- forward x{fr:.2f} (grows with S), "
      f"decode x{dr:.2f} (flat: the S bags share one decode)")
PYEOF
fi

# --------------------------------------------------------------------------------------
if want 5; then
hdr "5  the full 5-fold GPU eval command (written, NOT run)"
mkdir -p dev/kaggle
cat > dev/kaggle/next_tta_v2_cmd <<EOF
CUDA_VISIBLE_DEVICES=0 nice -n 19 ionice -c3 .venv/bin/python dev/tta_eval_v2.py --name $NAME --cache-root $CACHE --folds 0,1,2,3,4 --variants 3 --device cuda --batch 2 --workers 4 --check-combiner
EOF
cat dev/kaggle/next_tta_v2_cmd
echo "[note] queue it behind the current trainers (dev/gpu_queue.sh); do NOT run it here."
fi

echo; echo "smoke_v2_tta done."
