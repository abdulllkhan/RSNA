#!/usr/bin/env python
"""Measure SLOT-PHASE TTA for the v2 arm on OOF, before paying S x forward on Kaggle.

`dev/tta_eval.py` does this for the base/hr arms (whose variants come from `tta.py`). The v2
arm samples through `dataset_v2.plan_slots`, which `tta.dataset_factory` does not wrap, so it
gets its own evaluator over its own ladder -- `dataset_v2.variant_phases(S)` = 0, +1/3, -1/3,
... of ONE slot spacing, mirroring `tta.VARIANT_LADDER`'s ordering (reports/tta_v2.md).

For checkpoints `models/<name>_fold<k>.pt` this predicts fold k's HELD-OUT studies with fold
k's checkpoint -- honest OOF -- once per variant through `infer_v2.predict_studies_v2(...,
tta=[phases], return_frames=True)` on the v2 CACHE (`--cache-root data/cache_c140_384`, the
tree `train_v2.py` trained from), caches every (fold, variant) frame under `data/oof/tta_v2/`,
then stitches the folds and rank-averages the first S variants for each S you ask for, scoring:

  * macro-AUC vs binarised steven_v4 over every covered study    (n ~ 4,407 on a full run)
  * macro-AUC vs the 58 gold studies                             (true labels, noisy anchor)
  * the per-column table and the per-column delta vs S=1

FOLDS COME FROM THE LABEL FILE, not data/folds.csv: `train_v2.py` reads its split from
`data/labels/consensus_v1.parquet`'s `fold` column (`--labels`), and an evaluator that used a
different split would quietly score some studies with a checkpoint that trained on them.

Variant 0 IS the current deterministic eval sampling (`phase=0.0` is short-circuited in
`dataset_v2._slot_fracs`), so the v0 frame must reproduce the trainer's saved
`data/oof/<name>_fold<k>.parquet` up to bf16/fp32 noise -- `--compare-oof` (on by default)
prints the max |diff| per fold. The trainer wrote that parquet from a GPU/bf16 validation
pass, so a CPU run shows a small numerical gap; `--device cuda` is the strict comparison.

Resumable: a (fold, variant) frame that already exists and covers the requested studies is
reused, and only the MISSING phases are recomputed (`predict_studies_v2` accepts an explicit
phase list). Runs with `--limit-studies N` are tagged `_nN` so a smoke never poisons a full
run's cache. Frames are named `<name>_v<v><tag>_fold<k>.parquet`, which
`dev/gold_oof.py tta_v2/<name>_v<v>` globs unchanged.

SMOKE (CPU, does not touch the GPU):
    CUDA_VISIBLE_DEVICES="" nice -n 19 ionice -c3 .venv/bin/python dev/tta_eval_v2.py \
        --device cpu --folds 0 --limit-studies 4 --workers 2 --batch 1 --variants 3

FULL (GPU) -- the exact command is written to dev/kaggle/next_tta_v2_cmd by
dev/smoke_v2_tta.sh; see reports/tta_v2.md for the runtime estimate.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from src.rsna import TARGETS, ID_COL  # noqa: E402
from src.rsna.dataset_v2 import (  # noqa: E402
    SPAN_HI,
    SPAN_LO,
    build_index_v2,
    scaled_slot_counts,
    slot_instance_uids_v2,
    variant_label,
    variant_phases,
)
from src.rsna.infer_v2 import load_members_v2, predict_studies_v2  # noqa: E402
from src.rsna.metric import auc_report, load_gold  # noqa: E402
from src.rsna.order import SliceOrderResolver  # noqa: E402

# a smoke over few studies leaves single-class columns; the AUC is reported as NaN either way
warnings.filterwarnings("ignore", message=r"column \d+ has a single class present")

STEVEN_CSV = os.path.join(ROOT, "data", "labels",
                          "stevenleehans_rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
TRAIN_CSV = os.path.join(ROOT, "data", "comp_csv", "train.csv")
LABELS_PARQUET = os.path.join(ROOT, "data", "labels", "consensus_v1.parquet")


# --------------------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------------------
def frame_path(out_dir: str, name: str, fold: int, vidx: int, tag: str) -> str:
    """`data/oof/tta_v2/<name>_v<v><tag>_fold<k>.parquet`.

    The variant sits BEFORE the fold on purpose (same convention as `dev/tta_eval.py`):
    `dev/gold_oof.py` and `dev/fit_arm_weights.py` glob `{name}_fold[0-9].parquet`, so one
    variant's 5 folds are a valid `name` for them with no change to either script:

        .venv/bin/python dev/gold_oof.py        tta_v2/v2_effv2s384_lr_v1
        .venv/bin/python dev/fit_arm_weights.py effv2s_k24 tta_v2/v2_effv2s384_lr_v1 v2
    """
    return os.path.join(out_dir, f"{name}_v{vidx}{tag}_fold{fold}.parquet")


def read_frame(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if ID_COL not in df.columns:
        df = df.reset_index().rename(columns={"index": ID_COL})
    df[ID_COL] = df[ID_COL].astype(str)
    return df[[ID_COL] + list(TARGETS)]


def covers(path: str, wanted: Sequence[str]) -> bool:
    try:
        have = set(read_frame(path)[ID_COL])
    except Exception:
        return False
    return set(str(s) for s in wanted) <= have


def fold_map(labels_parquet: str) -> pd.Series:
    """`{StudyInstanceUID: fold}` — the SAME split `train_v2.py` trained on."""
    df = pd.read_parquet(labels_parquet, columns=["fold"])
    if ID_COL in df.columns:
        df = df.set_index(ID_COL)
    s = df["fold"].astype(int)
    s.index = s.index.astype(str)
    return s


def build_index(cache_root: str, split: str, order_parquet: Optional[str],
                dicom_root: Optional[str], keep: Sequence[str]):
    """The v2 manifest restricted to `keep` (keeps DataLoader-worker RAM small).

    `build_index_v2` reads `<cache_root>/study_index_v2_<split>.pkl` when it is there, so this
    never scans the cache tree. Built ONCE and reused for every fold and every variant.
    """
    t0 = time.time()
    resolver = SliceOrderResolver.from_parquet(order_parquet, dicom_root=dicom_root, split=split)
    full = build_index_v2(cache_root, split=split, order=resolver)
    keep = {str(s) for s in keep}
    idx = {s: e for s, e in full.items() if str(s) in keep}
    del full
    print(f"[index] {len(idx)}/{len(keep)} requested studies present ({time.time() - t0:.1f}s)")
    return idx


# --------------------------------------------------------------------------------------
# Prediction (per fold, per variant, resumable)
# --------------------------------------------------------------------------------------
def run_fold(fold: int, args, phases: List[float], index, side_map_path: Optional[str],
             wanted: List[str]) -> Dict[int, str]:
    """Ensure every (fold, variant) frame exists on disk; return {variant_idx: path}."""
    paths = {v: frame_path(args.out_dir, args.name, fold, v, args.tag)
             for v in range(len(phases))}
    todo = [v for v in range(len(phases))
            if args.force or not (os.path.exists(paths[v]) and covers(paths[v], wanted))]
    for v in range(len(phases)):
        if v not in todo:
            print(f"[fold {fold}] v{v}: cached -> {os.path.relpath(paths[v], ROOT)}")
    if not todo:
        return paths

    ckpt = args.ckpt_pattern.format(name=args.name, fold=fold)
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(ROOT, ckpt)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    print(f"[fold {fold}] {len(wanted)} held-out studies, {len(todo)} variant(s) to compute "
          f"({[variant_label(v, phases[v]) for v in todo]}), "
          f"ckpt={os.path.relpath(ckpt, ROOT)}")

    t0 = time.time()
    members = load_members_v2([ckpt], device=args.device)
    _, frames = predict_studies_v2(
        members, args.cache_root, args.split, batch=args.batch, workers=args.workers,
        device=args.device, study_ids=wanted, mode=args.member_mode,
        side_parquet=side_map_path, index=index,
        tta=[phases[v] for v in todo], tta_combine=args.variant_combine, return_frames=True)
    del members
    dt = time.time() - t0

    os.makedirs(args.out_dir, exist_ok=True)
    for pos, v in enumerate(todo):
        f = frames[pos].copy()
        f[ID_COL] = f[ID_COL].astype(str)
        f.to_parquet(paths[v], index=False)
        print(f"[fold {fold}] v{v}: {len(f)} rows -> {os.path.relpath(paths[v], ROOT)}")
    n = max(1, len(todo) * max(1, len(wanted)))
    print(f"[fold {fold}] {dt:.1f}s for {len(todo)} variant(s) ({dt / n:.2f}s per study-variant)")
    return paths


# --------------------------------------------------------------------------------------
# Proofs
# --------------------------------------------------------------------------------------
def _baseline_fracs(n_want: int) -> List[float]:
    """The pre-TTA `dataset_v2._slot_fracs` expression, written out independently.

    Recomputing it here rather than calling the patched function is the point: if a future
    edit changes the eval grid, this proof fails instead of agreeing with itself.
    """
    span = SPAN_HI - SPAN_LO
    return [min(SPAN_HI, max(0.0, SPAN_LO + span * (j + 0.5) / n_want)) for j in range(n_want)]


def prove_slot_identity(index, side_map: Dict[str, str], phases: List[float],
                        k_scale: float = 1.0, n_studies: int = 0) -> bool:
    """(a) phase 0 == the pre-TTA fraction grid; (b) how much each variant actually moves.

    `_slot_fracs` is the ONLY function on the plan path that the phase touches — the floor(),
    the 0..n-1 clamp, the sagittal order reversal and the flip in `plan_slots` are untouched —
    so equality of the FRACTIONS for every slot size is equality of the picked slices. The
    end-to-end version of this (the committed pre-edit module, real InstanceUIDs) is
    `dev/smoke_v2_tta.sh` section 1.
    """
    from src.rsna.dataset_v2 import _slot_fracs

    ok_prim = True
    sizes = sorted({n for ks in (k_scale, 0.75, 0.6, 0.25) for n in scaled_slot_counts(ks)}
                   | set(range(1, 25)))
    for n_want in sizes:
        if [float(x) for x in _slot_fracs(n_want, False, None, 1.0)] != _baseline_fracs(n_want):
            ok_prim = False
            print(f"  MISMATCH at n_want={n_want}")
    print(f"[proof] _slot_fracs(phase=0) == the pre-TTA expression for every slot size in "
          f"{min(sizes)}..{max(sizes)}: {ok_prim}")

    ids = list(index.keys())
    if n_studies:
        ids = ids[:n_studies]
    changed = {v: [0, 0] for v in range(1, len(phases))}
    k = 0
    for sid in ids:
        recs = index[sid]
        side = side_map.get(sid, "R")
        base = slot_instance_uids_v2(recs, side=side, k_scale=k_scale, phase=0.0)
        k = len(base)
        for v in range(1, len(phases)):
            got = slot_instance_uids_v2(recs, side=side, k_scale=k_scale, phase=phases[v])
            changed[v][0] += sum(1 for a, b in zip(got, base) if a != b)
            changed[v][1] += len(base)
    print(f"[proof] variant spread over {len(ids)} real studies (K={k}):")
    for v, (c, tot) in changed.items():
        print(f"[proof] v{v}: {c}/{tot} slices differ from the baseline "
              f"({100.0 * c / max(1, tot):.1f}%)")
    return ok_prim


def stitch(paths_per_fold: Dict[int, Dict[int, str]], vidx: int) -> pd.DataFrame:
    parts = [read_frame(p[vidx]) for p in paths_per_fold.values() if vidx in p]
    df = pd.concat(parts, ignore_index=True)
    return df[~df[ID_COL].duplicated()].reset_index(drop=True)


def score(pred: pd.DataFrame, steven: pd.DataFrame, gold: pd.DataFrame):
    """(steven report, n_steven, gold report, n_gold) for one prediction frame."""
    p = pred.set_index(ID_COL)[list(TARGETS)]
    j = steven.join(p, how="inner", rsuffix="_p")
    y = (j[list(TARGETS)].values >= 0.5).astype(int)
    yp = j[[t + "_p" for t in TARGETS]].values
    rs = auc_report(y, yp)
    g = gold.set_index(ID_COL)[list(TARGETS)].join(p, how="inner", rsuffix="_p")
    if len(g) > 1:
        rg = auc_report(g[list(TARGETS)].values, g[[t + "_p" for t in TARGETS]].values)
    else:
        rg = pd.Series(np.nan, index=list(TARGETS) + ["macro"])
    return rs, len(j), rg, len(g)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="v2_effv2s384_lr", help="checkpoint/OOF base name")
    ap.add_argument("--cache-root", default=os.path.join("data", "cache_c140_384"),
                    help="the v2 cache the checkpoints were trained from")
    ap.add_argument("--split", default="train")
    ap.add_argument("--ckpt-pattern", default=os.path.join("models", "{name}_fold{fold}.pt"))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--variants", type=int, default=3, help="S: how many slot phases to compute")
    ap.add_argument("--report-s", default="", help="comma list of S to report (default 1..S)")
    ap.add_argument("--seed", type=int, default=42, help="only affects variants beyond the ladder")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4,
                    help="torch intra-op threads on --device cpu (0 = torch's default). Capped "
                         "by default so a CPU smoke cannot starve a GPU trainer's workers.")
    ap.add_argument("--limit-studies", type=int, default=0, help="smoke: cap studies PER FOLD")
    ap.add_argument("--member-mode", default="prob", choices=["prob", "rank"])
    ap.add_argument("--variant-combine", default="rank", choices=["rank", "prob"])
    ap.add_argument("--labels", default=LABELS_PARQUET,
                    help="the parquet whose `fold` column defines the split (train_v2's --labels)")
    ap.add_argument("--side-parquet", default=os.path.join("data", "study_side.parquet"))
    ap.add_argument("--order-parquet", default=os.path.join("data", "series_order.parquet"))
    ap.add_argument("--dicom-root", default=None)
    ap.add_argument("--out-dir", default=os.path.join("data", "oof", "tta_v2"))
    ap.add_argument("--tag", default="", help="suffix for the cached frames (default: _n<limit>)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--compare-oof", dest="compare_oof", action="store_true", default=True)
    ap.add_argument("--no-compare-oof", dest="compare_oof", action="store_false")
    ap.add_argument("--prove", dest="prove", action="store_true", default=True,
                    help="prove v0 == the baseline slot plan on the real studies")
    ap.add_argument("--no-prove", dest="prove", action="store_false")
    ap.add_argument("--prove-studies", type=int, default=0, help="0 = every loaded study")
    ap.add_argument("--oof-dir", default=os.path.join("data", "oof"),
                    help="where the trainer's reference <name>_fold<k>.parquet live (read-only)")
    ap.add_argument("--check-combiner", action="store_true",
                    help="assert rank_average_variant_frames == infer_plane.rank_average_frames")
    args = ap.parse_args()

    if args.device == "cpu" and args.threads > 0:
        torch.set_num_threads(args.threads)
        print(f"[cpu] torch intra-op threads = {torch.get_num_threads()}")

    for a in ("cache_root", "out_dir", "oof_dir", "labels", "side_parquet", "order_parquet"):
        p = getattr(args, a)
        if p and not os.path.isabs(p):
            setattr(args, a, os.path.join(ROOT, p))
    if not args.tag and args.limit_studies:
        args.tag = f"_n{args.limit_studies}"
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    phases = variant_phases(args.variants, seed=args.seed)
    report_s = [int(x) for x in args.report_s.split(",") if x.strip()] or \
        list(range(1, len(phases) + 1))
    report_s = [s for s in report_s if 1 <= s <= len(phases)]

    print(f"[config] name={args.name} cache={os.path.relpath(args.cache_root, ROOT)} "
          f"folds={folds} S={len(phases)} device={args.device} batch={args.batch} "
          f"workers={args.workers} limit={args.limit_studies or 'all'} "
          f"tag={args.tag or '(none)'}")
    for v, ph in enumerate(phases):
        print(f"   {variant_label(v, ph)}")

    fo = fold_map(args.labels)
    wanted_per_fold: Dict[int, List[str]] = {}
    for k in folds:
        ids = fo.index[fo == k].tolist()
        wanted_per_fold[k] = ids[: args.limit_studies] if args.limit_studies else ids

    index = build_index(args.cache_root, args.split, args.order_parquet, args.dicom_root,
                        [s for ids in wanted_per_fold.values() for s in ids])
    for k in folds:
        wanted_per_fold[k] = [s for s in wanted_per_fold[k] if s in index]

    paths_per_fold: Dict[int, Dict[int, str]] = {}
    for k in folds:
        paths_per_fold[k] = run_fold(k, args, phases, index, args.side_parquet,
                                     wanted_per_fold[k])

    # ---- S=1 equivalence ------------------------------------------------------------------
    if args.prove:
        print("\n== S=1 equivalence proof (the slot PLAN, on the real studies) ==")
        from src.rsna.dataset_v2 import load_side_map
        prove_slot_identity(index, load_side_map(args.side_parquet), phases,
                            n_studies=args.prove_studies)

    if args.compare_oof:
        print("\n== v0 vs the TRAINER's stored OOF (written by the GPU/bf16 val pass) ==")
        worst = 0.0
        any_ref = False
        for k in folds:
            ref_p = os.path.join(args.oof_dir, f"{args.name}_fold{k}.parquet")
            if not os.path.exists(ref_p):
                print(f"  fold {k}: no reference {os.path.relpath(ref_p, ROOT)} -> skipped")
                continue
            any_ref = True
            ref = read_frame(ref_p).set_index(ID_COL)
            got = read_frame(paths_per_fold[k][0]).set_index(ID_COL)
            common = [s for s in got.index if s in ref.index]
            d = np.abs(got.loc[common][list(TARGETS)].values.astype(np.float64)
                       - ref.loc[common][list(TARGETS)].values.astype(np.float64))
            worst = max(worst, float(d.max()) if d.size else 0.0)
            print(f"  fold {k}: n={len(common)} max|diff|={d.max() if d.size else float('nan'):.3e} "
                  f"mean|diff|={d.mean() if d.size else float('nan'):.3e}")
        if any_ref:
            print(f"  WORST over folds: max|diff| = {worst:.3e}   (0 on the device that wrote "
                  f"the reference; a small gap on {args.device} vs the GPU/bf16 val pass is "
                  f"numerical, not a sampling change)")

    if args.check_combiner:
        from src.rsna.infer_plane import rank_average_frames as ref_comb
        from src.rsna.infer_tta import rank_average_variant_frames
        fr = [stitch(paths_per_fold, v) for v in range(len(phases))]
        a = rank_average_variant_frames(fr)[list(TARGETS)].values
        b = ref_comb(fr)[list(TARGETS)].values
        print(f"\n[check-combiner] |infer_tta - infer_plane| max = {np.abs(a - b).max():.3e}")

    # ---- score ---------------------------------------------------------------------------
    from src.rsna.infer_tta import rank_average_variant_frames

    steven = pd.read_csv(STEVEN_CSV)
    steven[ID_COL] = steven[ID_COL].astype(str)
    steven = steven.set_index(ID_COL)[list(TARGETS)]
    gold = load_gold(TRAIN_CSV)
    gold[ID_COL] = gold[ID_COL].astype(str)

    per_variant = {v: stitch(paths_per_fold, v) for v in range(len(phases))}
    print(f"\n== stitched OOF: {len(per_variant[0])} studies per variant ==")

    results = {}
    for S in report_s:
        frames = [per_variant[v] for v in range(S)]
        ens = frames[0] if S == 1 else rank_average_variant_frames(frames,
                                                                  mode=args.variant_combine)
        rs, ns, rg, ng = score(ens, steven, gold)
        results[S] = (rs, rg)
        print(f"\n-- S={S} ({args.variant_combine}-avg of {['v%d' % v for v in range(S)]}) --")
        print(f"   macro-AUC vs binarized steven_v4 (n={ns}): {rs['macro']:.5f}")
        print(f"   macro-AUC vs the {ng} gold studies        : {rg['macro']:.5f}"
              + ("   [too few gold in this run -> NaN]" if ng <= 1 else ""))

    if len(report_s) > 1:
        base_s = report_s[0]
        hdr = f"{'column':18s}" + "".join(f"{'S=%d' % s:>8}{'dS=%d' % s:>8}" for s in report_s)
        for which, title in ((0, "vs binarized steven_v4"), (1, "vs the 58 gold")):
            print(f"\n== per-column table ({title}), delta vs S={base_s} ==")
            print(hdr)
            for t in list(TARGETS) + ["macro"]:
                row = f"{t:18s}"
                for s in report_s:
                    v = results[s][which][t]
                    row += f"{v:>8.4f}{v - results[base_s][which][t]:>+8.4f}"
                print(row)

        best = max(report_s, key=lambda s: results[s][0]["macro"])
        d = results[best][0]["macro"] - results[base_s][0]["macro"]
        dg = results[best][1]["macro"] - results[base_s][1]["macro"]
        print(f"\nVERDICT: best S={best} (steven macro {results[best][0]['macro']:.5f}, "
              f"{d:+.5f} vs S={base_s}; gold {dg:+.5f}); Kaggle cost multiplier = {best}x the "
              f"v2 FORWARD only (the decode is shared -- reports/tta_v2.md). "
              f"PLAN §9 submit bar is >= +0.005 on BOTH anchors.")
    print(f"\n[frames] cached under {os.path.relpath(args.out_dir, ROOT)}/ as "
          f"{args.name}_v<v>{args.tag}_fold<k>.parquet -- directly globbable by the existing "
          f"scripts, e.g.\n"
          f"    .venv/bin/python dev/gold_oof.py tta_v2/{args.name}_v1{args.tag}")


if __name__ == "__main__":
    main()
