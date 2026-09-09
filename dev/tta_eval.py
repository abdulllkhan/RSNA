#!/usr/bin/env python
"""Measure slot-sampling TTA on OOF before paying S x inference cost on Kaggle.

For a checkpoint set `models/<name>_fold<k>.pt` and its matching cache root, this predicts
fold k's HELD-OUT studies (data/folds.csv) with fold k's checkpoint -- honest OOF -- once per
`tta.SamplingVariant`, caches every (fold, variant) frame under `data/oof/tta/`, then stitches
the folds and rank-averages the first S variants for each S you ask for, scoring:

  * macro-AUC vs binarized steven_v4 over every covered study   (n ~ 4,407 on a full run)
  * macro-AUC vs the 58 gold studies                            (true labels, noisy anchor)
  * the per-column table and the per-column delta vs S=1

Variant 0 IS the current deterministic eval sampling (see src/rsna/tta.py), so S=1 must
reproduce the existing `data/oof/<name>_fold<k>.parquet` -- `--compare-oof` (on by default)
prints the max |diff| per fold. On CPU vs the GPU/bf16 run that produced the stored parquet a
small numerical gap is expected; run with `--device cuda` for the strict comparison.

Resumable: a (fold, variant) frame that already exists on disk and covers the requested studies
is reused, so an interrupted 5-fold x S=3 sweep restarts where it stopped. Runs with
`--limit-studies N` are tagged `_nN` so a smoke never poisons a full run's cache. Frames are named
`<name>_v<v><tag>_fold<k>.parquet`, which `dev/gold_oof.py tta/<name>_v<v>` globs unchanged.

SMOKE (CPU, does not touch the GPU):
    .venv/bin/python dev/tta_eval.py --device cpu --limit-studies 12 --workers 2 --batch 2 \
        --name effv2s_k24 --root data/processed_jpeg224 --variants 3

FULL (GPU) -- see reports/tta_tooling.md for the runtime estimate:
    .venv/bin/python dev/tta_eval.py --name effv2s_k24_384 --root data/cache384 \
        --variants 3 --device cuda --workers 8 --batch 4
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
from src.rsna.dataset import build_study_index  # noqa: E402
from src.rsna.dataset_hr import resolve_series_csv  # noqa: E402
from src.rsna.infer_tta import (  # noqa: E402
    cpu_pin_memory_off,
    predict_studies_base_tta,
    predict_studies_tta,
    rank_average_variant_frames,
)
from src.rsna.metric import auc_report, load_gold  # noqa: E402
from src.rsna.tta import sampling_variants  # noqa: E402

# a smoke over few studies leaves single-class columns; the AUC is reported as NaN either way
warnings.filterwarnings("ignore", message=r"column \d+ has a single class present")

STEVEN_CSV = os.path.join(ROOT, "data", "labels",
                          "stevenleehans_rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
TRAIN_CSV = os.path.join(ROOT, "data", "comp_csv", "train.csv")
FOLDS_CSV = os.path.join(ROOT, "data", "folds.csv")


# --------------------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------------------
def peek_size(ckpt: str) -> int:
    """Input resolution the checkpoint was trained at (`args['size']`; absent -> 224)."""
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    return int((st.get("args") or {}).get("size", 224))


def default_index_cache(root: str) -> Optional[str]:
    """An EXISTING manifest pickle for this root, or None (never create one: caches are frozen).

    `data/oof/study_index_train.pkl` is the 224 JPEG tree's manifest (written by the trainer);
    `data/cache<size>/study_index_train.pkl` is each decoded cache's own.
    """
    for cand in (os.path.join(root, "study_index_train.pkl"),
                 os.path.join(ROOT, "data", "oof", "study_index_train.pkl")):
        if os.path.exists(cand):
            # the shared oof pickle only describes the 224 JPEG tree
            if cand.endswith(os.path.join("oof", "study_index_train.pkl")) and \
                    os.path.basename(root.rstrip("/")) != "processed_jpeg224":
                continue
            return cand
    return None


def load_index(root: str, split: str, index_cache: Optional[str], keep: Sequence[str]):
    """Study manifest restricted to `keep` (keeps DataLoader-worker RAM small)."""
    t0 = time.time()
    if index_cache and os.path.exists(index_cache):
        print(f"[index] reading cached manifest {index_cache}")
        full = build_study_index(root, "", split=split, cache_path=index_cache)
    else:
        series_csv = resolve_series_csv(root, split, root=ROOT)
        print(f"[index] scanning {root} ({split}) via {series_csv} -- no cached manifest")
        full = build_study_index(root, series_csv, split=split, cache_path=None)
    keep = set(str(s) for s in keep)
    idx = {s: e for s, e in full.items() if str(s) in keep}
    del full
    print(f"[index] {len(idx)}/{len(keep)} requested studies present ({time.time()-t0:.1f}s)")
    return idx


def frame_path(out_dir: str, name: str, fold: int, vidx: int, tag: str) -> str:
    """`data/oof/tta/<name>_v<v><tag>_fold<k>.parquet`.

    The variant sits BEFORE the fold on purpose: `dev/gold_oof.py` and `dev/fit_arm_weights.py`
    both glob `data/oof/{name}_fold[0-9].parquet`, so this layout makes one variant's 5 folds a
    valid `name` for them with no change to either script:

        .venv/bin/python dev/gold_oof.py       tta/effv2s_k24_384_v1
        .venv/bin/python dev/fit_arm_weights.py effv2s_k24 tta/effv2s_k24_384_v1 hr
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


# --------------------------------------------------------------------------------------
# Prediction (per fold, per variant, resumable)
# --------------------------------------------------------------------------------------
def maybe_channels_last(members, args):
    """See --channels-last: fixes the CPU weight/input layout mismatch in the hr path."""
    on = args.channels_last == "on" or (
        args.channels_last == "auto" and args.device == "cpu" and args.path == "hr")
    if on:
        for m in members:
            m.model = m.model.to(memory_format=torch.channels_last)
    return members


def run_fold(fold: int, args, variants, index, wanted: List[str]) -> Dict[int, str]:
    """Ensure every (fold, variant) frame exists on disk; return {variant_idx: path}."""
    paths = {v.idx: frame_path(args.out_dir, args.name, fold, v.idx, args.tag) for v in variants}
    todo = [v for v in variants
            if args.force or not (os.path.exists(paths[v.idx]) and covers(paths[v.idx], wanted))]
    for v in variants:
        if v not in todo:
            print(f"[fold {fold}] {v.name}: cached -> {os.path.relpath(paths[v.idx], ROOT)}")
    if not todo:
        return paths

    ckpt = args.ckpt_pattern.format(name=args.name, fold=fold)
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(ROOT, ckpt)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    print(f"[fold {fold}] {len(wanted)} held-out studies, {len(todo)} variant(s) to compute, "
          f"ckpt={os.path.relpath(ckpt, ROOT)}")

    t0 = time.time()
    if args.path == "base":
        from src.rsna.infer import load_members
        members = load_members([ckpt], device=args.device)
        _, frames = predict_studies_base_tta(
            args.root, args.split, members, variants=todo, batch=args.batch,
            workers=args.workers, device=args.device, study_ids=wanted,
            variant_combine=args.variant_combine, index=index, return_frames=True)
    else:
        from src.rsna.infer_hr import load_members_hr
        members = maybe_channels_last(load_members_hr([ckpt], device=args.device), args)
        _, frames = predict_studies_tta(
            args.root, args.split, members, variants=todo, batch=args.batch,
            workers=args.workers, device=args.device, study_ids=wanted, mode=args.member_mode,
            variant_combine=args.variant_combine, index=index, return_frames=True)
    del members
    dt = time.time() - t0

    os.makedirs(args.out_dir, exist_ok=True)
    for vidx, f in frames.items():
        f = f.copy()
        f[ID_COL] = f[ID_COL].astype(str)
        f.to_parquet(paths[vidx], index=False)
        print(f"[fold {fold}] v{vidx}: {len(f)} rows -> {os.path.relpath(paths[vidx], ROOT)}")
    n = max(1, len(todo) * max(1, len(wanted)))
    print(f"[fold {fold}] {dt:.1f}s for {len(todo)} variant(s) "
          f"({dt / n:.2f}s per study-variant)")
    return paths


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def prove_slot_identity(index, k: int, variants) -> None:
    """v0 must pick the SAME K paths as the untouched eval sampler, for every real study."""
    from src.rsna.dataset import sample_study_slices
    from src.rsna.tta import sample_study_slices_variant
    same = 0
    changed = {v.name: [0, 0] for v in variants[1:]}   # [changed slots, total slots]
    for sid, entry in index.items():
        base = sample_study_slices(entry, k, train=False, rng=None)
        same += int(sample_study_slices_variant(entry, k, variants[0]) == base)
        for v in variants[1:]:
            got = sample_study_slices_variant(entry, k, v)
            changed[v.name][0] += sum(1 for a, b in zip(got, base) if a != b)
            changed[v.name][1] += len(base)
    print(f"[proof] slot identity: v0 bag == baseline eval bag for {same}/{len(index)} studies")
    for nm, (c, tot) in changed.items():
        print(f"[proof] {nm}: {c}/{tot} slots differ from the baseline "
              f"({100.0 * c / max(1, tot):.1f}%)")


def prove_s1_equivalence(fold: int, args, index, wanted: List[str], v0_path: str,
                         n_studies: int = 4) -> float:
    """Re-run the UNPATCHED predictor on the same device and diff it against the cached v0.

    This is the strict S=1 check: same code, same device, same weights -> the only thing that
    could differ is the sampling, and v0 does not change it. Expect exactly 0.0.
    """
    ids = list(wanted)[:n_studies]
    if not ids:
        return float("nan")
    ckpt = args.ckpt_pattern.format(name=args.name, fold=fold)
    ckpt = ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
    if args.path == "base":
        from src.rsna.infer import load_members, predict_studies
        ms = load_members([ckpt], device=args.device)
        with cpu_pin_memory_off(predict_studies, args.device):
            ref = predict_studies(args.root, args.split, ms, batch=args.batch,
                                  workers=args.workers, device=args.device, study_ids=ids)
    else:
        from src.rsna.infer_hr import load_members_hr, predict_studies_hr
        ms = maybe_channels_last(load_members_hr([ckpt], device=args.device), args)
        with cpu_pin_memory_off(predict_studies_hr, args.device):
            ref = predict_studies_hr(args.root, args.split, ms, batch=args.batch,
                                     workers=args.workers, device=args.device, study_ids=ids,
                                     mode=args.member_mode)
    del ms
    ref = ref.set_index(ID_COL)
    ref.index = ref.index.astype(str)
    got = read_frame(v0_path).set_index(ID_COL)
    common = [s for s in ref.index if s in got.index]
    d = np.abs(got.loc[common][list(TARGETS)].values.astype(np.float64)
               - ref.loc[common][list(TARGETS)].values.astype(np.float64))
    m = float(d.max()) if d.size else float("nan")
    print(f"[proof] fold {fold}: TTA v0 vs the UNPATCHED predictor on the same device "
          f"(n={len(common)}): max|diff| = {m:.3e}  ->  {'EXACT' if m == 0.0 else 'NOT EXACT'}")
    return m


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
    ap.add_argument("--name", default="effv2s_k24", help="checkpoint/OOF base name")
    ap.add_argument("--root", default=os.path.join("data", "processed_jpeg224"),
                    help="cache root matching the checkpoints' resolution")
    ap.add_argument("--split", default="train")
    ap.add_argument("--ckpt-pattern", default=os.path.join("models", "{name}_fold{fold}.pt"))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--variants", type=int, default=3, help="S: how many samplings to compute")
    ap.add_argument("--report-s", default="", help="comma list of S to report (default 1..S)")
    ap.add_argument("--seed", type=int, default=42, help="only affects variants beyond the ladder")
    ap.add_argument("--path", choices=["auto", "base", "hr"], default="auto",
                    help="auto: infer.predict_studies for 224 ckpts, infer_hr otherwise")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4,
                    help="torch intra-op threads on --device cpu (0 = leave torch's default). "
                         "Capped by default so a CPU smoke cannot starve a GPU trainer's "
                         "DataLoader workers.")
    ap.add_argument("--limit-studies", type=int, default=0, help="smoke: cap studies PER FOLD")
    ap.add_argument("--member-mode", default="prob", choices=["prob", "rank"])
    ap.add_argument("--channels-last", default="auto", choices=["auto", "on", "off"],
                    help="put the model weights in channels_last. `infer.load_members` always "
                         "does; `infer_hr.load_members_hr` does it only on CUDA, yet "
                         "model.forward converts the INPUT to channels_last regardless -- on "
                         "CPU that layout mismatch makes oneDNN reorder the weights on every "
                         "forward (measured ~10x slower at 384). auto = on for --path hr on CPU.")
    ap.add_argument("--variant-combine", default="rank", choices=["rank", "prob"])
    ap.add_argument("--out-dir", default=os.path.join("data", "oof", "tta"))
    ap.add_argument("--tag", default="", help="suffix for the cached frames (default: _n<limit>)")
    ap.add_argument("--index-cache", default="", help="manifest pickle (default: auto if present)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--compare-oof", dest="compare_oof", action="store_true", default=True)
    ap.add_argument("--no-compare-oof", dest="compare_oof", action="store_false")
    ap.add_argument("--prove", dest="prove", action="store_true", default=True,
                    help="prove v0 == the baseline sampling (slot paths + a same-device rerun)")
    ap.add_argument("--no-prove", dest="prove", action="store_false")
    ap.add_argument("--prove-studies", type=int, default=4,
                    help="studies re-run through the UNPATCHED predictor for the strict check")
    ap.add_argument("--oof-dir", default=os.path.join("data", "oof"),
                    help="where the reference <name>_fold<k>.parquet live (read-only)")
    ap.add_argument("--check-combiner", action="store_true",
                    help="assert rank_average_variant_frames == infer_plane.rank_average_frames on real frames")
    args = ap.parse_args()

    if args.device == "cpu" and args.threads > 0:
        torch.set_num_threads(args.threads)
        print(f"[cpu] torch intra-op threads = {torch.get_num_threads()}")

    for a in ("root", "out_dir", "oof_dir"):
        p = getattr(args, a)
        if not os.path.isabs(p):
            setattr(args, a, os.path.join(ROOT, p))
    if not args.tag and args.limit_studies:
        args.tag = f"_n{args.limit_studies}"
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    variants = sampling_variants(args.variants, seed=args.seed)
    report_s = [int(x) for x in args.report_s.split(",") if x.strip()] or list(range(1, len(variants) + 1))
    report_s = [s for s in report_s if 1 <= s <= len(variants)]

    if args.path == "auto":
        probe = args.ckpt_pattern.format(name=args.name, fold=folds[0])
        probe = probe if os.path.isabs(probe) else os.path.join(ROOT, probe)
        size = peek_size(probe)
        args.path = "base" if size == 224 else "hr"
        print(f"[path] {os.path.basename(probe)} size={size} -> "
              f"{'infer.predict_studies (v7 combiner)' if args.path == 'base' else 'infer_hr.predict_studies_hr'}")

    print(f"[config] name={args.name} root={os.path.relpath(args.root, ROOT)} folds={folds} "
          f"S={len(variants)} device={args.device} batch={args.batch} workers={args.workers} "
          f"limit={args.limit_studies or 'all'} tag={args.tag or '(none)'}")
    for v in variants:
        print(f"   {v.label}")

    fdf = pd.read_csv(FOLDS_CSV)
    fdf[ID_COL] = fdf[ID_COL].astype(str)
    wanted_per_fold: Dict[int, List[str]] = {}
    for k in folds:
        ids = fdf.loc[fdf.fold == k, ID_COL].tolist()
        wanted_per_fold[k] = ids[: args.limit_studies] if args.limit_studies else ids

    index = load_index(args.root, args.split,
                       args.index_cache or default_index_cache(args.root),
                       [s for ids in wanted_per_fold.values() for s in ids])
    for k in folds:
        wanted_per_fold[k] = [s for s in wanted_per_fold[k] if s in index]

    paths_per_fold: Dict[int, Dict[int, str]] = {}
    for k in folds:
        paths_per_fold[k] = run_fold(k, args, variants, index, wanted_per_fold[k])

    # ---- S=1 equivalence --------------------------------------------------------------------
    if args.prove:
        print("\n== S=1 equivalence proof ==")
        k = int((torch.load(args.ckpt_pattern.format(name=args.name, fold=folds[0])
                            if os.path.isabs(args.ckpt_pattern)
                            else os.path.join(ROOT, args.ckpt_pattern.format(name=args.name,
                                                                             fold=folds[0])),
                            map_location="cpu", weights_only=False).get("args") or {}).get("k", 24))
        prove_slot_identity(index, k, variants)
        prove_s1_equivalence(folds[0], args, index, wanted_per_fold[folds[0]],
                             paths_per_fold[folds[0]][0], n_studies=args.prove_studies)

    if args.compare_oof:
        print("\n== S=1 vs the STORED OOF (written by the GPU/bf16 training run) ==")
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
            print(f"  WORST over folds: max|diff| = {worst:.3e}   "
                  f"(0 on the device that wrote the reference; a small gap on "
                  f"{args.device} vs the GPU/bf16 training run is numerical, not a sampling change)")

    if args.check_combiner:
        from src.rsna.infer_plane import rank_average_frames as ref_comb
        fr = [stitch(paths_per_fold, v.idx) for v in variants]
        a = rank_average_variant_frames(fr)[list(TARGETS)].values
        b = ref_comb(fr)[list(TARGETS)].values
        print(f"\n[check-combiner] |infer_tta - infer_plane| max = {np.abs(a - b).max():.3e}")

    # ---- score ---------------------------------------------------------------------------
    steven = pd.read_csv(STEVEN_CSV)
    steven[ID_COL] = steven[ID_COL].astype(str)
    steven = steven.set_index(ID_COL)[list(TARGETS)]
    gold = load_gold(TRAIN_CSV)
    gold[ID_COL] = gold[ID_COL].astype(str)

    per_variant = {v.idx: stitch(paths_per_fold, v.idx) for v in variants}
    print(f"\n== stitched OOF: {len(per_variant[0])} studies per variant ==")

    results = {}
    for S in report_s:
        frames = [per_variant[v.idx] for v in variants[:S]]
        ens = frames[0] if S == 1 else rank_average_variant_frames(frames, mode=args.variant_combine)
        rs, ns, rg, ng = score(ens, steven, gold)
        results[S] = (rs, rg)
        print(f"\n-- S={S} ({args.variant_combine}-avg of {[v.name for v in variants[:S]]}) --")
        print(f"   macro-AUC vs binarized steven_v4 (n={ns}): {rs['macro']:.5f}")
        print(f"   macro-AUC vs the {ng} gold studies        : {rg['macro']:.5f}"
              + ("   [too few gold in this run -> NaN]" if ng <= 1 else ""))

    if len(report_s) > 1:
        base_s = report_s[0]
        print(f"\n== per-column table (vs binarized steven_v4), delta vs S={base_s} ==")
        hdr = f"{'column':18s}" + "".join(f"{'S=%d' % s:>8}{'dS=%d' % s:>8}" for s in report_s)
        print(hdr)
        for t in list(TARGETS) + ["macro"]:
            row = f"{t:18s}"
            for s in report_s:
                v = results[s][0][t]
                d = v - results[base_s][0][t]
                row += f"{v:>8.4f}{d:>+8.4f}"
            print(row)
        print(f"\n== per-column table (vs the 58 gold), delta vs S={base_s} ==")
        print(hdr)
        for t in list(TARGETS) + ["macro"]:
            row = f"{t:18s}"
            for s in report_s:
                v = results[s][1][t]
                d = v - results[base_s][1][t]
                row += f"{v:>8.4f}{d:>+8.4f}"
            print(row)

        best = max(report_s, key=lambda s: results[s][0]["macro"])
        d = results[best][0]["macro"] - results[base_s][0]["macro"]
        print(f"\nVERDICT: best S={best} (steven macro {results[best][0]['macro']:.5f}, "
              f"{d:+.5f} vs S={base_s}); Kaggle inference cost multiplier = {best}x. "
              f"PLAN §9 submit bar is >= +0.005 on BOTH anchors.")
    print(f"\n[frames] cached under {os.path.relpath(args.out_dir, ROOT)}/ as "
          f"{args.name}_v<v>{args.tag}_fold<k>.parquet -- that layout is directly globbable by "
          f"the existing scripts, e.g.\n"
          f"    .venv/bin/python dev/gold_oof.py tta/{args.name}_v1{args.tag}\n"
          f"    .venv/bin/python dev/fit_arm_weights.py {args.name} tta/{args.name}_v1{args.tag} tta")


if __name__ == "__main__":
    main()
