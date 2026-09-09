#!/usr/bin/env python
"""A3 (plane arm) gates — run BEFORE launching `src.rsna.train_plane`. Exits non-zero on failure.

Proves, against the real study index:

  1. PURITY      — every slot of the Sagittal/Coronal/Axial block is a slice of THAT plane
                   (eval mode AND train mode, even and uneven `k_per_plane`).
  2. IDENTITY    — with `k_per_plane=(8,8,8)` the A3 bag is the SAME 24 slices the 224 baseline
                   samples with `k=24` (because `sample_study_slices` splits its slots evenly
                   across the planes present, and every study here has all three). This is what
                   makes "A3 @224 vs the 224 baseline" a single-variable experiment.
  3. MISSING     — dropping a plane from a study's index entry still yields a fixed-shape bag,
                   the correct `plane_mask`, and filler slices drawn only from the planes that
                   remain (never from the dropped one).
  4. PIXELS      — the plane dataset reads the same pixels as `dataset._read_slice` at 224.
  5. COVERAGE    — corpus per-plane slice statistics (printed; see reports/plane_arm.md §1).

Usage:
    .venv/bin/python dev/plane_gates.py                          # 224 tree, cached index if present
    .venv/bin/python dev/plane_gates.py --root data/cache384 --size 384
    .venv/bin/python dev/plane_gates.py --index-cache data/oof/study_index_train.pkl -n 500
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rsna.dataset import _read_slice, sample_study_slices  # noqa: E402
from src.rsna.dataset_plane import (  # noqa: E402
    PLANE_ORDER,
    StudyPlaneBagDataset,
    build_plane_index,
    path_plane_map,
    plane_coverage,
    plane_ids_for,
    sample_plane_slots,
    split_entry_by_plane,
)

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed_jpeg224", help="image tree / cache root")
    ap.add_argument("--split", default="train")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--series-csv", default=None)
    ap.add_argument("--index-cache", default=None,
                    help="reuse an existing study-index pickle instead of rescanning the tree")
    ap.add_argument("-n", "--n-studies", type=int, default=200, help="studies to check")
    ap.add_argument("--k-per-plane", type=int, nargs=3, default=[8, 8, 8])
    args = ap.parse_args()

    if args.index_cache and os.path.exists(args.index_cache):
        with open(args.index_cache, "rb") as f:
            index = pickle.load(f)
        print(f"[index] loaded {args.index_cache}: {len(index)} studies")
    else:
        index = build_plane_index(args.root, split=args.split, series_csv=args.series_csv,
                                  cache_path=args.index_cache)
        print(f"[index] built from {args.root}: {len(index)} studies")
    if not index:
        print(f"ERROR: empty index under {args.root}")
        return 2

    sids = sorted(index)[: args.n_studies]
    kpp = tuple(args.k_per_plane)
    pids = plane_ids_for(kpp)

    # ---- 1. purity, both modes, even + uneven allocation -----------------------------
    for label, kk, modes in [("even k_per_plane=%s" % (kpp,), kpp, (False, True)),
                             ("uneven k_per_plane=(12,8,4)", (12, 8, 4), (True,))]:
        pi = plane_ids_for(kk)
        bad = 0
        for tr in modes:
            for i, sid in enumerate(sids):
                pmap = path_plane_map(index[sid])
                paths, present = sample_plane_slots(index[sid], kk, train=tr,
                                                    rng=np.random.default_rng(42 + i))
                if len(paths) != sum(kk):
                    bad += 1
                    continue
                bad += sum(pmap[p] != PLANE_ORDER[pi[j]] for j, p in enumerate(paths))
        check(f"per-plane purity, {label}", bad == 0,
              f"{len(sids)} studies x {len(modes)} mode(s) x {sum(kk)} slots, {bad} wrong-plane slots")

    # ---- 2. (8,8,8) reproduces the baseline's K=24 mixed bag --------------------------
    if tuple(kpp) == (8, 8, 8):
        same = 0
        for i, sid in enumerate(sids):
            e = index[sid]
            a = sample_study_slices(e, 24, train=False, rng=np.random.default_rng(42 + i))
            b, _pres = sample_plane_slots(e, kpp, train=False, rng=np.random.default_rng(42 + i))
            same += int(a == b)
        check("A3 (8,8,8) bag == 224 baseline K=24 bag (single-variable experiment)",
              same == len(sids), f"{same}/{len(sids)} studies identical")

    # ---- 3. missing-plane handling ----------------------------------------------------
    entry = index[sids[0]]
    ok = True
    print("    study %s series/plane: %s" % (
        sids[0][-12:], {p: len(v) for p, v in split_entry_by_plane(entry).items()}))
    for drop in PLANE_ORDER:
        sub = [e for e in entry if e[1] != drop]
        if not sub:
            continue
        pmap = path_plane_map(sub)
        paths, present = sample_plane_slots(sub, kpp, train=False, rng=np.random.default_rng(0))
        exp = [p != drop for p in PLANE_ORDER]
        fill = {pmap[p] for j, p in enumerate(paths) if PLANE_ORDER[pids[j]] == drop}
        pure = all(pmap[p] == PLANE_ORDER[pids[j]]
                   for j, p in enumerate(paths) if PLANE_ORDER[pids[j]] != drop)
        good = len(paths) == sum(kpp) and present == exp and drop not in fill and bool(fill) and pure
        print(f"    drop {drop:<9s} slots={len(paths)} mask={present} fill_from={sorted(fill)} pure={pure}")
        ok &= good
    for keep in PLANE_ORDER:
        sub = [e for e in entry if e[1] == keep]
        if not sub:
            continue
        paths, present = sample_plane_slots(sub, kpp, train=False, rng=np.random.default_rng(0))
        ok &= (len(paths) == sum(kpp)) and present == [p == keep for p in PLANE_ORDER]
    check("missing-plane handling (shape, mask, fill source, surviving-plane purity)", ok)

    # ---- 4. dataset item + pixel equality --------------------------------------------
    mini = {s: index[s] for s in sids[:4]}
    ds = StudyPlaneBagDataset(list(mini), mini, labels=None, k_per_plane=kpp, train=False,
                              size=args.size)
    x, y, mask, sid = ds[0]
    shape_ok = tuple(x.shape) == (sum(kpp), 3, args.size, args.size) and mask.dtype == torch.bool
    check("dataset item shape/dtype/mask", shape_ok,
          f"x={tuple(x.shape)} y={tuple(y.shape)} mask={mask.tolist()}")
    check("slot planes == plane_ids", ds.slot_planes(0) == [PLANE_ORDER[p] for p in pids])

    mutated = {s: [e for e in mini[s] if e[1] != "Axial"] for s in mini}
    if all(mutated.values()):
        ds2 = StudyPlaneBagDataset(list(mutated), mutated, labels=None, k_per_plane=kpp,
                                   train=False, size=args.size)
        x2, _y2, m2, _s2 = ds2[0]
        check("missing-Axial item well-formed", m2.tolist() == [True, True, False]
              and tuple(x2.shape) == (sum(kpp), 3, args.size, args.size)
              and bool(torch.isfinite(x2).all()), f"mask={m2.tolist()}")

    if args.size == 224 and not any(p.lower().endswith(".dcm") for p in ds.slot_paths_plane(0)[0]):
        paths, _ = ds.slot_paths_plane(0)
        ref = np.stack([_read_slice(p) for p in paths])
        got = (x.numpy() * np.asarray(ds.std) + np.asarray(ds.mean))[:, 0]
        d = float(np.abs(got - ref).max())
        check("pixels == dataset._read_slice at 224", d < 1e-5, f"max|diff|={d:.3e}")

    # ---- 5. coverage ------------------------------------------------------------------
    cov = plane_coverage(index)
    print(f"[coverage] studies={cov['n_studies']}  all-three-planes={cov['n_all_three']}")
    for p in PLANE_ORDER:
        s = cov["slices_per_plane"][p]
        print(f"    {p:<9s} missing={cov['missing_counts'][p]:5d} "
              f"({100*cov['missing_frac'][p]:5.2f}%)  slices/study mean={s['mean']:6.1f} "
              f"median={s['median']:5.0f} p10={s['p10']:5.0f} p90={s['p90']:5.0f}")
    print(f"[coverage] plane combos: {cov['plane_combos']}")

    if FAILURES:
        print(f"\nFAIL: {len(FAILURES)} gate(s) failed: {FAILURES}")
        return 1
    print("\nPASS: all A3 plane gates green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
