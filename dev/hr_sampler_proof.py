#!/usr/bin/env python
"""Prove the 384px HR path samples the SAME slices as the 224 baseline (CPU, seconds).

Three claims, all checked rather than asserted (see reports/hr_arm.md §1.1):

  1. the study index built over a `cache<size>` tree has the same series order, planes and
     per-series InstanceUID lists as the 224 JPEG tree;
  2. `dataset.sample_study_slices` picks identical InstanceUIDs on both, for several K, item
     indices and train/eval modes -- and, end-to-end, `StudyBagDataset.__getitem__` and
     `StudyBagDatasetHR.__getitem__` OPEN the same files in the same order (checked by
     wrapping each dataset's per-slice reader);
  3. `StudyBagDatasetHR(size=224)` on the 224 JPEG tree returns a bit-identical tensor to
     `StudyBagDataset` -- the HR path is a strict generalisation, not a re-implementation.

It needs a study whose file set exists in BOTH trees, i.e. a study decoded completely into
the cache. Defaults target the full `data/cache384`; pass --cache384 / --study for a mini
cache built from a handful of downloaded DICOM.

    .venv/bin/python dev/hr_sampler_proof.py                              # full cache384
    .venv/bin/python dev/hr_sampler_proof.py --cache384 /tmp/mini/cache384 --study <UID>
    .venv/bin/python dev/hr_sampler_proof.py --mount-cache384 /tmp/mount/cache384
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rsna import dataset as D          # noqa: E402
from src.rsna import dataset_hr as H       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JPEG224 = os.path.join(ROOT, "data", "processed_jpeg224")


def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def uids(entry, k, i=0, train=False):
    return [stem(p) for p in D.sample_study_slices(entry, k, train=train,
                                                   rng=np.random.default_rng(42 + i))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache384", default=os.path.join(ROOT, "data", "cache384"))
    ap.add_argument("--jpeg224", default=JPEG224)
    ap.add_argument("--study", default=None, help="default: a study present in both trees")
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--mount-cache384", default=None,
                    help="also compare /tmp/dicom_mount (.dcm) against its decoded 384 cache")
    ap.add_argument("--dicom-mount", default="/tmp/dicom_mount")
    args = ap.parse_args()

    idx224 = D.build_study_index(args.jpeg224, os.path.join(args.jpeg224, "train_series.csv"),
                                 split="train",
                                 cache_path=os.path.join(ROOT, "data", "oof", "study_index_train.pkl"))
    idx384 = H.build_hr_index(args.cache384, split="train", cache_path=None)
    print(f"[index] 224 tree: {len(idx224)} studies | {args.cache384}: {len(idx384)} studies")
    if not idx384:
        print(f"ERROR: empty index under {args.cache384} -- build the cache first")
        return 2

    # pick a study whose per-series file COUNTS match (i.e. fully decoded into the cache)
    def complete(s):
        a = {ser: len(ps) for ser, _pl, ps in idx224.get(s, [])}
        b = {ser: len(ps) for ser, _pl, ps in idx384[s]}
        return a and a == b

    study = args.study or next((s for s in idx384 if complete(s)), None)
    if study is None:
        print("ERROR: no study is completely present in BOTH trees; pass --study")
        return 2
    e224, e384 = idx224[study], idx384[study]
    print(f"[study] {study}\n[study] series 224={len(e224)} 384={len(e384)} "
          f"slices 224={sum(len(p) for _s,_p,p in e224)} 384={sum(len(p) for _s,_p,p in e384)}")

    ok = True
    same_tree = ([(s, pl, [stem(p) for p in ps]) for s, pl, ps in e224]
                 == [(s, pl, [stem(p) for p in ps]) for s, pl, ps in e384])
    print(f"[index-identity] series order + plane + per-series InstanceUID lists identical: {same_tree}")
    ok &= same_tree

    same_sampler = all(uids(e224, k, i, tr) == uids(e384, k, i, tr)
                       for k in (8, 16, 24, 32) for i in range(5) for tr in (False, True))
    print(f"[sampler] identical InstanceUIDs across K in (8,16,24,32) x i in 0..4 x train/eval: "
          f"{same_sampler}")
    ok &= same_sampler

    # end-to-end: record what each __getitem__ actually opens
    r224, r384 = [], []
    o_img, o_hr = D._read_image, H.read_slice_hr
    D._read_image = lambda p: (r224.append(p), o_img(p))[1]
    H.read_slice_hr = lambda p, s: (r384.append(p), o_hr(p, s))[1]
    try:
        x224, _, _ = D.StudyBagDataset([study], idx224, None, k=args.k, train=False,
                                       mean=(0.5,) * 3, std=(0.5,) * 3)[0]
        x384, _, _ = H.StudyBagDatasetHR([study], idx384, None, k=args.k, train=False, size=384,
                                         mean=(0.5,) * 3, std=(0.5,) * 3)[0]
    finally:
        D._read_image, H.read_slice_hr = o_img, o_hr
    same_e2e = [stem(p) for p in r224] == [stem(p) for p in r384]
    print(f"[end-to-end] StudyBagDataset[0] read {len(r224)} slices, "
          f"StudyBagDatasetHR[0] read {len(r384)}")
    print(f"[end-to-end] InstanceUID sequences identical (order included): {same_e2e}")
    print(f"[shapes] 224 bag {tuple(x224.shape)}  |  384 bag {tuple(x384.shape)}")
    ok &= same_e2e

    xh, _, _ = H.StudyBagDatasetHR([study], idx224, None, k=args.k, train=False, size=224,
                                   mean=(0.5,) * 3, std=(0.5,) * 3)[0]
    bit = bool(torch.equal(xh, x224))
    print(f"[equiv@224] max|StudyBagDatasetHR(size=224) - StudyBagDataset| = "
          f"{float((xh - x224).abs().max()):.8f}  bit-identical={bit}")
    ok &= bit

    if args.mount_cache384:
        im_dcm = H.build_hr_index(args.dicom_mount, split="test", cache_path=None)
        im_384 = H.build_hr_index(args.mount_cache384, split="test", cache_path=None)
        same_mount = all(uids(im_dcm[s], args.k) == uids(im_384[s], args.k) for s in im_dcm)
        print(f"[mount] {len(im_dcm)} studies: raw .dcm vs cache384 sampler picks identical: "
              f"{same_mount}")
        ok &= same_mount

    print(f"\n{'PASS' if ok else 'FAIL'}: 224 and {args.cache384} sample the same slices")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
