#!/usr/bin/env python
"""Equivalence + sanity tests for `src/rsna/decode.py` (CPU only, no GPU, no downloads).

1. EQUIVALENCE: `decode.read_dicom_slice(path, 224) / 255` must reproduce the trained-path
   loader `dataset._read_slice_dicom(path)` (float32 [0,1] 224x224) to <= 1/255 on a
   sample of real competition DICOM. This is the proof that switching the cache builder
   and inference onto the new single code path changes nothing at 224.
2. SANITY at 384 / 512: correct shape, dtype, non-degenerate content, and decode
   throughput (slices/s single-process) for run-time estimation.

Usage:
    .venv/bin/python dev/test_decode.py [--mount /tmp/dicom_mount] [--n 24]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rsna.decode import read_dicom_slice  # noqa: E402


def list_dicoms(mount: str) -> list:
    out = []
    for split in ("train_series", "test_series"):
        base = os.path.join(mount, split)
        if not os.path.isdir(base):
            continue
        for study in sorted(os.listdir(base)):
            sdir = os.path.join(base, study)
            if not os.path.isdir(sdir):
                continue
            for series in sorted(os.listdir(sdir)):
                serdir = os.path.join(sdir, series)
                if not os.path.isdir(serdir):
                    continue
                for fn in sorted(os.listdir(serdir)):
                    if fn.lower().endswith(".dcm"):
                        out.append(os.path.join(serdir, fn))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mount", default="/tmp/dicom_mount")
    ap.add_argument("--n", type=int, default=24, help="slices to compare (spread over the tree)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = list_dicoms(args.mount)
    print(f"[mount] {args.mount}: {len(files)} .dcm files")
    if not files:
        print("FAIL: no DICOM found")
        return 1

    # Spread the sample evenly across the (study, series)-sorted file list, plus a few
    # random ones, so we cover all planes/scanners rather than one series.
    step = max(1, len(files) // args.n)
    sample = files[::step][: args.n]
    rng = random.Random(args.seed)
    sample += rng.sample(files, min(6, len(files)))
    sample = list(dict.fromkeys(sample))
    print(f"[sample] {len(sample)} slices")

    # --- 1. equivalence vs the trained loader ------------------------------------------
    from src.rsna.dataset import _read_slice_dicom  # noqa: E402  (read-only import)

    diffs, exact, n_ok = [], 0, 0
    for p in sample:
        ref = _read_slice_dicom(p)          # float32 [0,1], 224x224
        new = read_dicom_slice(p, 224)      # uint8, 224x224
        assert ref is not None and new is not None, f"decode failed on {p}"
        assert new.dtype == np.uint8 and new.shape == (224, 224), (new.dtype, new.shape)
        d = float(np.abs(ref - new.astype(np.float32) / 255.0).max())
        diffs.append(d)
        exact += int(d == 0.0)
        n_ok += 1
    mx = max(diffs)
    print(f"[equiv 224] n={n_ok}  max|diff| = {mx:.8f}  ({mx * 255:.4f}/255)  "
          f"bit-identical: {exact}/{n_ok}")
    ok_equiv = mx <= 1.0 / 255.0
    print(f"[equiv 224] <= 1/255 ? {'PASS' if ok_equiv else 'FAIL'}")

    # --- 2. sanity + throughput at the wave-3 sizes ------------------------------------
    ok_sizes = True
    for size in (224, 384, 512):
        t0 = time.time()
        stats = []
        for p in sample:
            img = read_dicom_slice(p, size)
            if img is None or img.shape != (size, size) or img.dtype != np.uint8:
                ok_sizes = False
                print(f"[size {size}] FAIL on {p}")
                break
            stats.append((int(img.min()), int(img.max()), float(img.mean())))
        dt = time.time() - t0
        mins = [s[0] for s in stats]; maxs = [s[1] for s in stats]; means = [s[2] for s in stats]
        nondegen = all(mx_ - mn_ > 32 for mn_, mx_ in zip(mins, maxs))
        ok_sizes = ok_sizes and nondegen
        print(f"[size {size}] {len(stats)} slices in {dt:.2f}s = {len(stats)/max(dt,1e-9):.1f} sl/s "
              f"| min {min(mins)} max {max(maxs)} mean {np.mean(means):.1f} "
              f"| non-degenerate: {nondegen}")

    print(f"\nRESULT: {'PASS' if (ok_equiv and ok_sizes) else 'FAIL'}")
    return 0 if (ok_equiv and ok_sizes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
