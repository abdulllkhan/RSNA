"""Repair `args['crop_mm']` in an ALREADY-TRAINED v2 checkpoint, weights untouched.

WHY THIS EXISTS
---------------
`args['crop_mm']` is not a label. `infer_v2._predict_dicom` crops the Kaggle DICOMs with it,
so a checkpoint whose value disagrees with the cache it was trained on is fed a different
physical field of view at test time than at train time — silently, with plausible numbers.
`models/v2_effv2s384_c110_fold{0..4}.pt` were trained off `data/cache_c110_384` but launched
without `--crop-mm 110` (dev/kaggle/next_v2_cmd6), so they shipped saying `crop_mm=140.0` and
the Kaggle kernel decoded 140 mm crops for them. `train_v2.py` now derives the crop from the
cache-root name and refuses to start on a disagreement; this script fixes the ones already on
disk.

WHAT IT DOES, AND ONLY THIS
---------------------------
  * copies each checkpoint to `models/full/crop_meta_backup/<basename>` (skipped if a backup
    is already there — a second run must never overwrite the pre-patch original);
  * sets `ckpt['args']['crop_mm']` (dict or `argparse.Namespace`) to `--crop-mm`;
  * rewrites the file atomically (temp + `os.replace`) so an interrupted save cannot leave a
    truncated 300 MB checkpoint behind.

Every other key — `model`, `raw_model`, `ema`, `opt`, `sched`, `fingerprint`, `mean`, `std` —
is re-saved byte-for-byte as loaded. `--verify` re-reads both files and asserts every tensor
in every state_dict is identical to the backup's.

Usage:
    PYTHONPATH=. .venv/bin/python dev/patch_ckpt_crop.py \
        models/v2_effv2s384_c110_fold{0..4}.pt --crop-mm 110 [--verify] [--dry-run]

NOT for a checkpoint a trainer is still writing (`models/v2_effv2s512_c110_fold0.pt` while
that run is alive) — patch it after the run ends.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUP_DIR = os.path.join(ROOT, "models", "full", "crop_meta_backup")


def _get_crop(args_obj):
    if isinstance(args_obj, dict):
        return args_obj.get("crop_mm")
    return getattr(args_obj, "crop_mm", None)


def _set_crop(args_obj, value: float):
    if isinstance(args_obj, dict):
        args_obj["crop_mm"] = float(value)
    else:
        setattr(args_obj, "crop_mm", float(value))


def _state_dicts(ckpt):
    return {k: v for k, v in ckpt.items()
            if isinstance(v, dict) and v and all(torch.is_tensor(t) for t in v.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt", nargs="+")
    ap.add_argument("--crop-mm", type=float, required=True)
    ap.add_argument("--backup-dir", default=BACKUP_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="re-read patched + backup and assert every tensor is identical")
    a = ap.parse_args(argv)

    os.makedirs(a.backup_dir, exist_ok=True)
    rc = 0
    for cp in a.ckpt:
        if not os.path.isfile(cp):
            print(f"[patch] MISSING {cp}")
            rc = 1
            continue
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        args_obj = ck.get("args")
        if args_obj is None:
            print(f"[patch] {cp}: no 'args' key -> SKIP")
            rc = 1
            continue
        was = _get_crop(args_obj)
        name = os.path.basename(cp)
        if was is not None and abs(float(was) - float(a.crop_mm)) < 1e-9:
            print(f"[patch] {name}: crop_mm already {float(was):g} -> nothing to do")
            del ck
            continue
        bk = os.path.join(a.backup_dir, name)
        if a.dry_run:
            print(f"[patch] DRY-RUN {name}: crop_mm {was} -> {a.crop_mm} "
                  f"(backup {'exists' if os.path.exists(bk) else 'would be written'} at {bk})")
            del ck
            continue
        if os.path.exists(bk):
            print(f"[patch] {name}: backup already at {bk} ({os.path.getsize(bk) / 1e6:.0f} MB) "
                  f"-> kept (NOT overwritten)")
        else:
            shutil.copy2(cp, bk)
            print(f"[patch] {name}: backed up -> {bk} ({os.path.getsize(bk) / 1e6:.0f} MB)")
        _set_crop(args_obj, a.crop_mm)
        tmp = cp + ".croptmp"
        torch.save(ck, tmp)
        os.replace(tmp, cp)
        print(f"[patch] {name}: crop_mm {was} -> {float(a.crop_mm):g}  "
              f"({os.path.getsize(cp) / 1e6:.0f} MB)")
        del ck

        if a.verify:
            new = torch.load(cp, map_location="cpu", weights_only=False)
            old = torch.load(bk, map_location="cpu", weights_only=False)
            got = _get_crop(new.get("args"))
            assert abs(float(got) - float(a.crop_mm)) < 1e-9, f"{name}: crop_mm is {got}"
            assert abs(float(_get_crop(old.get("args"))) - float(was)) < 1e-9, \
                f"{name}: backup crop_mm changed"
            sd_new, sd_old = _state_dicts(new), _state_dicts(old)
            assert set(sd_new) == set(sd_old), f"{name}: state_dict keys differ"
            n_t = 0
            for key in sorted(sd_new):
                assert set(sd_new[key]) == set(sd_old[key]), f"{name}/{key}: tensor keys differ"
                for t in sd_new[key]:
                    assert torch.equal(sd_new[key][t], sd_old[key][t]), f"{name}/{key}/{t}"
                    n_t += 1
            for key in ("mean", "std", "fingerprint", "targets", "epoch", "best_auc"):
                if key in old:
                    ok = (torch.equal(new[key], old[key]) if torch.is_tensor(old[key])
                          else str(new[key]) == str(old[key]))
                    assert ok, f"{name}: {key} changed"
            print(f"[patch] {name}: VERIFIED — {len(sd_new)} state_dict(s), {n_t} tensors "
                  f"identical to the backup; only args.crop_mm changed")
            del new, old
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
