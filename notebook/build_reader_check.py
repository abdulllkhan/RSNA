"""Generate the Kaggle-side reader parity notebook (sub #13, task C): private kernel abdullllkhan/rsna-knee-reader-check
(T4, no internet, competition + goodpjw2008/rsna-knee-2-5d-convnext-reader attached).

Cell 1 is the reader module block produced by build_notebook_multi.isolated_block() from the SAME source file the
sub #13 bundle embeds (RSNA_BUNDLE_PKG honoured) -- the build asserts the block is a substring of the sub #13
standalone when --bundle is given. Cell 2 runs that code on 60 fixed training studies (reader folds 0-2, held out,
sample(60, random_state=2026) of outputs/pub_reader/pubreader_gpjw_oof_provenance.csv) from the competition mount,
plus the 3 visible test studies, and writes per-checkpoint predictions + per-study slot/error counters to
/kaggle/working; it prints the python/torch/timm/cv2/pydicom versions, the pixel-data handlers, and the Transfer
Syntax UIDs of the visible test DICOMs and of the 60 train studies.

Usage: .venv/bin/python notebook/build_reader_check.py --out <push_dir> [--bundle <submission_multi_standalone.py>]
"""
import argparse
import importlib.util
import json
import os
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SEED = 2026
N = 60

DRIVER = r'''
# ===================== READER CHECK DRIVER (sub #13 task C: Kaggle-side parity) =====================
import collections
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

T0 = time.time()
OUT = "/kaggle/working"
STUDIES = __STUDIES__
print(f"[check] {len(STUDIES)} fixed training studies (reader folds 0-2 held out, sample(60, random_state=__SEED__))")


def env_report():
    import cv2
    import PIL
    import pydicom
    import timm
    print(f"[env] python {sys.version.split()[0]} torch {torch.__version__} (cuda {torch.version.cuda}, cudnn "
          f"{torch.backends.cudnn.version()}) timm {timm.__version__} cv2 {cv2.__version__} pydicom "
          f"{pydicom.__version__} numpy {np.__version__} pandas {pd.__version__} pillow {PIL.__version__}")
    print(f"[env] cpu_count {os.cpu_count()} gpu "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    for mod in ("gdcm", "pylibjpeg", "libjpeg", "openjpeg", "rle", "pyjpegls", "jpeg_ls"):
        try:
            m = __import__(mod)
            print(f"[env] import {mod}: OK {getattr(m, '__version__', '')}")
        except Exception as e:
            print(f"[env] import {mod}: missing ({type(e).__name__})")
    import pydicom.config as pc
    print("[env] pydicom legacy pixel_data_handlers: "
          + ", ".join(f"{h.__name__.split('.')[-1]}={bool(h.is_available())}" for h in pc.pixel_data_handlers))
    try:
        from pydicom import uid as U
        from pydicom.pixels import get_decoder
        for name in ("ExplicitVRLittleEndian", "ImplicitVRLittleEndian", "DeflatedExplicitVRLittleEndian",
                     "ExplicitVRBigEndian", "RLELossless", "JPEGBaseline8Bit", "JPEGExtended12Bit", "JPEGLossless",
                     "JPEGLosslessSV1", "JPEGLSLossless", "JPEGLSNearLossless", "JPEG2000Lossless", "JPEG2000",
                     "HTJ2KLossless", "HTJ2KLosslessRPCL", "HTJ2K"):
            u = getattr(U, name, None)
            if u is None:
                continue
            try:
                d = get_decoder(u)
                print(f"[env] decoder {name:30s} {u}: available={d.is_available} "
                      f"plugins={sorted(d.available_plugins)}")
            except Exception as e:
                print(f"[env] decoder {name:30s} {u}: none ({type(e).__name__})")
    except Exception as e:
        print(f"[env] pydicom.pixels decoder API unavailable ({type(e).__name__}: {e})")


def comp_root():
    for c in ("/kaggle/input/competitions/rsna-knee-abnormality-detection",
              "/kaggle/input/rsna-knee-abnormality-detection"):
        if os.path.exists(os.path.join(c, "test.csv")):
            return c
    for c in sorted(glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*")):
        if os.path.exists(os.path.join(c, "test.csv")) and os.path.isdir(os.path.join(c, "test_series")):
            return c
    return None


def transfer_syntaxes(root, split, studies):
    import pydicom
    from pydicom.uid import UID
    c, n = collections.Counter(), 0
    for st in studies:
        sd = os.path.join(root, f"{split}_series", st)
        if not os.path.isdir(sd):
            c["<study dir missing>"] += 1
            continue
        for sr in sorted(os.listdir(sd)):
            for fn in sorted(os.listdir(os.path.join(sd, sr))):
                n += 1
                try:
                    ds = pydicom.dcmread(os.path.join(sd, sr, fn), stop_before_pixels=True)
                    c[str(ds.file_meta.get("TransferSyntaxUID", "<none>"))] += 1
                except Exception as e:
                    c[f"<unreadable {type(e).__name__}>"] += 1
    desc = ", ".join(f"{k} ({UID(k).name if k[:1].isdigit() else '-'}) x{v}" for k, v in c.most_common())
    return n, desc


env_report()
root = comp_root()
print(f"[mount] competition root = {root}")
print(f"[mount] /kaggle/input: {sorted(os.listdir('/kaggle/input'))}")
test_ids = pd.read_csv(os.path.join(root, "test.csv")).StudyInstanceUID.astype(str).tolist()
n, desc = transfer_syntaxes(root, "test", test_ids)
print(f"[tsuid] visible test: {len(test_ids)} studies, {n} files: {desc}")
have = [s for s in STUDIES if os.path.isdir(os.path.join(root, "train_series", s))]
print(f"[mount] train_series present: {os.path.isdir(os.path.join(root, 'train_series'))}; "
      f"fixed train studies present: {len(have)}/{len(STUDIES)}")
if have:
    n, desc = transfer_syntaxes(root, "train", have)
    print(f"[tsuid] {len(have)} train studies, {n} files: {desc}")
paths = reader_gpjw.find_reader_checkpoints(("/kaggle/input",))
print(f"[weights] reader checkpoints: {paths}")
device = "cuda" if torch.cuda.is_available() else "cpu"
if have:
    res = reader_gpjw.predict_reader(paths, root, "train", have, device=device, batch=4, workers=4)
    reader_gpjw.write_reader_outputs(res, OUT, prefix="reader")
    print(f"[check] train run: {res['summary']}")
rt = reader_gpjw.predict_reader(paths, root, "test", test_ids, device=device, batch=4, workers=4)
reader_gpjw.write_reader_outputs(rt, OUT, prefix="reader_test")
print(f"[check] visible test run: {rt['summary']}")
print(f"[time] stage=total elapsed={time.time() - T0:.1f}s")
'''


def load_main_builder():
    spec = importlib.util.spec_from_file_location("bnm", os.path.join(HERE, "build_notebook_multi.py"))
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    return B


def pick_studies():
    p = pd.read_csv(os.path.join(ROOT, "outputs/pub_reader/pubreader_gpjw_oof_provenance.csv"))
    held = p[p.reader_fold >= 0]
    assert len(held) == 2610, len(held)
    return held.sample(N, random_state=SEED).StudyInstanceUID.astype(str).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle", help="sub #13 standalone .py: assert the reader block is byte-identical to it")
    a = ap.parse_args()
    B = load_main_builder()
    src = open(os.path.join(B.PKG, "reader_gpjw.py")).read()
    block = B.isolated_block(src, B.ISOLATED_MODULES["reader_gpjw.py"])
    if a.bundle:
        assert block in open(a.bundle).read(), "reader block differs from the sub #13 bundle's"
        print(f"reader block is byte-identical to the one in {a.bundle}")
    ids = pick_studies()
    driver = DRIVER.replace("__STUDIES__", json.dumps(ids, indent=1)).replace("__SEED__", str(SEED))
    compile(block + "\n\n" + driver, "<reader_check>", "exec")

    def code(s):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": s.splitlines(keepends=True)}

    nb = {"cells": [{"cell_type": "markdown", "metadata": {},
                     "source": ["# Sub #13 reader parity check (Kaggle side)\n", "\n",
                                "Runs the exact `reader_gpjw` block of the sub #13 bundle on 60 fixed training "
                                "studies and the visible test; writes predictions + counters to /kaggle/working.\n"]},
                    code("# ===================== isolated: src/rsna/reader_gpjw.py -> module `reader_gpjw` =====\n"
                         + block + "\n"),
                    code(driver.lstrip("\n"))],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "reader_check.ipynb"), "w") as f:
        json.dump(nb, f, indent=1)
    shutil.copy(os.path.join(HERE, "kernel-metadata-reader-check.json"), os.path.join(a.out, "kernel-metadata.json"))
    with open(os.path.join(a.out, "studies.txt"), "w") as f:
        f.write("\n".join(ids) + "\n")
    print(f"bundle modules from {B.PKG}; wrote {a.out}/reader_check.ipynb + kernel-metadata.json "
          f"({len(ids)} studies, seed {SEED})")


if __name__ == "__main__":
    main()
