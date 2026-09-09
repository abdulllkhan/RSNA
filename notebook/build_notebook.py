"""Generate notebook/submission.ipynb (and a standalone .py mirror) for the Kaggle code comp.

Logic lives in src/rsna/*.py; this script INLINES that source into a self-contained
notebook (no `src` import, no network needed at Kaggle rerun). The driver honours env
overrides (RSNA_INPUT_ROOT / RSNA_WEIGHTS_DIR / RSNA_OUT / RSNA_DEVICE / RSNA_WORKERS /
RSNA_BATCH) so the EXACT generated code path can be validated locally; on Kaggle the
defaults kick in (scan /kaggle/input, write /kaggle/working/submission.csv).

Run:  .venv/bin/python notebook/build_notebook.py
"""
from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PKG = os.path.join(ROOT, "src", "rsna")

# Modules inlined in dependency order (metric.py is not needed for inference).
MODULES = ["__init__.py", "dataset.py", "model.py", "infer.py"]

_REL_IMPORT = re.compile(r"^\s*from\s+\.[\w.]*\s+import\s+.*$")
_FUTURE = re.compile(r"^\s*from\s+__future__\s+import\s+.*$")


def _strip_module(src: str) -> str:
    out = []
    for line in src.splitlines():
        if _FUTURE.match(line):
            continue
        if _REL_IMPORT.match(line) or re.match(r"^\s*from\s+\.\s+import\s+.*$", line):
            continue
        out.append(line)
    return "\n".join(out).strip("\n")


def build_bundle() -> str:
    parts = ["from __future__ import annotations", ""]
    for m in MODULES:
        with open(os.path.join(PKG, m)) as f:
            body = _strip_module(f.read())
        parts.append(f"# ===================== inlined: src/rsna/{m} =====================")
        parts.append(body)
        parts.append("")
    return "\n".join(parts)


DRIVER = r'''
# ============================== SUBMISSION DRIVER ==============================
import glob, os, time, traceback
import pandas as pd
import torch

_T_START = time.time()


def _stage(name, t0):
    """One line per stage. The efficiency metric charges TOTAL notebook wall-clock
    (reports/efficiency_prize.md §2), so the Kaggle log has to show where it went."""
    print(f"[time] stage={name} elapsed={time.time() - t0:.1f}s", flush=True)


OUT = os.environ.get("RSNA_OUT", "/kaggle/working/submission.csv")
DEVICE = os.environ.get("RSNA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = int(os.environ.get("RSNA_WORKERS", "2"))
BATCH = int(os.environ.get("RSNA_BATCH", "8"))
WEIGHTS_DIR = os.environ.get("RSNA_WEIGHTS_DIR", "/kaggle/input")

# Where the attached weights dataset actually mounts. Checked FIRST, shallowly.
WEIGHTS_HINTS = ("rsna-knee-weights-v1", "datasets/abdullllkhan/rsna-knee-weights-v1")
# NEVER descend into these: they are the DICOM trees (~819k files in the competition mount).
PRUNE_DIRS = {"train_series", "test_series", "competitions"}


def _is_member_ckpt(fn):
    return fn.endswith(".pt") and "_fold" in fn and not fn.endswith("_last.pt")


def _shallow_pt(root, max_depth=2):
    """Checkpoints at most `max_depth` directory levels under `root`. Listing directories is
    cheap; it is descending into train_series/ that costs minutes."""
    hits = []
    if not os.path.isdir(root):
        return hits
    for dp, dn, fns in os.walk(root):
        rel = os.path.relpath(dp, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        dn[:] = [] if depth >= max_depth else [d for d in dn if d not in PRUNE_DIRS]
        hits += [os.path.join(dp, fn) for fn in fns if _is_member_ckpt(fn)]
    return hits


def find_weights(search):
    """Collect trained checkpoints (best-by-fold; skip *_last.pt) from the attached dataset.

    Fast path: the known weights-dataset mount points, listed shallowly. The old unbounded
    `os.walk("/kaggle/input")` cost a MEASURED 341 s per run, because /kaggle/input carries
    the whole competition dataset (~819k training DICOMs) and we were walking all of it to
    find ten .pt files in a 12 MB attached dataset -- 0.0048 AUC-equivalent burned on every
    submission (reports/efficiency_prize.md §4). Only if the fast path comes up empty do we
    walk recursively, and even then the DICOM trees are pruned.
    """
    t0 = time.time()
    cands = [os.path.join(search, h) for h in WEIGHTS_HINTS]
    cands += [os.path.join("/kaggle/input", h) for h in WEIGHTS_HINTS]
    cands += sorted(glob.glob(os.path.join(search, "*weights*")))
    cands += sorted(glob.glob(os.path.join(search, "datasets", "*", "*weights*")))
    cands.append(search)  # the mount root itself -- still depth-bounded, still pruned
    hits, seen = [], set()
    for d in cands:
        rd = os.path.realpath(d)
        if rd in seen or not os.path.isdir(rd):
            continue
        seen.add(rd)
        hits += _shallow_pt(d)
    how = "shallow"
    if not hits:  # last resort: recursive, but never into a DICOM tree
        how = "pruned-walk"
        for dp, dn, fns in os.walk(search):
            dn[:] = [d for d in dn if d not in PRUNE_DIRS]
            hits += [os.path.join(dp, fn) for fn in fns if _is_member_ckpt(fn)]
    out = sorted(set(hits))
    print(f"[time] stage=weights_search elapsed={time.time() - t0:.1f}s via={how} "
          f"found={len(out)}", flush=True)
    return out


# Stage timers around the library entry points. `run_submission` resolves these names in the
# module globals at CALL time, so rebinding them here times the inside of it without editing
# src/rsna/infer.py. Pure pass-through: same args, same return value, same exceptions.
_load_members_inner, _predict_studies_inner, _write_submission_inner = (
    load_members, predict_studies, write_submission)


def load_members(*a, **kw):
    t = time.time()
    try:
        return _load_members_inner(*a, **kw)
    finally:
        _stage("member_load", t)


def predict_studies(*a, **kw):
    t = time.time()
    try:
        return _predict_studies_inner(*a, **kw)
    finally:
        _stage("predict", t)


def write_submission(*a, **kw):
    t = time.time()
    try:
        return _write_submission_inner(*a, **kw)
    finally:
        _stage("write", t)


def _fallback_test_df():
    root = find_data_root("/kaggle/input") or os.environ.get("RSNA_INPUT_ROOT")
    for cand in [root, "/kaggle/input/rsna-knee-abnormality-detection",
                 "/kaggle/input/rsna-knee-abnormalities-detection"]:
        if cand and os.path.exists(os.path.join(cand, "test.csv")):
            return pd.read_csv(os.path.join(cand, "test.csv"))
    raise FileNotFoundError("no test.csv found for fallback")


try:
    _t = time.time()
    root = os.environ.get("RSNA_INPUT_ROOT") or find_data_root("/kaggle/input")
    if root is None:
        raise FileNotFoundError("could not locate data mount (test.csv + test_series/)")
    test_df = pd.read_csv(os.path.join(root, "test.csv"))
    write_benchmark_submission(test_df, OUT)  # valid 0.5 file FIRST, before anything expensive
    print("[driver] wrote 0.5 benchmark; test rows =", len(test_df))
    _stage("mount_discovery", _t)
    weights = find_weights(WEIGHTS_DIR)  # prints its own [time] stage=weights_search line
    print("[driver] weights:", weights)
    if not weights:
        raise FileNotFoundError(f"no checkpoints under {WEIGHTS_DIR}")
    _t = time.time()
    run_submission(input_root=root, checkpoint_paths=weights, output_path=OUT,
                   batch=BATCH, workers=WORKERS, device=DEVICE, require_fingerprint=True)
    _stage("run_submission", _t)
except Exception:
    traceback.print_exc()
    if not os.path.exists(OUT):
        write_benchmark_submission(_fallback_test_df(), OUT)
        print("[driver] fallback 0.5 submission written")

# ---- final schema assertions ----
sub = pd.read_csv(OUT)
assert list(sub.columns) == [ID_COL] + TARGETS, ("bad columns", list(sub.columns))
vals = sub[TARGETS].values
assert (vals >= 0).all() and (vals <= 1).all(), "values out of [0,1]"
assert sub[ID_COL].is_unique, "duplicate StudyInstanceUID"
print("[driver] submission OK: rows=%d cols=%d" % (len(sub), sub.shape[1]))
print(sub.head())
_stage("total", _T_START)
'''


def make_notebook(bundle: str, driver: str) -> dict:
    def code(src):
        return {"cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": src.splitlines(keepends=True)}

    def md(src):
        return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

    cells = [
        md("# RSNA Knee Abnormality Detection - inference notebook\n"
           "\n"
           "Self-contained (no `src` import, no network). Package source is inlined below.\n"
           "Survival patterns: 0.5 file first, mount discovery, self-size to hidden test,\n"
           "pretrained=False weight load, fingerprint self-check, rank+reindex+fillna(0.5).\n"),
        code(bundle),
        code(driver.lstrip("\n")),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    bundle = build_bundle()
    nb = make_notebook(bundle, DRIVER)
    ipynb_path = os.path.join(HERE, "submission.ipynb")
    with open(ipynb_path, "w") as f:
        json.dump(nb, f, indent=1)
    # standalone .py mirror (same source) for local execution/validation
    py_path = os.path.join(HERE, "submission_standalone.py")
    with open(py_path, "w") as f:
        f.write(bundle + "\n\n" + DRIVER.lstrip("\n"))
    print("wrote", ipynb_path)
    print("wrote", py_path)


if __name__ == "__main__":
    main()
