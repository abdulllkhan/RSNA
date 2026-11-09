"""Re-prove, for src/rsna/reader_gpjw.py (the module the sub #13 bundle embeds), what dev/pub_reader/verify_port.py
proved for dev/pub_reader/gpjw_port.py:

  static    every ported constant/function/class is AST-identical to the downloaded original (parse only:
            nothing downloaded is executed), and the whole port block equals gpjw_port.py's byte-for-byte;
  windows   the instrumented ReaderStudyDataset (counters in the vol_of wrapper) gives byte-identical windows /
            mask / positions to the verbatim DicomStudyDataset on N real training studies;
  counters  on a scratch copy of one study with one corrupted slice and one fully corrupted series, the counters
            report exactly those, and the windows are STILL byte-identical to DicomStudyDataset's;
  load      all three checkpoints pass the name/args/sha256 checks and load strictly (weights_only=True).

Usage (repo root, CPU): CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD .venv/bin/python -P dev/sub13/verify_reader_module.py \
          --studies 4 --scratch <dir>
"""
import argparse
import ast
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "models/public/gpjw_reader"
MOD = ROOT / "src/rsna/reader_gpjw.py"
PORT = ROOT / "dev/pub_reader/gpjw_port.py"
WANT = {
    "preprocess.py": ["MM_PER_PX", "SIZE", "MAX_SLICES", "CANON", "PLANES", "_read", "_canon_slice",
                      "_resample", "load_series"],
    "knee.py": ["LABELS", "N_SLOTS", "MEAN", "STD", "slot_of", "build_study_table", "window_centres",
                "make_views", "KneeNet"],
    "infer.py": ["DicomStudyDataset"],
}
BEGIN = "# ================================================================ verbatim port (dev/pub_reader/gpjw_port.py) ===="
END = "# ================================================================ end of the verbatim port"


def top_nodes(tree):
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out.setdefault(node.targets[0].id, node)
    return out


def static_check():
    text = MOD.read_text()
    block = text.split(BEGIN, 1)[1].split(END, 1)[0].strip("\n")
    port_block = "\n".join(PORT.read_text().split("\n")[17:345]).strip("\n")   # gpjw_port.py lines 18-345
    ok = block == port_block
    print(f"[static] port block of reader_gpjw.py == gpjw_port.py lines 18-345 byte-for-byte: {ok}")
    mod = top_nodes(ast.parse(text))
    port = top_nodes(ast.parse(PORT.read_text()))
    for fname, names in WANT.items():
        orig = top_nodes(ast.parse((SRC / fname).read_text()))
        for nm in names:
            same = ast.dump(orig[nm]) == ast.dump(mod[nm])
            ok &= same
            print(f"[static] {fname}:{nm:<20} {'IDENTICAL' if same else 'DIFFERENT'}")
    for nm in ("build_study_table_fallback", "load_models", "series_file_counts", "study_windows"):
        same = ast.dump(port[nm]) == ast.dump(mod[nm])
        ok &= same
        print(f"[static] gpjw_port.py:{nm:<26} {'IDENTICAL' if same else 'DIFFERENT'}")
    return ok


def window_check(n_studies, scratch):
    import numpy as np
    import pandas as pd
    import torch
    from src.rsna import reader_gpjw as R

    root = str(ROOT / "data/raw/train_series")
    se_all = pd.read_csv(ROOT / "data/comp_csv/train_series.csv")
    studies = se_all.StudyInstanceUID.drop_duplicates().sample(n_studies, random_state=11).tolist()
    se = se_all[se_all.StudyInstanceUID.isin(set(studies))]
    nf = R.series_file_counts(se, root)
    table = R.build_study_table(se, nf)
    ok = True
    ref = R.DicomStudyDataset(studies, table, root, R.READER_K)
    new = R.ReaderStudyDataset(studies, table, root, R.READER_K)
    for i, st in enumerate(studies):
        x0, m0, p0 = ref[i]
        st1, x1, m1, p1, info = new[i]
        same = st1 == st and torch.equal(x0, x1) and torch.equal(m0, m1) and torch.equal(p0, p1)
        ok &= same
        errs = sum(1 for r in info["series"] if r["error"] or r["n_bad"])
        print(f"[windows] {st[-12:]} slots={m0.int().tolist()} series={len(info['series'])} err={errs} "
              f"identical={same}")

    # counters: one study copied to scratch, one slice corrupted in slot A, every slice corrupted in slot B
    st = studies[0]
    slots = table[st]
    picked = [c[0] for c in slots if c][:2]
    dst = Path(scratch) / "corrupt_mount"
    shutil.rmtree(dst, ignore_errors=True)
    sdir = dst / st
    for sr in sorted({s for c in slots for s in c}):
        (sdir / sr).mkdir(parents=True)
        for fn in sorted(os.listdir(f"{root}/{st}/{sr}")):
            os.symlink(f"{root}/{st}/{sr}/{fn}", sdir / sr / fn)
    a, b = picked
    fa = sorted(os.listdir(sdir / a))
    os.unlink(sdir / a / fa[len(fa) // 2])
    (sdir / a / fa[len(fa) // 2]).write_bytes(b"not a dicom")
    for fn in os.listdir(sdir / b):
        os.unlink(sdir / b / fn)
        (sdir / b / fn).write_bytes(b"garbage")
    ref = R.DicomStudyDataset([st], table, str(dst), R.READER_K)
    new = R.ReaderStudyDataset([st], table, str(dst), R.READER_K)
    x0, m0, p0 = ref[0]
    _, x1, m1, p1, info = new[0]
    same = torch.equal(x0, x1) and torch.equal(m0, m1) and torch.equal(p0, p1)
    rec = {r["series"]: r for r in info["series"]}
    ca = rec[a]["n_bad"] == 1 and not rec[a]["error"]
    cb = rec[b]["n_bad"] == len(os.listdir(sdir / b)) and rec[b]["n"] == 1
    print(f"[counters] slot A ({a[-8:]}): n_bad={rec[a]['n_bad']} n={rec[a]['n']} (want 1 bad) -> {ca}")
    print(f"[counters] slot B ({b[-8:]}): n_bad={rec[b]['n_bad']} n={rec[b]['n']} (want all bad, slot masked) -> {cb}")
    print(f"[counters] mask {m1.int().tolist()}; windows still identical to DicomStudyDataset: {same}")
    ok &= same and ca and cb and not bool(m1[[i for i, c in enumerate(slots) if c and c[0] == b][0]])
    return ok


def load_check():
    import torch
    from src.rsna import reader_gpjw as R
    paths = [str(SRC / f"cnxt_v0_fold{f}.pt") for f in (0, 1, 2)]
    mem = R.load_reader_members(paths, device="cpu")
    ok = [f for f, _, _ in mem] == [0, 1, 2]
    try:
        R.load_reader_members(paths[:2], device="cpu")
        ok = False
        print("[load] 2 of 3 checkpoints accepted -> WRONG")
    except ValueError as e:
        print(f"[load] 2 of 3 checkpoints refused as expected: {e}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", type=int, default=4)
    ap.add_argument("--scratch", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    ok = static_check()
    ok &= window_check(a.studies, a.scratch)
    ok &= load_check()
    print("READER MODULE VERIFIED" if ok else "READER MODULE MISMATCH")
    sys.exit(0 if ok else 1)
