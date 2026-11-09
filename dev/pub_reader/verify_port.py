"""Prove dev/pub_reader/gpjw_port.py is a verbatim copy of the downloaded reader code.

Static part (no downloaded code is executed: the originals are only parsed with `ast`):
  every ported constant / function / class must have an AST identical to the original's.
Behavioural part (--studies N): the memoised refactor `study_windows` must give byte-identical
  windows / mask / positions to the verbatim `DicomStudyDataset` on N real training studies, and all
  three checkpoints must load strictly into the ported KneeNet.

Usage (repo root):
  PYTHONPATH=$PWD .venv/bin/python -P dev/pub_reader/verify_port.py [--studies 4]
"""
import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "models/public/gpjw_reader"
NB954 = ROOT / ("outputs/survey_oct/k/goodpjw2008__rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954/"
                "rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954.ipynb")
PORT = ROOT / "dev/pub_reader/gpjw_port.py"

WANT = {
    "preprocess.py": ["MM_PER_PX", "SIZE", "MAX_SLICES", "CANON", "PLANES", "_read", "_canon_slice",
                      "_resample", "load_series"],
    "knee.py": ["LABELS", "N_SLOTS", "MEAN", "STD", "slot_of", "build_study_table", "window_centres",
                "make_views", "KneeNet"],
    "infer.py": ["DicomStudyDataset"],
}


def top_nodes(tree):
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out.setdefault(node.targets[0].id, node)  # first assignment wins (PLANES is defined twice upstream)
    return out


def strip_docstrings(node):
    node = ast.parse(ast.unparse(node)).body[0]
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr) \
                and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
    return ast.dump(node)


def static_check():
    port = top_nodes(ast.parse(PORT.read_text()))
    ok = True
    for fname, names in WANT.items():
        orig = top_nodes(ast.parse((SRC / fname).read_text()))
        for nm in names:
            same = ast.dump(orig[nm]) == ast.dump(port[nm])
            ok &= same
            print(f"[static] {fname}:{nm:<20} {'IDENTICAL' if same else 'DIFFERENT'}")
    # the 0.954 notebook's knee.py (cell 13) build_study_table == our build_study_table_fallback (renamed)
    cells = json.loads(NB954.read_text())["cells"]
    src = "".join(cells[13]["source"])
    assert src.startswith("%%writefile /kaggle/working/own_src/knee.py")
    nb = top_nodes(ast.parse(src.split("\n", 1)[1]))
    a = nb["build_study_table"]
    b = ast.parse(ast.unparse(port["build_study_table_fallback"])).body[0]
    b.name = "build_study_table"
    same = strip_docstrings(a) == strip_docstrings(b)
    ok &= same
    print(f"[static] nb0954 cell13:build_study_table == build_study_table_fallback (docstring aside): "
          f"{'IDENTICAL' if same else 'DIFFERENT'}")
    # the dataset's PLANES in knee.py equals preprocess.py's (we keep one copy)
    k = top_nodes(ast.parse((SRC / "knee.py").read_text()))["PLANES"]
    same = ast.dump(k) == ast.dump(port["PLANES"])
    ok &= same
    print(f"[static] knee.py:PLANES == preprocess.py:PLANES: {same}")
    return ok


def behaviour_check(n_studies):
    import numpy as np
    import pandas as pd
    import torch
    from dev.pub_reader import gpjw_port as G

    root = str(ROOT / "data/raw/train_series")
    se = pd.read_csv(ROOT / "data/comp_csv/train_series.csv")
    studies = se.StudyInstanceUID.drop_duplicates().sample(n_studies, random_state=1).tolist()
    se = se[se.StudyInstanceUID.isin(set(studies))]
    nf = G.series_file_counts(se, root)
    table = G.build_study_table(se, nf)
    ok = True
    for k in (12, 16):
        ds = G.DicomStudyDataset(studies, table, root, k)
        for i, st in enumerate(studies):
            x0, m0, p0 = ds[i]
            memo = {}

            def vol_of(sr, st=st, memo=memo):
                if sr not in memo:
                    memo[sr] = G.load_series(f"{root}/{st}/{sr}")[0]
                return memo[sr]

            x1, m1, p1 = G.study_windows(vol_of, st, table.get(st, [[] for _ in range(G.N_SLOTS)]), k)
            same = torch.equal(x0, torch.from_numpy(x1)) and torch.equal(m0, torch.from_numpy(m1)) \
                and torch.equal(p0, torch.from_numpy(p1))
            ok &= same
            print(f"[behaviour] k={k} {st[-12:]} slots={m0.int().tolist()} identical={same}")
    for f in range(3):
        G.load_models([str(SRC / f"cnxt_v0_fold{f}.pt")], device="cpu")  # strict load_state_dict
        print(f"[behaviour] cnxt_v0_fold{f}.pt loads strictly into the ported KneeNet")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", type=int, default=0)
    a = ap.parse_args()
    ok = static_check()
    if a.studies:
        ok &= behaviour_check(a.studies)
    print("PORT VERIFIED" if ok else "PORT MISMATCH")
    sys.exit(0 if ok else 1)
