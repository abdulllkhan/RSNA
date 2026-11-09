"""Unit tests (synthetic frames, CPU, seconds) for the sub #13 combine + guard logic:
  1. rank_average_frames_partial: reader covering every study -> IS rank_average_frames (bitwise);
     reader lacking rows -> those rows == rank_average_frames(other arms) bitwise, the others ==
     (3 r_hr + 10 r_v2 + 6 r_reader_valid) / 19 with the reader ranked within its valid rows (bitwise);
  2. the driver's own functions, exec'd from the DRIVER string of notebook/build_notebook_multi.py:
     parse_arm_weights (reader opt-in = 0), parse_tta_spec (reader pinned to S=1), combine_frames (no reader ->
     the old statements; reader complete -> identical; reader partial -> fallback; EMPTY intersection), and the
     final write_submission ranks;
  3. reader_gpjw.reader_guard thresholds (pre-registered: drop if invalid > 2 % or series_err > 2 %).
Usage: CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD .venv/bin/python -P dev/sub13/test_combine_guard.py
"""
import ast
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FAILS = []


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILS.append(name)


def frame(ids, rng, ties=False):
    v = rng.random((len(ids), len(TARGETS)))
    if ties:
        v = np.round(v * 5) / 5
    d = pd.DataFrame(v, columns=TARGETS)
    d.insert(0, ID_COL, ids)
    return d


def builder_driver_text(builder_path):
    tree = ast.parse(Path(builder_path).read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "DRIVER" for t in n.targets))


def driver_ns(drv):
    """The driver's own parse/combine functions + constants, exec'd from the DRIVER text (nothing else runs)."""
    dt = ast.parse(drv)
    want = {"parse_arm_weights", "parse_tta_spec", "combine_frames"}
    consts = {"ARM_ORDER", "OPT_IN_ARMS", "ROW_OPTIONAL_ARMS", "TTA_CAPABLE"}
    keep = [n for n in dt.body if (isinstance(n, ast.FunctionDef) and n.name in want)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in consts for t in n.targets))]
    ns = {"np": np, "re": re, "os": os, "ID_COL": ID_COL, "rank_average_frames": rank_average_frames,
          "rank_average_frames_partial": rank_average_frames_partial}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<driver-subset>", "exec"), ns)
    return ns


def run_all(ID_COL, TARGETS, rank_average_frames, rank_average_frames_partial, write_submission, R, drv, tmp):
    globals().update(ID_COL=ID_COL, TARGETS=TARGETS, rank_average_frames=rank_average_frames,
                     rank_average_frames_partial=rank_average_frames_partial)
    print(f"[env] pandas {pd.__version__} numpy {np.__version__}")
    rng = np.random.default_rng(0)
    ids = [f"1.2.3.{i}" for i in range(200)]
    hr, v2 = frame(ids, rng), frame(ids[::-1], rng, ties=True)          # v2 in another order, with ties
    rd_full = frame(ids, rng)
    W = [3.0, 10.0, 6.0]

    # 1a. reader covers every study -> rank_average_frames itself
    a = rank_average_frames_partial([hr, v2, rd_full], W, [False, False, True])
    b = rank_average_frames([hr, v2, rd_full], weights=W)
    check("partial(full reader) == rank_average_frames bitwise", a.equals(b))

    # 1b. reader lacks 7 studies
    lack = set(ids[5:12])
    rd = rd_full[~rd_full[ID_COL].isin(lack)].reset_index(drop=True)
    p = rank_average_frames_partial([hr, v2, rd], W, [False, False, True], log=None).set_index(ID_COL)
    two = rank_average_frames([hr, v2], weights=[3.0, 10.0]).set_index(ID_COL)
    check("rows lacking the reader == rank_average_frames(hr, v2) bitwise",
          np.array_equal(p.loc[sorted(lack)].values, two.loc[sorted(lack)].values))
    ok_ids = [s for s in ids if s not in lack]
    r_hr = hr.set_index(ID_COL).loc[ids][TARGETS].rank(pct=True)
    r_v2 = v2.set_index(ID_COL).loc[ids][TARGETS].rank(pct=True)
    r_rd = rd.set_index(ID_COL).loc[ok_ids][TARGETS].rank(pct=True)
    exp = (3.0 * r_hr.loc[ok_ids].values + 10.0 * r_v2.loc[ok_ids].values + 6.0 * r_rd.values) / 19.0
    check("rows with the reader == (3 r_hr + 10 r_v2 + 6 r_reader_valid)/19 bitwise",
          np.array_equal(p.loc[ok_ids].values, exp))
    check("partial result keeps every required study, values in (0, 1]",
          list(p.index) == ids and float(p.values.min()) > 0 and float(p.values.max()) <= 1)

    # 2. the driver's own functions
    ns = driver_ns(drv)
    w0 = ns["parse_arm_weights"]("base=0,hr=3,v2=10")
    check("parse_arm_weights('base=0,hr=3,v2=10'): reader stays 0 (opt-in), others as before",
          w0["reader"] == 0.0 and w0["hr"] == 3.0 and w0["v2"] == 10.0 and w0["base"] == 0.0
          and w0["plane"] == 1.0 and w0["25d"] == 1.0 and w0["ordered"] == 1.0)
    w6 = ns["parse_arm_weights"]("base=0,hr=3,v2=10,reader=6")
    check("parse_arm_weights('...,reader=6') -> reader 6.0", w6["reader"] == 6.0)
    s3 = ns["parse_tta_spec"]("3")
    check("parse_tta_spec('3'): base/hr/v2 S=3, reader pinned to 1", s3["reader"] == 1 and s3["v2"] == 3
          and s3["hr"] == 3 and s3["base"] == 3)
    ns["ARM_WEIGHTS"] = w6
    pr, ua = ns["combine_frames"]([hr, v2], ["hr", "v2"])
    check("combine_frames(no reader) == rank_average_frames([hr, v2], [3, 10]) (old path)",
          pr.equals(rank_average_frames([hr, v2], weights=[3.0, 10.0])) and ua == ["hr", "v2"])
    pr, ua = ns["combine_frames"]([hr, v2, rd_full], ["hr", "v2", "reader"])
    check("combine_frames(reader complete) == rank_average_frames([hr, v2, reader], [3, 10, 6])",
          pr.equals(rank_average_frames([hr, v2, rd_full], weights=W)))
    pr, ua = ns["combine_frames"]([hr, v2, rd], ["hr", "v2", "reader"])
    check("combine_frames(reader partial) == rank_average_frames_partial",
          pr.set_index(ID_COL).equals(p))
    pr, ua = ns["combine_frames"]([rd], ["reader"])
    check("combine_frames(reader alone) == rank_average_frames([reader])",
          pr.equals(rank_average_frames([rd], weights=[6.0])) and ua == ["reader"])
    hr_a, v2_b = frame(ids[:50], rng), frame(ids[50:], rng)
    pr, ua = ns["combine_frames"]([hr_a, v2_b, rd], ["hr", "v2", "reader"])
    check("combine_frames(EMPTY required intersection) keeps the widest required arm alone",
          ua == ["v2"] and pr.equals(rank_average_frames([v2_b], weights=[10.0])))

    # final submission ranks: write_submission re-ranks the combined score per column
    test_df = pd.DataFrame({ID_COL: ids[::-1]})
    out = Path(tmp) / "sub13_unit_sub.csv"
    sub = write_submission(p.reset_index(), test_df, str(out))
    exp_final = p.rank(pct=True).loc[test_df[ID_COL]].values
    check("write_submission(partial) == pct-rank of the partial score, in test.csv order",
          np.array_equal(sub[TARGETS].values, exp_final) and list(sub[ID_COL]) == ids[::-1])
    out.unlink()

    # 3. guard thresholds
    def stats(n, n_invalid, n_series, n_err):
        d = pd.DataFrame({ID_COL: [str(i) for i in range(n)], "slots_valid": 5, "series_selected": 0,
                          "series_err": 0, "valid": True})
        d.loc[: n_invalid - 1, "valid"] = False if n_invalid else True
        per = np.full(n, n_series // n)
        per[: n_series % n] += 1
        d["series_selected"] = per
        err = np.zeros(n, int)
        err[:n_err] = 1
        d["series_err"] = err
        return d

    g = R.reader_guard(stats(100, 2, 500, 0), 100)
    check("guard: 2/100 invalid (2.0%) -> ok", g["guard_ok"] and g["invalid"] == 2)
    g = R.reader_guard(stats(100, 3, 500, 0), 100)
    check("guard: 3/100 invalid (3.0%) -> TRIPPED", not g["guard_ok"])
    g = R.reader_guard(stats(100, 0, 500, 10), 100)
    check("guard: 10/500 series with decode errors (2.0%) -> ok", g["guard_ok"])
    g = R.reader_guard(stats(100, 0, 500, 11), 100)
    check("guard: 11/500 series with decode errors (2.2%) -> TRIPPED", not g["guard_ok"])
    g = R.reader_guard(stats(3, 1, 15, 0), 3)
    check("guard: visible test 1/3 invalid -> TRIPPED", not g["guard_ok"])
    g = R.reader_guard(stats(3, 0, 15, 0), 3)
    check("guard: visible test 0/3 invalid, 0 errors -> ok", g["guard_ok"])
    g = R.reader_guard(stats(90, 0, 450, 0), 100)
    check("guard: 10 studies never processed count as invalid (10%) -> TRIPPED", not g["guard_ok"]
          and g["invalid"] == 10)
    g = R.reader_guard(stats(5, 5, 0, 0), 5)
    check("guard: no selected series at all -> TRIPPED", not g["guard_ok"])
    check("guard limits are the pre-registered 2 % / 2 %",
          R.GUARD_MAX_INVALID_FRAC == 0.02 and R.GUARD_MAX_SERIES_ERR_FRAC == 0.02)

    # 4. discovery: both Kaggle mount layouts, the DICOM-tree pruning, and the bounded fallback
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for lay in ("classic", "datasets", "deep"):
            base = Path(td) / lay
            if lay == "classic":
                d = base / R.READER_DATASET
            elif lay == "datasets":
                d = base / "datasets" / R.READER_OWNER / R.READER_DATASET
            else:
                d = base / "datasets" / "someone-else" / "copy-of-weights" / "v1"
            d.mkdir(parents=True)
            for f in range(3):
                (d / f"cnxt_v0_fold{f}.pt").write_bytes(b"x")
            trap = base / "rsna-knee-abnormality-detection" / "train_series" / "s" / "r"
            trap.mkdir(parents=True)
            (trap / "cnxt_v0_fold9.pt").write_bytes(b"x")       # inside a DICOM tree: must never be found
            got = R.find_reader_checkpoints((str(base), str(base)))
            names = [os.path.basename(x) for x in got]
            check(f"discovery ({lay} layout): finds exactly folds 0-2, never inside train_series",
                  names == [f"cnxt_v0_fold{f}.pt" for f in range(3)])
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    return 1 if FAILS else 0


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from src.rsna import ID_COL, TARGETS
    from src.rsna import reader_gpjw as R
    from src.rsna.combine_partial import rank_average_frames_partial
    from src.rsna.infer import write_submission
    from src.rsna.infer_plane import rank_average_frames
    drv = builder_driver_text(root / "notebook/build_notebook_multi.py")
    return run_all(ID_COL, TARGETS, rank_average_frames, rank_average_frames_partial, write_submission, R, drv,
                   os.environ.get("TMPDIR", "/tmp"))


if __name__ == "__main__":
    sys.exit(main())
