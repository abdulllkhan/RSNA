"""Does a NEW fold-0 member beat the 110 mm member it would replace, inside the shipped ensemble?

    .venv/bin/python dev/fold0_member_swap.py data/oof/<new_name>_fold0.parquet

A solo OOF macro-AUC is not the decision: the arm is shipped as one term of a rank-average, and a
member that is a little worse alone but less correlated with the others can still be the better
pick (and vice versa). So this scores the ENSEMBLE on the 882 fold-0 studies, rank-averaging

    hr(3) = effv2s_k24_384_tta3      c140(5) = v2_effv2s384_lr      X(5) = the swapped member

for X = the incumbent 110 mm arm vs X = the new member, and then the 4-arm variant that KEEPS the
incumbent and adds the new one (hr=3, c140=5, c110=5, new=5). Yardstick = the binarised steven
report labels (> 0.5); the 14 gold studies that fall in fold 0 are printed beside it as a second,
independent (but tiny, so noisy) read.

Sanity check built in: pass the incumbent's own parquet as the new member and the two 3-arm rows
must print identical numbers.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rsna import TARGETS, ID_COL  # noqa: E402

HR = "data/oof/effv2s_k24_384_tta3_fold0.parquet"
C140 = "data/oof/v2_effv2s384_lr_fold0.parquet"
C110 = "data/oof/v2_effv2s384_c110_fold0.parquet"
STEVEN = "data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
TRAIN_CSV = "data/comp_csv/train.csv"


def load_oof(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: missing OOF parquet {path}")
    d = pd.read_parquet(path)
    if ID_COL in d.columns:
        d = d.set_index(ID_COL)
    d.index = d.index.astype(str)
    missing = [c for c in TARGETS if c not in d.columns]
    if missing:
        raise SystemExit(f"ERROR: {path}: missing target columns {missing}")
    return d[TARGETS].astype(np.float64)


def macro(P: pd.DataFrame, Y: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score
    a = [roc_auc_score(Y[c], P[c]) for c in TARGETS if Y[c].nunique() > 1]
    return float(np.mean(a)) if a else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("new_oof", help="fold-0 OOF parquet of the candidate member")
    ap.add_argument("--hr", default=HR)
    ap.add_argument("--c140", default=C140)
    ap.add_argument("--c110", default=C110)
    ap.add_argument("--w-hr", type=float, default=3.0)
    ap.add_argument("--w-c140", type=float, default=5.0)
    ap.add_argument("--w-x", type=float, default=5.0)
    args = ap.parse_args()
    os.chdir(ROOT)

    hr, c140, c110 = load_oof(args.hr), load_oof(args.c140), load_oof(args.c110)
    new = load_oof(args.new_oof)
    idx = hr.index.intersection(c140.index).intersection(c110.index).intersection(new.index)
    if len(idx) == 0:
        raise SystemExit("ERROR: the four OOF tables share no studies")

    sv = pd.read_csv(STEVEN).set_index(ID_COL)
    sv.index = sv.index.astype(str)
    ys = (sv.reindex(idx)[TARGETS] > 0.5).astype(int)
    tr = pd.read_csv(TRAIN_CSV).set_index(ID_COL)
    tr.index = tr.index.astype(str)
    gold = tr.dropna(subset=TARGETS)[TARGETS].astype(int)
    g = idx.intersection(gold.index)
    yg = gold.loc[g]

    r = {k: v.loc[idx].rank(pct=True) for k, v in
         {"hr": hr, "c140": c140, "c110": c110, "new": new}.items()}

    def ens(parts):                                  # parts: list of (rank frame, weight)
        tot = sum(w for _, w in parts)
        E = sum(f * w for f, w in parts) / tot
        return macro(E, ys), (macro(E.loc[g], yg) if len(g) else float("nan"))

    print(f"fold-0 member swap: rank-average, steven llm_labels_v4_blend > 0.5")
    print(f"  common studies n={len(idx)}   gold-in-fold n={len(g)}")
    print(f"  candidate: {args.new_oof}")
    print(f"  {'member (solo)':<46}{'steven':>9}{'gold':>9}")
    for k, nm, p in [("hr", args.hr, hr), ("c140", args.c140, c140),
                     ("c110", args.c110, c110), ("new", args.new_oof, new)]:
        lab = f"{k:<5}{os.path.basename(nm).replace('.parquet', '')}"
        print(f"  {lab:<46}"
              f"{macro(p.loc[idx], ys):>9.4f}{(macro(p.loc[g], yg) if len(g) else float('nan')):>9.4f}")

    wh, wc, wx = args.w_hr, args.w_c140, args.w_x
    s_old, g_old = ens([(r["hr"], wh), (r["c140"], wc), (r["c110"], wx)])
    s_new, g_new = ens([(r["hr"], wh), (r["c140"], wc), (r["new"], wx)])
    s_4, g_4 = ens([(r["hr"], wh), (r["c140"], wc), (r["c110"], wx), (r["new"], wx)])
    print(f"\n  {'ensemble':<46}{'steven':>9}{'gold':>9}{'d(steven)':>11}")
    print(f"  {f'3-arm hr={wh:g} c140={wc:g} X=c110':<46}{s_old:>9.4f}{g_old:>9.4f}{'(ref)':>11}")
    print(f"  {f'3-arm hr={wh:g} c140={wc:g} X=new':<46}{s_new:>9.4f}{g_new:>9.4f}{s_new - s_old:>+11.4f}")
    print(f"  {f'4-arm hr={wh:g} c140={wc:g} c110={wx:g} new={wx:g}':<46}"
          f"{s_4:>9.4f}{g_4:>9.4f}{s_4 - s_old:>+11.4f}")
    same = os.path.realpath(args.new_oof) == os.path.realpath(args.c110)
    if same:
        ok = abs(s_new - s_old) < 1e-12 and (not np.isfinite(g_new) or abs(g_new - g_old) < 1e-12)
        print(f"\n  SELF-SWAP CHECK (new == c110): 3-arm rows identical -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(1)
    d = s_new - s_old
    verdict = "WINS" if d > 1e-4 else ("ties" if abs(d) <= 1e-4 else "loses")
    print(f"\n  VERDICT: swapping c110 -> new {verdict} ({d:+.4f} steven); keeping BOTH "
          f"{'WINS' if s_4 > max(s_old, s_new) + 1e-4 else 'does not beat the better 3-arm'} "
          f"({s_4 - max(s_old, s_new):+.4f})")


if __name__ == "__main__":
    main()
