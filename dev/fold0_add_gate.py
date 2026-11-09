"""Fold-0 ADD gate for a candidate v2 member X (pre-registered for the CoAtNet-2 arm, reports/coat2_raptor.md §6).

    PYTHONPATH=. .venv/bin/python dev/fold0_add_gate.py data/oof/<X>_fold0.parquet [--ws 2,3,5] [--min-doof 0.003]

Baseline = the shipped composition on fold 0: per-column percentile-rank average of
    hr=3   data/oof/effv2s_k24_384_tta3_fold0.parquet   (old-contract 384 arm, TTA3)
    lr=5   data/oof/v2_effv2s384_lr_fold0.parquet        (v2 effv2s, 140 mm)
    c110=5 data/oof/v2_effv2s384_c110_fold0.parquet      (v2 effv2s, 110 mm)
Candidate = baseline + X at weight w, for w in --ws. The weight is CHOSEN on the steven OOF only
(llm_labels_v4_blend > 0.5, all common fold-0 studies); the gold-in-fold delta (train.csv rows with
all 12 labels, n=14 in fold 0) is read ONLY at that chosen w. PASS iff
    best d(steven) >= --min-doof   AND   d(gold) >= 0 at that same w.
Prints a table and machine-readable lines: GATE_W=, GATE_DOOF=, GATE_DGOLD=, GATE=PASS|FAIL.
Self-check: --self-check runs the gate with X := the c110 member itself and asserts the w=0 path and
the rank arithmetic reproduce the baseline exactly.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
BASE = [("hr", "data/oof/effv2s_k24_384_tta3_fold0.parquet", 3.0),
        ("lr", "data/oof/v2_effv2s384_lr_fold0.parquet", 5.0),
        ("c110", "data/oof/v2_effv2s384_c110_fold0.parquet", 5.0)]
STEVEN = "data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
TRAIN_CSV = "data/comp_csv/train.csv"


def rd(p: str) -> pd.DataFrame:
    if not os.path.exists(p):
        raise SystemExit(f"ERROR: missing OOF parquet {p}")
    d = pd.read_parquet(p)
    d = d.set_index("StudyInstanceUID") if "StudyInstanceUID" in d.columns else d
    d.index = d.index.astype(str)
    miss = [c for c in T if c not in d.columns]
    if miss:
        raise SystemExit(f"ERROR: {p} lacks columns {miss}")
    return d[T].astype(np.float64)


def macro(P: pd.DataFrame, Y: pd.DataFrame) -> float:
    a = [roc_auc_score(Y[c], P[c]) for c in T if Y[c].nunique() > 1]
    return float(np.mean(a)) if a else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cand", nargs="?", help="fold-0 OOF parquet of the candidate member")
    ap.add_argument("--ws", default="2,3,5")
    ap.add_argument("--min-doof", type=float, default=0.003)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    cand_path = BASE[2][1] if a.self_check else a.cand
    if not cand_path:
        raise SystemExit("usage: fold0_add_gate.py <cand_fold0.parquet> | --self-check")
    frames = {k: rd(p) for k, p, _w in BASE}
    X = rd(cand_path)
    idx = X.index
    for f in frames.values():
        idx = idx.intersection(f.index)
    if len(idx) < 100:
        raise SystemExit(f"ERROR: only {len(idx)} common fold-0 studies")
    sv = pd.read_csv(STEVEN).set_index("StudyInstanceUID"); sv.index = sv.index.astype(str)
    ys = (sv.reindex(idx)[T] > 0.5).astype(int)
    tr = pd.read_csv(TRAIN_CSV).set_index("StudyInstanceUID"); tr.index = tr.index.astype(str)
    gold = tr.dropna(subset=T)[T].astype(int)
    g = idx.intersection(gold.index); yg = gold.loc[g]
    R = {k: v.loc[idx].rank(pct=True) for k, v in frames.items()}
    RX = X.loc[idx].rank(pct=True)

    def ens(wx: float):
        num = sum(w * R[k] for k, _p, w in BASE) + (wx * RX if wx else 0.0)
        return num / (sum(w for _k, _p, w in BASE) + wx)

    base = ens(0.0)
    b_s, b_g = macro(base, ys), macro(base.loc[g], yg)
    print(f"fold-0 ADD gate: candidate {cand_path}")
    print(f"  common studies n={len(idx)}  gold-in-fold n={len(g)}  steven = llm_labels_v4_blend > 0.5")
    print(f"  {'member / ensemble':<40}{'steven':>9}{'gold':>9}{'d(steven)':>11}{'d(gold)':>10}")
    for k, _p, w in BASE:
        print(f"  {k + ' solo':<40}{macro(R[k], ys):>9.4f}{macro(R[k].loc[g], yg):>9.4f}")
    print(f"  {'X solo':<40}{macro(RX, ys):>9.4f}{macro(RX.loc[g], yg):>9.4f}")
    print(f"  {'baseline hr=3 lr=5 c110=5':<40}{b_s:>9.4f}{b_g:>9.4f}{'(ref)':>11}")
    rows = []
    for wx in [float(s) for s in a.ws.split(",") if s.strip()]:
        E = ens(wx)
        s_, g_ = macro(E, ys), macro(E.loc[g], yg)
        rows.append((wx, s_ - b_s, g_ - b_g))
        print(f"  {f'+ X at w={wx:g}':<40}{s_:>9.4f}{g_:>9.4f}{s_ - b_s:>+11.4f}{g_ - b_g:>+10.4f}")
    if a.self_check:
        # X == c110: adding it at w=3 must equal the baseline with c110 at 5 + 3 (same ensemble
        # values up to float summation order; AUC may move ~1e-7 where those 1e-16 differences break ties)
        E2 = (3 * R["hr"] + 5 * R["lr"] + (5 + 3) * R["c110"]) / 16.0
        dv = float((E2 - ens(3.0)).abs().max().max())
        da = abs(macro(E2, ys) - macro(ens(3.0), ys))
        ok = bool(RX.equals(R["c110"])) and dv < 1e-12 and da < 1e-5
        print(f"  SELF-CHECK (X := c110): rank frames equal={RX.equals(R['c110'])} "
              f"max|dE|={dv:.1e} |dAUC|={da:.1e} -> {'OK' if ok else 'MISMATCH'}")
        return 0 if ok else 1
    # weight chosen on the big anchor only; ties -> the smaller weight
    wbest, dbest, gbest = max(rows, key=lambda r: (r[1], -r[0]))
    ok = dbest >= a.min_doof and gbest >= 0.0
    print(f"GATE_W={wbest:g}\nGATE_DOOF={dbest:+.4f}\nGATE_DGOLD={gbest:+.4f}")
    print(f"GATE={'PASS' if ok else 'FAIL'}  (bar: d(steven) >= {a.min_doof:+.3f} at the steven-chosen "
          f"w AND d(gold) >= 0 there)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
