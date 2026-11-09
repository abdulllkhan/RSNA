"""Fold-0 ADD check for a candidate member X against the sub-#13 composition (sub #10b + public reader at w=6).

    PYTHONPATH=. .venv/bin/python dev/fold0_add_gate_reader.py data/oof/<X>_fold0.parquet [--ws 2,3,5]

Companion to dev/fold0_add_gate.py (whose baseline is sub #10b = hr=3 + lr=5 + c110=5). Here the baseline also
carries the goodpjw2008 reader at rank weight 6 (EXPERIMENTS A18), so the study set is fold 0 INTERSECT the reader's
honest OOF (reader folds 0-2 held-out + gold; ~61 % of fold 0). Per-column percentile ranks are computed within
that set. Weight chosen on steven OOF only; gold-in-fold read at the chosen w. Same bars as the original gate
(d(steven) >= +0.003, d(gold) >= 0). Informational next to the pre-registered gate; decisive once sub #13 confirms
the reader on the board.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
BASE = [("hr", "data/oof/effv2s_k24_384_tta3_fold0.parquet", 3.0),
        ("lr", "data/oof/v2_effv2s384_lr_fold0.parquet", 5.0),
        ("c110", "data/oof/v2_effv2s384_c110_fold0.parquet", 5.0),
        ("reader", "data/oof/pubreader_gpjw_oof.parquet", 6.0)]
STEVEN = "data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
TRAIN_CSV = "data/comp_csv/train.csv"


def rd(p: str) -> pd.DataFrame:
    d = pd.read_parquet(os.path.join(ROOT, p))
    d = d.set_index("StudyInstanceUID") if "StudyInstanceUID" in d.columns else d
    d.index = d.index.astype(str)
    return d[T].astype(np.float64)


def macro(P: pd.DataFrame, Y: pd.DataFrame) -> float:
    return float(np.mean([roc_auc_score(Y[c], P[c]) for c in T if Y[c].nunique() > 1]))


def blend(frames: dict, weights: dict, idx) -> pd.DataFrame:
    E = sum(weights[n] * frames[n].loc[idx].rank(pct=True) for n in weights)
    return E / sum(weights.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("--ws", default="2,3,5")
    a = ap.parse_args()
    frames = {n: rd(p) for n, p, _ in BASE}
    frames["X"] = rd(a.cand)
    idx = frames["hr"].index
    for d in frames.values():
        idx = idx.intersection(d.index)
    ys = (pd.read_csv(os.path.join(ROOT, STEVEN)).set_index("StudyInstanceUID")[T] > 0.5).astype(int)
    tr = pd.read_csv(os.path.join(ROOT, TRAIN_CSV)).set_index("StudyInstanceUID")
    gold = tr.dropna(subset=T)[T].astype(int)
    gi = idx.intersection(gold.index)
    print(f"[set] fold 0 x reader OOF x candidate: {len(idx)} studies ({len(gi)} gold)")
    w10b = {n: w for n, _, w in BASE if n != "reader"}
    w13 = {n: w for n, _, w in BASE}
    rows = []
    for tag, base in (("sub10b", w10b), ("sub13", w13)):
        E0 = blend(frames, base, idx)
        s0, g0 = macro(E0, ys.loc[idx]), macro(E0.loc[gi], gold.loc[gi])
        rows.append({"baseline": tag, "w": 0, "steven": s0, "gold": g0, "d_steven": 0.0, "d_gold": 0.0})
        for w in [float(x) for x in a.ws.split(",")]:
            E = blend(frames, {**base, "X": w}, idx)
            s, g = macro(E, ys.loc[idx]), macro(E.loc[gi], gold.loc[gi])
            rows.append({"baseline": tag, "w": w, "steven": s, "gold": g, "d_steven": s - s0, "d_gold": g - g0})
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))
    for tag in ("sub10b", "sub13"):
        c = t[(t.baseline == tag) & (t.w > 0)]
        b = c.loc[c.steven.idxmax()]
        ok = b.d_steven >= 0.003 and b.d_gold >= 0
        print(f"GATE_{tag.upper()}: w={b.w:g} d_steven={b.d_steven:+.4f} d_gold={b.d_gold:+.4f} -> "
              f"{'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
