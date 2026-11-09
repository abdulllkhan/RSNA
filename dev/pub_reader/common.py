"""Shared loaders / metrics for the public-reader validation (dev/pub_reader/*)."""
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
STEVEN = ROOT / "data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
TRAIN = ROOT / "data/comp_csv/train.csv"
FOLDS_CSV = ROOT / "outputs/survey_oct/gpjw_reader_folds_rederived.csv"
RUN = Path(os.environ.get("PUBREADER_RUN", ROOT / "outputs/pub_reader/run_all"))  # override only for tests


def steven_bin():
    sv = pd.read_csv(STEVEN).set_index("StudyInstanceUID")[T]
    return (sv > 0.5).astype(int)


def steven_soft():
    return pd.read_csv(STEVEN).set_index("StudyInstanceUID")[T].astype(float)


def gold58():
    tr = pd.read_csv(TRAIN).set_index("StudyInstanceUID")
    return tr.dropna(subset=T)[T].astype(int)


def macro(P, Y, cols=T):
    """mean of per-column AUCs over columns where Y has both classes (the competition metric)."""
    aucs = [roc_auc_score(Y[c], P[c]) for c in cols if Y[c].nunique() > 1]
    return float(np.mean(aucs))


def per_col(P, Y, cols=T):
    return pd.Series({c: roc_auc_score(Y[c], P[c]) if Y[c].nunique() > 1 else np.nan for c in cols})


def reader_folds(kind="stable"):
    """goodpjw2008 make_labels.py L96-105 re-implemented: GroupKFold(5) over normalised report text of the 4,349
    non-gold studies, gold fold = -1. kind='stable' = sklearn >= 1.6 behaviour (argsort kind='stable', platform
    independent; equals outputs/survey_oct/gpjw_reader_folds_rederived.csv); kind='quicksort' = the older
    unstable argsort on THIS machine (used only as a negative control)."""
    tr = pd.read_csv(TRAIN)
    is_gold = tr[T].notna().all(axis=1).values
    weak = ~is_gold
    groups = tr.Report.map(lambda t: re.sub(r"\W+", "", str(t).lower())).values[weak]
    if kind == "stable":
        order = np.random.RandomState(0).permutation(weak.sum())
        folds = np.zeros(weak.sum(), int)
        for k, (_, va) in enumerate(GroupKFold(5).split(order, groups=groups[order])):
            folds[order[va]] = k
    else:
        uniq, inv = np.unique(groups, return_inverse=True)
        cnt = np.bincount(inv)
        load, g2f = np.zeros(5), np.zeros(len(uniq), int)
        for gi in np.argsort(cnt, kind=kind)[::-1]:
            f = int(np.argmin(load))
            load[f] += cnt[gi]
            g2f[gi] = f
        folds = g2f[inv]
    out = pd.Series(-1, index=tr.StudyInstanceUID, name="fold")
    out[weak] = folds
    return out


def reader_pred(variant="k12", fold=0, run=RUN):
    d = pd.read_parquet(Path(run) / f"pred_{variant}_fold{fold}.parquet")
    assert list(d.columns) == T and d.index.is_unique
    return d


def stitch(name):
    """our stitched 5-fold OOF of one member (as dev/fit_v2_stitched2.py)."""
    fs = sorted(glob.glob(str(ROOT / f"data/oof/{name}_fold[0-9].parquet")))
    assert len(fs) == 5, f"{name}: need 5 folds, found {len(fs)}"
    d = pd.concat([pd.read_parquet(f) for f in fs])
    if "StudyInstanceUID" in d.columns:
        d = d.set_index("StudyInstanceUID")
    d = d[~d.index.duplicated()]
    return d[T]


def sub10b_members():
    """sub #10b = rank-avg(hr_tta3 x 3, v2 x 10), v2 = equal rank-average of lr and c110 (dev/fit_v2_stitched2.py)."""
    return {"hr": stitch("effv2s_k24_384_tta3"), "lr": stitch("v2_effv2s384_lr"), "c110": stitch("v2_effv2s384_c110")}


def sub10b_blend(members, idx, reader=None, w=0.0):
    """percentile ranks computed WITHIN idx (as if idx were the test set), then
    (3*rank(hr) + 10*v2 [+ w*rank(reader)]) / (13 + w), v2 = (rank(lr) + rank(c110)) / 2."""
    R = {n: d.loc[idx].rank(pct=True) for n, d in members.items()}
    rv = (R["lr"] + R["c110"]) / 2
    E = 3 * R["hr"] + 10 * rv
    if reader is not None and w:
        E = E + w * reader.loc[idx].rank(pct=True)
    return E / (13 + w)
