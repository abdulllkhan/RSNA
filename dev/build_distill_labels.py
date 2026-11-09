"""Knowledge-distillation labels: blend the consensus targets with the ensemble's OOF teacher.

    .venv/bin/python dev/build_distill_labels.py [--alpha 0.5] [--reweight] [--out ...]

WHAT THIS BUILDS
----------------
`data/labels/consensus_v2_distill.parquet` — the same table as `data/labels/consensus_v1.parquet`
(index `StudyInstanceUID`, 12 target columns, 12 `<T>__w` weight columns, `is_gold`, `fold`, same
column order and dtypes) with ONE change: for the NON-gold studies the soft target becomes

    T_new = (1 - ALPHA) * T_consensus + ALPHA * T_teacher            (ALPHA = 0.5 by default)

where `T_teacher` is the per-study mean of the out-of-fold sigmoid probabilities of the two best
v2 arms (`v2_effv2s384_lr` = 140 mm crop, `v2_effv2s384_c110` = 110 mm crop), stitched over their
5 fold files so every study is predicted by a model that did NOT train on it.

The point is label-noise reduction: the consensus targets are LLM-report labels (three teachers,
per-cell confidence weights), and the model's OOF probability is an independent, image-grounded
estimate of the same quantity. Averaging the two shrinks the idiosyncratic noise of each.

GOLD ROWS ARE NEVER TOUCHED
---------------------------
The 58 radiologist-labelled studies (`is_gold == True`) keep their exact consensus_v1 targets
(true 0/1) and weights (3.0). They are the only ground truth in the table; blending a model's
opinion into them would be strictly destructive.

Non-gold weights `<T>__w` are copied unchanged by default. `--reweight` implements the optional

    w_new = w * (0.5 + 0.5 * (1 - |T_consensus - T_teacher|))

i.e. down-weight (to at most 0.5x) the cells where the report label and the model disagree most.
It is OFF by default: it is a second, entangled change, and the ALPHA blend should be measured
on its own first.

LEAKAGE CAVEAT  (read this before trusting the student's OOF)
-------------------------------------------------------------
The teacher value on a study in fold j comes from the arm's fold-j model, which was trained on
folds != j — INCLUDING fold k. So when the student is trained without fold k, its *training*
targets on folds != k carry, indirectly, information from the fold-k labels that the teacher
models saw. This is second-order leakage and it is the standard OOF-stacking caveat: the
student's own fold-k OOF macro-AUC is therefore slightly OPTIMISTIC and is NOT comparable on
equal footing with the 0.8828 baseline it is gated against. Treat a small gain as noise; the
leaderboard (and a same-fold ensemble swap) is the arbiter. A leak-free variant would need a
teacher built from nested CV (a 4-fold teacher inside each 5-fold training split), which costs
20 trainings instead of 5 and is not affordable here.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rsna import TARGETS, ID_COL  # noqa: E402

W_COLS = [f"{t}__w" for t in TARGETS]
DEFAULT_ARMS = ["v2_effv2s384_lr", "v2_effv2s384_c110"]
STEVEN_CSV = "data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
TRAIN_CSV = "data/comp_csv/train.csv"
NOGOLD = "data/labels/consensus_v1_nogold.parquet"


def stitch_arm(name: str, oof_dir: str) -> pd.DataFrame:
    """All 5 fold files of one arm, concatenated: one row per study, predicted out of fold."""
    files = sorted(glob.glob(os.path.join(oof_dir, f"{name}_fold[0-9].parquet")))
    if not files:
        raise SystemExit(f"ERROR: no OOF parquets for arm {name!r} under {oof_dir}")
    parts = []
    for f in files:
        d = pd.read_parquet(f)
        if ID_COL in d.columns:
            d = d.set_index(ID_COL)
        d.index = d.index.astype(str)
        parts.append(d[TARGETS].astype(np.float64))
    d = pd.concat(parts)
    dup = int(d.index.duplicated().sum())
    if dup:
        raise SystemExit(f"ERROR: arm {name}: {dup} study appears in more than one fold file "
                         f"(an OOF table must have exactly one prediction per study)")
    print(f"[teacher] {name}: {len(files)} fold files -> {len(d)} studies "
          f"(range {d.values.min():.4f}..{d.values.max():.4f})")
    return d


def macro_auc(P: pd.DataFrame, Y: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score(Y[c], P[c]) for c in TARGETS if Y[c].nunique() > 1]
    return float(np.mean(aucs)) if aucs else float("nan")


def per_col_auc(P: pd.DataFrame, Y: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score
    return {c: (roc_auc_score(Y[c], P[c]) if Y[c].nunique() > 1 else float("nan")) for c in TARGETS}


def fmt(x) -> str:
    return "  n/a " if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="weight on the teacher: T_new = (1-a)*T_consensus + a*T_teacher (non-gold only)")
    ap.add_argument("--labels", default="data/labels/consensus_v1.parquet")
    ap.add_argument("--oof-dir", default="data/oof")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help="comma-separated arm names; teacher = per-study mean of their OOF probabilities")
    ap.add_argument("--out", default="data/labels/consensus_v2_distill.parquet")
    ap.add_argument("--reweight", action="store_true",
                    help="also scale NON-gold weights by 0.5 + 0.5*(1 - |T_consensus - T_teacher|) (default OFF)")
    ap.add_argument("--no-write", action="store_true", help="diagnostics only, do not write the parquet")
    args = ap.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit(f"ERROR: --alpha {args.alpha} outside [0, 1]")
    os.chdir(ROOT)

    # ---- consensus ---------------------------------------------------------------------
    cons = pd.read_parquet(args.labels)
    if ID_COL in cons.columns:
        cons = cons.set_index(ID_COL)
    cons.index = cons.index.astype(str)
    cols_in = list(cons.columns)
    dtypes_in = cons.dtypes.copy()
    gold = cons["is_gold"].astype(bool).values
    print(f"[labels] {args.labels}: {len(cons)} studies, gold={int(gold.sum())}, "
          f"folds={sorted(cons['fold'].unique().tolist())}")

    # ---- teacher -----------------------------------------------------------------------
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    mats = []
    for a in arms:
        d = stitch_arm(a, args.oof_dir)
        missing = cons.index.difference(d.index)
        assert len(missing) == 0, (f"arm {a}: {len(missing)} consensus studies have no teacher row "
                                   f"(first: {list(missing[:3])})")
        mats.append(d.reindex(cons.index)[TARGETS].values)
    teacher_v = np.mean(np.stack(mats, axis=0), axis=0)
    assert teacher_v.shape == (len(cons), len(TARGETS)), teacher_v.shape
    assert np.isfinite(teacher_v).all(), "teacher has NaN/inf"
    assert len(cons) == 4407, f"expected 4407 consensus studies, got {len(cons)}"
    teacher = pd.DataFrame(teacher_v, index=cons.index, columns=TARGETS)
    print(f"[teacher] mean of {len(arms)} arms: {teacher.shape[0]} studies x {teacher.shape[1]} targets, "
          f"no NaNs, range {teacher_v.min():.4f}..{teacher_v.max():.4f}")

    # ---- blend (non-gold only) ---------------------------------------------------------
    cons_v = cons[TARGETS].values.astype(np.float64)
    blend_v = (1.0 - args.alpha) * cons_v + args.alpha * teacher_v
    blend_v[gold] = cons_v[gold]                        # gold rows: EXACTLY consensus_v1
    out = cons.copy()
    out[TARGETS] = blend_v.astype(np.float32)

    w_v = cons[W_COLS].values.astype(np.float64).copy()
    if args.reweight:
        agree = 1.0 - np.abs(cons_v - teacher_v)
        scaled = w_v * (0.5 + 0.5 * agree)
        scaled[gold] = w_v[gold]                        # gold weights stay 3.0
        out[W_COLS] = scaled.astype(np.float32)
        print(f"[weights] --reweight ON: non-gold mean weight {w_v[~gold].mean():.4f} -> "
              f"{scaled[~gold].mean():.4f}")
    else:
        print(f"[weights] copied unchanged (mean {w_v.mean():.4f}); --reweight is OFF")

    out["is_gold"] = cons["is_gold"].values
    out["fold"] = cons["fold"].values
    out = out[cols_in]
    for c in cols_in:                                   # same dtypes as consensus_v1
        out[c] = out[c].astype(dtypes_in[c])
    out.index.name = cons.index.name

    # gold invariance, proved on the frame we are about to write
    gi = cons.index[gold]
    assert np.array_equal(out.loc[gi, TARGETS].values, cons.loc[gi, TARGETS].values), "gold targets changed!"
    assert np.array_equal(out.loc[gi, W_COLS].values, cons.loc[gi, W_COLS].values), "gold weights changed!"
    assert out.index.equals(cons.index) and list(out.columns) == cols_in
    assert (out.dtypes == dtypes_in).all(), out.dtypes[out.dtypes != dtypes_in]

    if not args.no_write:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out.to_parquet(args.out)
        print(f"[write] {args.out}: {out.shape[0]} x {out.shape[1]}  "
              f"({os.path.getsize(args.out)/1024:.0f} KiB)")
        # ---- the trainer's own loader must accept it, identically shaped ---------------
        from src.rsna.dataset_v2 import load_labels_v2
        a = load_labels_v2(args.labels)
        b = load_labels_v2(args.out)
        assert a.shape == b.shape and a.index.equals(b.index) and list(a.columns) == list(b.columns), \
            "load_labels_v2 disagrees between consensus_v1 and the distilled table"
        assert np.array_equal(a.loc[gi, TARGETS].values, b.loc[gi, TARGETS].values)
        chg = float(np.abs(a[TARGETS].values - b[TARGETS].values).mean())
        print(f"[verify] load_labels_v2 OK: same shape {b.shape}, same index, same columns, "
              f"gold identical, mean|target change| over all rows = {chg:.4f}")

    # ---- diagnostics -------------------------------------------------------------------
    ng = ~gold
    idx_ng = cons.index[ng]
    sv = pd.read_csv(STEVEN_CSV).set_index(ID_COL)
    sv.index = sv.index.astype(str)
    ys = (sv.reindex(idx_ng)[TARGETS] > 0.5).astype(int)
    tr = pd.read_csv(TRAIN_CSV).set_index(ID_COL)
    tr.index = tr.index.astype(str)
    gold_lab = tr.dropna(subset=TARGETS)[TARGETS].astype(int)
    gi2 = cons.index.intersection(gold_lab.index)
    yg = gold_lab.loc[gi2]
    cons_ng_tbl = pd.read_parquet(NOGOLD) if os.path.exists(NOGOLD) else None
    if cons_ng_tbl is not None:
        cons_ng_tbl.index = cons_ng_tbl.index.astype(str)

    C = pd.DataFrame(cons_v, index=cons.index, columns=TARGETS)
    B = pd.DataFrame(blend_v, index=cons.index, columns=TARGETS)
    dabs = np.abs(blend_v - cons_v)
    corr = {t: float(np.corrcoef(cons_v[ng, i], teacher_v[ng, i])[0, 1]) for i, t in enumerate(TARGETS)}
    a_c = per_col_auc(C.loc[idx_ng], ys); a_t = per_col_auc(teacher.loc[idx_ng], ys)
    a_b = per_col_auc(B.loc[idx_ng], ys)
    g_c = per_col_auc(C.loc[gi2], yg); g_t = per_col_auc(teacher.loc[gi2], yg)
    g_b = per_col_auc(B.loc[gi2], yg)
    g_n = per_col_auc(cons_ng_tbl.loc[gi2, TARGETS], yg) if cons_ng_tbl is not None else {t: np.nan for t in TARGETS}

    print(f"\n=== diagnostics: ALPHA={args.alpha:g}, reweight={'ON' if args.reweight else 'OFF'}, "
          f"teacher = mean({', '.join(arms)}) ===")
    print(f"non-gold n={int(ng.sum())} (AUC vs steven llm_labels_v4_blend > 0.5)   "
          f"gold n={len(gi2)} (AUC vs train.csv 0/1)")
    head = (f"{'target':<18}{'mean|dT|':>9}{'corr':>7} | {'cons':>7}{'teach':>7}{'blend':>7} | "
            f"{'cons':>7}{'consNG':>7}{'teach':>7}{'blend':>7}")
    print(f"{'':<18}{'':>9}{'':>7} | {'--- steven (non-gold) ---':^23} | {'------ gold 58 -------':^30}")
    print(head)
    print("-" * len(head))
    for i, t in enumerate(TARGETS):
        print(f"{t:<18}{dabs[ng, i].mean():>9.4f}{corr[t]:>7.3f} | "
              f"{fmt(a_c[t]):>7}{fmt(a_t[t]):>7}{fmt(a_b[t]):>7} | "
              f"{fmt(g_c[t]):>7}{fmt(g_n[t]):>7}{fmt(g_t[t]):>7}{fmt(g_b[t]):>7}")
    print("-" * len(head))
    m_c, m_t, m_b = macro_auc(C.loc[idx_ng], ys), macro_auc(teacher.loc[idx_ng], ys), macro_auc(B.loc[idx_ng], ys)
    mg_c, mg_t, mg_b = macro_auc(C.loc[gi2], yg), macro_auc(teacher.loc[gi2], yg), macro_auc(B.loc[gi2], yg)
    mg_n = macro_auc(cons_ng_tbl.loc[gi2, TARGETS], yg) if cons_ng_tbl is not None else float("nan")
    print(f"{'MACRO':<18}{dabs[ng].mean():>9.4f}{np.mean(list(corr.values())):>7.3f} | "
          f"{fmt(m_c):>7}{fmt(m_t):>7}{fmt(m_b):>7} | {fmt(mg_c):>7}{fmt(mg_n):>7}{fmt(mg_t):>7}{fmt(mg_b):>7}")
    print("\nnotes: gold `cons`/`blend` are 1.0000 BY CONSTRUCTION (consensus_v1 stores the true 0/1 on")
    print("       those 58 rows and the blend leaves them untouched) - the informative column is `consNG`")
    print(f"       ({NOGOLD}, the un-overwritten soft consensus). The steven panel favours `cons` for the")
    print("       same reason: steven is one of the three teachers the consensus was built from, so it is")
    print("       not an independent yardstick for the label itself - it is here to show how far the blend")
    print("       moves away from the report labels, and the gold panel is the one that measures quality.")


if __name__ == "__main__":
    main()
