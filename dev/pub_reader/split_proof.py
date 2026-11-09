"""Task 2: prove the re-derived reader folds are the folds the public checkpoints were trained with.

For checkpoint cnxt_v0_fold<k> (which, if the split is right, never saw reader-fold k):
  macro-AUC vs binarised steven-v4 labels (> 0.5) on every reader fold 0..4 (non-gold studies), on its own fold
  (held out) vs the union of the four folds it trained on, the gap, a bootstrap CI of the gap, and its macro-AUC on
  the 58 gold studies (official labels; never trained on by any fold).
Within-fold comparison: on reader-fold k, the held-out checkpoint k vs the two checkpoints that trained on it.
Negative control: the same table with folds re-derived through an UNSTABLE argsort (only 12.5 % agreement):
  a wrong split must show no gap, which shows the test has power.
Also: per-class AUC of the honest 3-fold OOF + gold-58 next to the author's published per-class figures
  (nb medvax/rsna-knee-independent-2-5d-reader-trial c1 L69-71), as a reproduction check.

Usage: PYTHONPATH=$PWD .venv/bin/python -P dev/pub_reader/split_proof.py [--variant k12] [--boot 300]
"""
import argparse

import numpy as np
import pandas as pd

from dev.pub_reader.common import T, gold58, macro, per_col, reader_folds, reader_pred, steven_bin, steven_soft

# author's published per-class figures for the 3-fold ensemble (held-out folds vs v0 labels, and gold-58)
PUB_GOLD = {"Baker's": 0.976, "Contusion": 0.966, "Medial OA": 0.966, "ACL": 0.965, "Medial Meniscus": 0.963,
            "MCL": 0.946, "Fracture": 0.911, "Lateral Meniscus": 0.889, "Effusion": 0.880, "Lateral OA": 0.865,
            "PF OA": 0.843, "Synovitis": 0.761}
PUB_WEAK_V0 = {"Baker's": 0.927, "Contusion": 0.884, "Medial OA": 0.891, "ACL": 0.903, "Medial Meniscus": 0.933,
               "MCL": 0.828, "Fracture": 0.918, "Lateral Meniscus": 0.884, "Effusion": 0.875, "Lateral OA": 0.864,
               "PF OA": 0.853, "Synovitis": 0.900}


def boot_gap(P, Y, own_idx, tr_idx, n, seed=0):
    rng = np.random.default_rng(seed)
    own_idx, tr_idx = np.asarray(own_idx), np.asarray(tr_idx)
    out = []
    for _ in range(n):
        a = rng.choice(own_idx, len(own_idx))
        b = rng.choice(tr_idx, len(tr_idx))
        out.append(macro(P.loc[b], Y.loc[b]) - macro(P.loc[a], Y.loc[a]))
    return np.percentile(out, [2.5, 97.5])


def bce(P, S):
    p = P.clip(1e-6, 1 - 1e-6)
    return float(-(S * np.log(p) + (1 - S) * np.log(1 - p)).mean().mean())


def table(folds, P, ys, ss, g, boot):
    rows = []
    for k in (0, 1, 2):
        r = {"ckpt": f"fold{k}"}
        for f in range(5):
            ids = folds.index[folds == f]
            r[f"f{f}"] = macro(P[k].loc[ids], ys.loc[ids])
        own = folds.index[folds == k]
        trn = folds.index[(folds >= 0) & (folds != k)]
        r["own(held-out)"] = macro(P[k].loc[own], ys.loc[own])
        r["trained-on"] = macro(P[k].loc[trn], ys.loc[trn])
        r["gap"] = r["trained-on"] - r["own(held-out)"]
        if boot:
            lo, hi = boot_gap(P[k], ys, own, trn, boot, seed=k)
            r["gap95"] = f"[{lo:+.4f}, {hi:+.4f}]"
        r["bce_own"] = bce(P[k].loc[own], ss.loc[own])
        r["bce_trained"] = bce(P[k].loc[trn], ss.loc[trn])
        r["gold58"] = macro(P[k].loc[g.index], g)
        rows.append(r)
    return pd.DataFrame(rows).set_index("ckpt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="k12")
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    P = {k: reader_pred(a.variant, k) for k in (0, 1, 2)}
    assert all(len(p) == 4407 for p in P.values())
    ys, ss, g = steven_bin(), steven_soft(), gold58()
    folds = reader_folds("stable")
    print(f"== variant {a.variant}; reader folds (stable GroupKFold, == survey csv): "
          f"{folds.value_counts().sort_index().to_dict()}")
    t = table(folds, P, ys, ss, g, a.boot)
    print("\n[split proof] macro-AUC vs steven-v4 > 0.5, by reader fold (non-gold), held-out vs trained-on, gold-58:")
    print(t.round(4).to_string())

    print("\n[within-fold] same studies, held-out checkpoint vs the two checkpoints that trained on them:")
    for f in (0, 1, 2):
        ids = folds.index[folds == f]
        others = [macro(P[k].loc[ids], ys.loc[ids]) for k in (0, 1, 2) if k != f]
        print(f"  reader fold {f}: held-out ckpt{f} {macro(P[f].loc[ids], ys.loc[ids]):.4f} | trained ckpts "
              f"{others[0]:.4f} / {others[1]:.4f} | diff {np.mean(others) - macro(P[f].loc[ids], ys.loc[ids]):+.4f}")
    for f in (3, 4):
        ids = folds.index[folds == f]
        print(f"  reader fold {f} (trained by all three): " +
              " / ".join(f"{macro(P[k].loc[ids], ys.loc[ids]):.4f}" for k in (0, 1, 2)))

    fq = reader_folds("quicksort")
    agree = (fq[folds >= 0] == folds[folds >= 0]).mean()
    tq = table(fq, P, ys, ss, g, 0)
    print(f"\n[negative control] folds through an unstable argsort ({agree:.1%} agreement with the real split):")
    print(tq[["own(held-out)", "trained-on", "gap"]].round(4).to_string())

    # honest 3-fold OOF (each held-out study by the checkpoint that never saw it) + gold by the 3-ckpt mean
    held = folds.index[folds.isin([0, 1, 2])]
    oof = pd.concat([P[k].loc[folds.index[folds == k]] for k in (0, 1, 2)])
    gm = sum(P[k].loc[g.index] for k in (0, 1, 2)) / 3
    print(f"\n[reader solo] honest OOF {len(held)} held-out studies: steven macro {macro(oof, ys.loc[oof.index]):.4f}"
          f" | gold-58 (3-ckpt mean) {macro(gm, g):.4f} | gold-58 per ckpt "
          + " / ".join(f"{macro(P[k].loc[g.index], g):.4f}" for k in (0, 1, 2)))
    # reproduction check against the author's per-class figures
    pg = per_col(gm, g)
    pub = pd.Series(PUB_GOLD)[T]
    sources = {
        "steven": ss,
        "pilkwang": pd.read_csv("data/labels/pilkwang_rsna-knee-llm-labels/report_labels_v2.csv").set_index("StudyInstanceUID")[T].astype(float),
        "lixin": pd.read_csv("data/labels/lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv").set_index("StudyInstanceUID")[T].astype(float),
    }
    ids = oof.index
    num = sum(s.reindex(ids).fillna(0) for s in sources.values())
    den = sum(s.reindex(ids).notna().astype(float) for s in sources.values())
    v0p = ((num / den).clip(0.01, 0.99) > 0.5).astype(int)
    pw = per_col(oof, v0p)
    cmp_ = pd.DataFrame({"gold58_ours": pg, "gold58_author": pub, "d_gold": pg - pub,
                         "weak_v0approx_ours": pw, "weak_v0_author": pd.Series(PUB_WEAK_V0)[T],
                         "weak_steven_ours": per_col(oof, ys.loc[ids])})
    cmp_.loc["MEAN"] = cmp_.mean()
    print("\n[reproduction] per-class AUC, ours (raw DICOM -> released inference) vs the author's published figures"
          "\n  (weak_v0approx uses 3 of the author's 4 label tables - laymond Qwen is not available locally):")
    print(cmp_.round(3).to_string())


if __name__ == "__main__":
    main()
