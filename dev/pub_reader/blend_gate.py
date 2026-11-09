"""Task 4: pre-registered blend gate for the public reader inside our sub-#10b ensemble.

Ensemble (sub #10b, dev/fit_v2_stitched2.py): (3*rank(hr_tta3) + 10*v2) / 13, v2 = (rank(lr) + rank(c110)) / 2.
Candidate: (3*rank(hr) + 10*v2 + w*rank(reader)) / (13 + w), one global w in {0, 1, 2, 3, 4, 6}.
All percentile ranks are computed WITHIN the overlap set (our stitched OOF  x  the reader's honest OOF = 2,610
held-out + 58 gold), i.e. as if the overlap were the test set.
Metrics: steven-v4 (> 0.5) macro-AUC on the full overlap (2,668; repo convention: gold rows carry steven labels too)
= the OOF metric that picks w; also shown on the 2,610 non-gold rows; gold-58 macro-AUC (official labels) = check.
Gate (pre-registered in the task): best dOOF >= +0.003 AND dGold >= 0 at the same w (w chosen by OOF only).

Usage: PYTHONPATH=$PWD .venv/bin/python -P dev/pub_reader/blend_gate.py [--oof data/oof/pubreader_gpjw_oof.parquet]
"""
import argparse

import numpy as np
import pandas as pd

from dev.pub_reader.common import ROOT, T, gold58, macro, per_col, reader_folds, reader_pred, steven_bin, sub10b_blend, \
    sub10b_members

WS = [0, 1, 2, 3, 4, 6]


def paired_boot(EA, EB, Y, n=1000, seed=0):
    """CI of macro(EB) - macro(EA) over bootstrap resamples of studies (same resample for both)."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(Y.index)
    d = []
    for _ in range(n):
        b = rng.choice(ids, len(ids))
        d.append(macro(EB.loc[b], Y.loc[b]) - macro(EA.loc[b], Y.loc[b]))
    d = np.asarray(d)
    return np.percentile(d, [2.5, 50, 97.5]), float((d > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=str(ROOT / "data/oof/pubreader_gpjw_oof.parquet"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--leaky-variant", default="k12", help="for the LEAKY contrast row only")
    ap.add_argument("--extra-ws", default="8,10,13,20,39", help="informational weights outside the gate's grid")
    a = ap.parse_args()
    pd.set_option("display.width", 250)

    M = sub10b_members()
    idx = M["hr"].index
    for d in M.values():
        idx = idx.intersection(d.index)
    ys, g = steven_bin(), gold58()
    E_full = sub10b_blend(M, idx)
    gi_full = idx.intersection(g.index)
    print(f"[baseline] sub #10b stitched on all {len(idx)} studies: steven {macro(E_full, ys.loc[idx]):.4f} | "
          f"gold-58 {macro(E_full.loc[gi_full], g.loc[gi_full]):.4f}   (reference 0.8899 / 0.9135)")

    R = pd.read_parquet(a.oof)
    assert list(R.columns) == T
    ov = idx.intersection(R.index)
    gi = ov.intersection(g.index)
    weak = ov.difference(gi)
    print(f"[overlap] {len(ov)} studies = {len(weak)} held-out + {len(gi)} gold   (reader OOF: {a.oof})")
    Yov, Yw, Yg = ys.loc[ov], ys.loc[weak], g.loc[gi]
    cons = pd.read_parquet(ROOT / "data/labels/consensus_v1_nogold.parquet")[T]
    Yc = (cons.loc[ov] > 0.5).astype(int)  # informational second label set (as dev/fit_v2_stitched2.py)
    rr = R.loc[ov]
    print(f"[reader solo] steven overlap {macro(rr, Yov):.4f} | steven 2,610 {macro(rr.loc[weak], Yw):.4f} | "
          f"consensus overlap {macro(rr, Yc):.4f} | gold-58 {macro(rr.loc[gi], Yg):.4f}")
    E0 = sub10b_blend(M, ov)
    v2 = (M["lr"].loc[ov].rank(pct=True) + M["c110"].loc[ov].rank(pct=True)) / 2

    def sp(A, B):
        return np.mean([A[c].corr(B[c], method="spearman") for c in T])

    print(f"[diversity] mean per-column Spearman on the overlap: reader vs sub10b {sp(rr, E0):.3f} | "
          f"reader vs hr {sp(rr, M['hr'].loc[ov]):.3f} | reader vs v2 arm {sp(rr, v2):.3f} | "
          f"hr vs v2 arm {sp(M['hr'].loc[ov], v2):.3f}")

    rows, E = [], {}
    for w in WS:
        E[w] = sub10b_blend(M, ov, R, w)
        rows.append({"w": w, "steven_overlap": macro(E[w], Yov), "steven_2610": macro(E[w].loc[weak], Yw),
                     "consensus_overlap": macro(E[w], Yc), "gold58": macro(E[w].loc[gi], Yg)})
    t = pd.DataFrame(rows).set_index("w")
    for c in ["steven_overlap", "steven_2610", "consensus_overlap", "gold58"]:
        t["d_" + c] = t[c] - t.loc[0, c]
    print("\n[w-table] (3*hr + 10*v2 + w*reader)/(13+w), ranks within the overlap:")
    print(t.round(4).to_string())

    cand = t.drop(index=0)
    w_star = int(cand["steven_overlap"].idxmax())
    d_oof, d_gold = t.loc[w_star, "d_steven_overlap"], t.loc[w_star, "d_gold58"]
    verdict = "PASS" if (d_oof >= 0.003 and d_gold >= 0) else "FAIL"
    print(f"\n[gate] w* (by OOF only) = {w_star}: dOOF {d_oof:+.4f} (need >= +0.0030), dGold {d_gold:+.4f} "
          f"(need >= 0) -> {verdict}")
    # nested check of the w choice: pick w on a random half of the overlap, score dOOF on the other half
    rng = np.random.default_rng(0)
    ids = np.asarray(ov)
    nd, picks = [], []
    for _ in range(50):
        perm = rng.permutation(len(ids))
        A, B = ids[perm[: len(ids) // 2]], ids[perm[len(ids) // 2:]]
        wa = max([w for w in WS if w], key=lambda w: macro(E[w].loc[A], Yov.loc[A]))
        picks.append(wa)
        nd.append(macro(E[wa].loc[B], Yov.loc[B]) - macro(E[0].loc[B], Yov.loc[B]))
    pick_counts = {int(k): int(v) for k, v in zip(*np.unique(picks, return_counts=True))}
    print(f"[nested 50x split-half] w picked on half A, dOOF on half B: mean {np.mean(nd):+.4f} "
          f"(sd {np.std(nd):.4f}); picks {pick_counts}")
    if a.boot:
        (lo, med, hi), p = paired_boot(E[0], E[w_star], Yov, a.boot)
        (glo, gmed, ghi), gp = paired_boot(E[0].loc[gi], E[w_star].loc[gi], Yg, a.boot, seed=1)
        print(f"[bootstrap {a.boot}x] dOOF at w*={w_star}: median {med:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], "
              f"P(d>0) {p:.2f} | dGold: median {gmed:+.4f}, 95% CI [{glo:+.4f}, {ghi:+.4f}], P(d>0) {gp:.2f}")
    if a.extra_ws:
        xr = []
        for w in [int(x) for x in a.extra_ws.split(",")]:
            Ex = sub10b_blend(M, ov, R, w)
            xr.append({"w": w, "reader_share": w / (13 + w), "steven_overlap": macro(Ex, Yov),
                       "d_steven_overlap": macro(Ex, Yov) - t.loc[0, "steven_overlap"],
                       "d_consensus_overlap": macro(Ex, Yc) - t.loc[0, "consensus_overlap"],
                       "d_gold58": macro(Ex.loc[gi], Yg) - t.loc[0, "gold58"]})
        print("\n[outside the pre-registered grid - INFORMATIONAL ONLY, not used by the gate]:")
        print(pd.DataFrame(xr).set_index("w").round(4).to_string())
    pc = pd.DataFrame({"sub10b": per_col(E[0], Yov), f"w={w_star}": per_col(E[w_star], Yov),
                       "reader": per_col(rr, Yov), "gold_sub10b": per_col(E[0].loc[gi], Yg),
                       f"gold_w={w_star}": per_col(E[w_star].loc[gi], Yg)})
    pc["d"] = pc[f"w={w_star}"] - pc["sub10b"]
    pc["d_gold"] = pc[f"gold_w={w_star}"] - pc["gold_sub10b"]
    print("\n[per column, informational only - not used to pick anything] steven overlap / gold-58:")
    print(pc.round(4).to_string())

    # LEAKY contrast (NOT the gate): the naive 3-checkpoint mean on the same overlap, i.e. two of the three
    # checkpoints trained on every held-out study. Shows how much a naive blend would overstate the gain.
    P = {k: reader_pred(a.leaky_variant, k) for k in (0, 1, 2)}
    L = sum(P[k] for k in (0, 1, 2)) / 3
    lrows = []
    for w in WS:
        El = sub10b_blend(M, ov, L, w)
        lrows.append({"w": w, "steven_overlap": macro(El, Yov), "gold58": macro(El.loc[gi], Yg)})
    lt = pd.DataFrame(lrows).set_index("w")
    lt["d_steven_overlap"] = lt.steven_overlap - lt.loc[0, "steven_overlap"]
    print(f"\n[LEAKY contrast, not the gate] reader = mean of 3 checkpoints on the overlap (solo steven "
          f"{macro(L.loc[ov], Yov):.4f}):")
    print(lt.round(4).to_string())
    # and on folds 3-4 the naive mean is in-sample for all three checkpoints
    f = reader_folds("stable")
    ins = idx.intersection(f.index[f.isin([3, 4])])
    print(f"[LEAKY] naive 3-ckpt mean solo on reader folds 3-4 ({len(ins)} studies, trained on by all three): "
          f"steven {macro(L.loc[ins], ys.loc[ins]):.4f}")


if __name__ == "__main__":
    main()
