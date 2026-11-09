"""Compare reader predictions written by reader_gpjw.write_reader_outputs (reader_pred_fold<k>.csv,
reader_pred_mean.csv, reader_study_stats.csv) against the VALIDATED per-checkpoint predictions
outputs/pub_reader/run_all/pred_k12_fold<k>.parquet (GB10, fp16 autocast, 4,407 training studies).

Prints, per checkpoint and for the 3-checkpoint probability mean: max |dp|, mean |dp|, and the per-column
Spearman correlation over the compared studies; plus the slot/error counters.
Usage: .venv/bin/python dev/sub13/compare_reader_preds.py <dir> [--ref outputs/pub_reader/run_all] [--tol 2e-3]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ref", default=str(ROOT / "outputs/pub_reader/run_all"))
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--prefix", default="reader")
    a = ap.parse_args()
    d = Path(a.dir)
    refs = {f: pd.read_parquet(Path(a.ref) / f"pred_k12_fold{f}.parquet") for f in (0, 1, 2)}
    st = pd.read_csv(d / f"{a.prefix}_study_stats.csv")
    print(f"[stats] studies={len(st)} valid={int(st.valid.sum())} slots_mean={st.slots_valid.mean():.2f} "
          f"series={int(st.series_selected.sum())} series_err={int(st.series_err.sum())} "
          f"slices_bad={int(st.slices_bad.sum())} forward={st.forward.value_counts().to_dict()}")
    meta = pd.read_parquet(Path(a.ref) / "meta.parquet")
    if "n_series" in meta.columns:
        m = meta.loc[st.StudyInstanceUID]
        print(f"[stats] reference run: series decoded per study {m.n_series.mean():.2f} (ours "
              f"{st.series_selected.mean():.2f}), series errors {int(m.n_err.sum())}")
    worst = 0.0
    out = {}
    for name in ["fold0", "fold1", "fold2", "mean"]:
        p = pd.read_csv(d / f"{a.prefix}_pred_{name}.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[T]
        if name == "mean":
            r = sum(refs[f].loc[p.index] for f in (0, 1, 2)) / 3.0
        else:
            r = refs[int(name[-1])].loc[p.index]
        diff = (p.values - r.values.astype(np.float64))
        rho = [spearmanr(p[c], r[c]).correlation for c in T]
        mx = float(np.abs(diff).max())
        worst = max(worst, mx)
        out[name] = (mx, float(np.abs(diff).mean()), float(np.min(rho)), float(np.mean(rho)))
        print(f"[{name:5s}] n={len(p)} max|dp|={mx:.2e} mean|dp|={np.abs(diff).mean():.2e} "
              f"spearman min={np.min(rho):.5f} mean={np.mean(rho):.5f}")
        if name == "mean":
            print("        per-column max|dp|: " + " ".join(f"{c[:6]}={np.abs(diff[:, i]).max():.1e}"
                                                    for i, c in enumerate(T)))
            print("        per-column spearman: " + " ".join(f"{c[:6]}={rh:.4f}" for c, rh in zip(T, rho)))
            j = np.unravel_index(np.abs(diff).argmax(), diff.shape)
            print(f"        worst cell: study {p.index[j[0]][-12:]} col {T[j[1]]} ours {p.values[j]:.6f} "
                  f"ref {r.values[j]:.6f}")
    ok = out["mean"][0] <= a.tol
    print(f"PARITY {'PASS' if ok else 'FAIL'}: max|dp| of the 3-checkpoint mean {out['mean'][0]:.2e} "
          f"(tol {a.tol:.0e}); worst per-checkpoint {worst:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
