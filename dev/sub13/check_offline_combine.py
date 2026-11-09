"""Offline recomputation of the notebook's final submission from its dumped arm frames (RSNA_DUMP_ARMS), with an
INDEPENDENT implementation (scipy rankdata, not pandas rank): per arm, per column, percentile ranks (average ties / n)
within the study set -- the inner join of the required arms, ranked within each arm; an optional arm (the reader)
ranked within the studies it has -- then (sum_a w_a r_a) / (sum of the weights of the arms present in that row),
then the per-column percentile rank of that score (write_submission), in test.csv order, 0.5 for absent studies.
Usage: .venv/bin/python -P dev/sub13/check_offline_combine.py --arms DIR --weights hr=3,v2=10,reader=6 \
           --sub submission.csv --test test.csv [--optional reader]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
ID = "StudyInstanceUID"


def pct(v):
    return rankdata(v, method="average") / len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--optional", default="reader")
    a = ap.parse_args()
    W = {k: float(v) for k, v in (t.split("=") for t in a.weights.split(","))}
    frames = {k: pd.read_pickle(Path(a.arms) / f"arm_{k}.pkl") for k in W}
    for k, f in frames.items():
        f[ID] = f[ID].astype(str)
        frames[k] = f.set_index(ID)
    opt = set(a.optional.split(",")) if a.optional else set()
    req = [k for k in W if k not in opt]
    ids = None
    for k in req:
        ids = set(frames[k].index) if ids is None else ids & set(frames[k].index)
    ids = sorted(ids)
    num = np.zeros((len(ids), len(T)))
    den = np.zeros((len(ids), 1))
    for k, w in W.items():
        f = frames[k]
        have = [s for s in ids if s in f.index]
        r = np.full((len(ids), len(T)), np.nan)
        pos = [ids.index(s) for s in have]
        for j, c in enumerate(T):
            r[pos, j] = pct(f.loc[have, c].values.astype(np.float64))
        present = ~np.isnan(r[:, :1])
        num += w * np.nan_to_num(r)
        den += w * present
        print(f"[offline] arm {k}: weight {w}, covers {len(have)}/{len(ids)} studies")
    score = num / den
    final = np.column_stack([pct(score[:, j]) for j in range(len(T))])
    test = pd.read_csv(a.test, dtype={ID: str})[ID].tolist()
    exp = pd.DataFrame(final, index=ids, columns=T).reindex(test).fillna(0.5)
    sub = pd.read_csv(a.sub, dtype={ID: str}).set_index(ID)
    assert list(sub.index) == test and list(sub.columns) == T
    d = np.abs(sub.values - exp.values)
    print(f"[offline] {len(test)} studies x 12: max |submission - offline| = {d.max():.3e}; exact cells "
          f"{int((d == 0).sum())}/{d.size}")
    ok = d.max() <= 1e-12
    print("OFFLINE COMBINE MATCH" if ok else "OFFLINE COMBINE MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
