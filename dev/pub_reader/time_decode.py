"""CPU-only timing of the reader's per-study decode (load_series of every chosen series + k=12 windows).

Run it pinned to one core to get single-core seconds per study, e.g. on a Cortex-A725 (CPUs 0-4, 10-14 here) and on
a Cortex-X925 (CPUs 5-9, 15-19) to see how the cost scales with core speed; Kaggle scaling is done in the report.
No GPU, one process (not a heavy CPU job). Usage:
  PYTHONPATH=$PWD taskset -c 2 nice -n 10 .venv/bin/python -P dev/pub_reader/time_decode.py --sample 40 --seed 7
"""
import argparse
import time

import cv2
import numpy as np
import pandas as pd

from dev.pub_reader import gpjw_port as G
from dev.pub_reader.common import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--passes", type=int, default=2, help="pass 1 warms the page cache; the last pass is reported")
    a = ap.parse_args()
    cv2.setNumThreads(1)
    root = str(ROOT / "data/raw/train_series")
    tr = pd.read_csv(ROOT / "data/comp_csv/train.csv")
    studies = pd.Series(tr.StudyInstanceUID.tolist()).sample(a.sample, random_state=a.seed).tolist()
    se = pd.read_csv(ROOT / "data/comp_csv/train_series.csv")
    se = se[se.StudyInstanceUID.isin(set(studies))]
    table = G.build_study_table(se, G.series_file_counts(se, root))
    for p in range(a.passes):
        ts, nsl = [], []
        for st in studies:
            t0 = time.perf_counter()
            memo = {}

            def vol_of(sr, st=st, memo=memo):
                if sr not in memo:
                    memo[sr] = G.load_series(f"{root}/{st}/{sr}")
                return memo[sr][0]

            G.study_windows(vol_of, st, table.get(st, [[] for _ in range(G.N_SLOTS)]), 12)
            ts.append(time.perf_counter() - t0)
            nsl.append(sum(v[1]["n_raw"] for v in memo.values()))
        ts = np.asarray(ts)
        print(f"pass {p + 1}: {len(studies)} studies, decode+windows s/study mean {ts.mean():.3f} median "
              f"{np.median(ts):.3f} p90 {np.percentile(ts, 90):.3f}; DICOM slices read/study {np.mean(nsl):.1f}; "
              f"ms/slice {1000 * ts.sum() / max(sum(nsl), 1):.2f}", flush=True)


if __name__ == "__main__":
    main()
