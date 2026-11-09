"""Why does CPU-fp32 reader output differ from the GB10 fp16 reference by up to ~3e-3 (B.1)?
Same windows (ReaderStudyDataset, byte-identical to the validated pipeline), forward per checkpoint in four modes:
  fp32_bs4  (the bundle's CPU path)      fp32_bs1 (batch-composition check)
  fp16_ac   (torch.autocast("cpu", float16): convs/linears/matmuls in fp16, norms in fp32 -- the CUDA fp16
             autocast policy emulated on CPU)      bf16_ac (same with bfloat16: a coarser-precision reference point)
and compare each with outputs/pub_reader/run_all/pred_k12_fold<k>.parquet (GB10, CUDA fp16 autocast, bs=2).
Usage: CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD .venv/bin/python -P dev/sub13/precision_probe.py --ids FILE
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
from src.rsna import reader_gpjw as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--modes", default="fp32_bs4,fp32_bs1,fp16_ac,bf16_ac")
    a = ap.parse_args()
    torch.set_num_threads(4)
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    root = str(ROOT / "data/raw/train_series")
    se = pd.read_csv(ROOT / "data/comp_csv/train_series.csv")
    se = se[se.StudyInstanceUID.isin(set(ids))]
    table = R.build_study_table(se, R.series_file_counts(se, root))
    ds = R.ReaderStudyDataset(ids, table, root, R.READER_K)
    items = [ds[i] for i in range(len(ids))]
    x = torch.stack([it[1] for it in items])
    m = torch.stack([it[2] for it in items])
    p = torch.stack([it[3] for it in items])
    models = R.load_models([str(ROOT / f"models/public/gpjw_reader/cnxt_v0_fold{f}.pt") for f in (0, 1, 2)], "cpu")
    refs = [pd.read_parquet(ROOT / f"outputs/pub_reader/run_all/pred_k12_fold{f}.parquet").loc[ids].values
            for f in (0, 1, 2)]
    res = {}
    for mode in a.modes.split(","):
        outs = []
        for (model, rs, _) in models:
            with torch.no_grad():
                if mode == "fp32_bs4":
                    o = torch.cat([model(x[i:i + 4], m[i:i + 4], p[i:i + 4], res=rs).float().sigmoid()
                                   for i in range(0, len(ids), 4)])
                elif mode == "fp32_bs1":
                    o = torch.cat([model(x[i:i + 1], m[i:i + 1], p[i:i + 1], res=rs).float().sigmoid()
                                   for i in range(len(ids))])
                else:
                    dt = torch.float16 if mode == "fp16_ac" else torch.bfloat16
                    with torch.autocast("cpu", dtype=dt):
                        o = torch.cat([model(x[i:i + 2], m[i:i + 2], p[i:i + 2], res=rs).float().sigmoid()
                                       for i in range(0, len(ids), 2)])
            outs.append(o.numpy().astype(np.float64))
        res[mode] = outs
        per = [float(np.abs(o - r).max()) for o, r in zip(outs, refs)]
        mean_d = float(np.abs(np.mean(outs, 0) - np.mean(refs, 0)).max())
        print(f"[{mode:8s}] vs GB10-fp16 reference: max|dp| per ckpt {['%.2e' % v for v in per]} "
              f"| 3-ckpt mean {mean_d:.2e} | mean|dp| {float(np.mean([np.abs(o - r).mean() for o, r in zip(outs, refs)])):.2e}",
              flush=True)
    if "fp32_bs4" in res and "fp32_bs1" in res:
        d = max(float(np.abs(u - v).max()) for u, v in zip(res["fp32_bs4"], res["fp32_bs1"]))
        print(f"[batch  ] fp32 bs=4 vs bs=1: max|dp| {d:.2e}")
    if "fp32_bs4" in res and "fp16_ac" in res:
        d = max(float(np.abs(u - v).max()) for u, v in zip(res["fp32_bs4"], res["fp16_ac"]))
        print(f"[prec   ] CPU fp32 vs CPU fp16-autocast: max|dp| {d:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
