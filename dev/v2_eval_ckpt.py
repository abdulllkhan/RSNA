"""Evaluate an EXISTING v2 checkpoint on a cache root, WITHOUT ever touching data/oof/.

`dev/v2_raw_oof.py` rebuilds a checkpoint from a `_last.pt` and *writes*
`data/oof/<name>_fold<k>.parquet`. This script does neither: it loads the checkpoint's
`model` weights as they are, predicts the fold's held-out studies off whichever cache root
is given, and prints the SAME two numbers (`steven macro`, `gold-fold macro`) plus the
per-column breakdown. Predictions go only where `--pred-out` says (default: nowhere).

Usage:
    PYTHONPATH=. .venv/bin/python dev/v2_eval_ckpt.py \
        --ckpt models/v2_effv2s384_c110_fold0.pt --cache-root data/cache_c110_384 \
        --device cuda --batch 2 --workers 2 --pred-out /tmp/pred.parquet
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rsna import TARGETS, ID_COL                       # noqa: E402
from src.rsna.infer_v2 import load_members_v2, predict_studies_v2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--cache-root", required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--fold", type=int, default=-1, help="-1 = the checkpoint's own fold")
ap.add_argument("--labels", default="data/labels/consensus_v1.parquet")
ap.add_argument("--pred-out", default=None,
                help="parquet for the raw predictions; MUST NOT be under data/oof/")
a = ap.parse_args()

if a.pred_out and os.path.abspath(a.pred_out).startswith(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "oof"))):
    raise SystemExit("refusing to write into data/oof/ — that holds the project's real OOF")

st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
args = dict(st["args"])
fold = int(args["fold"]) if a.fold < 0 else a.fold
print(f"[eval] ckpt={a.ckpt} name={args.get('name')} fold={fold} size={args.get('size')} "
      f"ckpt_crop_mm={args.get('crop_mm')} trained_on={args.get('cache_root')}")
print(f"[eval] evaluating on cache_root={a.cache_root}")
del st

labels = pd.read_parquet(a.labels)
ids = [str(s) for s in labels.index[labels.fold == fold]]
if a.limit:
    ids = ids[:a.limit]
print(f"[eval] fold {fold}: {len(ids)} held-out studies")

members = load_members_v2([a.ckpt], device=a.device)
pred = predict_studies_v2(members, a.cache_root, split="train", batch=a.batch,
                          workers=a.workers, device=a.device, study_ids=ids,
                          side_parquet="data/study_side.parquet",
                          order_parquet="data/series_order.parquet", dicom_root="data/raw")
if ID_COL not in pred.columns:
    pred = pred.reset_index().rename(columns={"index": ID_COL})
pred = pred[[ID_COL] + TARGETS]

sv = pd.read_csv("data/labels/stevenleehans_rsna-knee-llm-report-labels/"
                 "llm_labels_v4_blend.csv").set_index(ID_COL)
P = pred.set_index(ID_COL)
idx = P.index.intersection(sv.index)
per_col = {}
for t in TARGETS:
    y = (sv.loc[idx, t] > 0.5).astype(int)
    if y.nunique() > 1:
        per_col[t] = roc_auc_score(y, P.loc[idx, t])
macro = float(np.mean(list(per_col.values())))

tr = pd.read_csv("data/comp_csv/train.csv").set_index(ID_COL)
gold = tr.dropna(subset=TARGETS)[TARGETS]
g = P.index.intersection(gold.index)
gper = {}
if len(g) > 1:
    for t in TARGETS:
        if gold.loc[g, t].nunique() > 1:
            gper[t] = roc_auc_score(gold.loc[g, t].astype(int), P.loc[g, t])
gmacro = float(np.mean(list(gper.values()))) if gper else float("nan")

print(f"[eval] n_steven={len(idx)} steven macro={macro:.4f} | "
      f"gold-fold n={len(g)} macro={gmacro:.4f}")
print("[eval] per-column steven AUC:")
for t in TARGETS:
    print(f"   {t:<18s} {per_col.get(t, float('nan')):.4f}  gold={gper.get(t, float('nan')):.4f}")

if a.pred_out:
    os.makedirs(os.path.dirname(os.path.abspath(a.pred_out)), exist_ok=True)
    pred.to_parquet(a.pred_out, index=False)
    print(f"[eval] predictions -> {a.pred_out} ({len(pred)} rows)")
print("[eval] DONE (no data/oof file was written)")
