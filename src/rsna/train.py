"""Train ONE fold end-to-end.

- bf16 autocast + channels_last, AdamW + warmup-cosine, BCE on soft targets
  (optional mild rare-column pos_weight).
- Checkpoints each epoch (resumable): models/<name>_fold<k>_last.pt.
- Saves best-by-OOF-fold-AUC to models/<name>_fold<k>.pt and per-study OOF preds to
  data/oof/<name>_fold<k>.parquet.
- Prints per-epoch OOF macro-AUC (vs binarized steven labels on the held-out fold) and
  gold macro-AUC (on the gold studies that fall in this val fold; noisy, flagged).

Reproduce (fold 0):
    .venv/bin/python -m src.rsna.train --fold 0 --epochs 10 --name effv2s_k24
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import TARGETS, ID_COL
from .dataset import StudyBagDataset, build_study_index
from .metric import auc_report, load_gold, macro_auc, per_column_auc
from .model import ModelConfig, build_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_paths() -> Dict[str, str]:
    return dict(
        img_root=os.path.join(ROOT, "data", "processed_jpeg224"),
        train_series=os.path.join(ROOT, "data", "processed_jpeg224", "train_series.csv"),
        labels_csv=os.path.join(
            ROOT, "data", "labels", "stevenleehans_rsna-knee-llm-report-labels",
            "llm_labels_v4_blend.csv",
        ),
        folds=os.path.join(ROOT, "data", "folds.csv"),
        train_csv=os.path.join(ROOT, "data", "comp_csv", "train.csv"),
        index_cache=os.path.join(ROOT, "data", "oof", "study_index_train.pkl"),
        models_dir=os.path.join(ROOT, "models"),
        oof_dir=os.path.join(ROOT, "data", "oof"),
    )


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def predict(model, loader, device) -> pd.DataFrame:
    model.eval()
    ids, probs = [], []
    for x, _y, sid in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        ids.extend(list(sid))
    probs = np.concatenate(probs, axis=0)
    df = pd.DataFrame(probs, columns=TARGETS)
    df.insert(0, ID_COL, ids)
    return df


def compute_fingerprint(model, device, k: int = 24) -> np.ndarray:
    """Deterministic output on a fixed synthetic input; stored in the checkpoint."""
    # true fp32 (TF32 DISABLED), NO autocast -> portable across GPUs/CPU (catches wrong load)
    model.eval()
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        g = torch.Generator().manual_seed(1234)
        x = torch.randn(1, k, 3, 224, 224, generator=g).to(device)
        with torch.no_grad():
            out = torch.sigmoid(model(x).float()).cpu().numpy().ravel()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cudnn
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--name", type=str, default="effv2s_k24")
    ap.add_argument("--backbone", type=str, default="tf_efficientnetv2_s")
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-pos-weight", type=float, default=3.0,
                    help="cap on per-column pos_weight (neg/pos), for rare-column upweighting")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-train-minutes", type=float, default=0.0,
                    help="if >0, stop after this many minutes of wall-clock (time-box)")
    ap.add_argument("--limit-studies", type=int, default=0, help="smoke: cap #train studies")
    args = ap.parse_args()

    P = default_paths()
    os.makedirs(P["models_dir"], exist_ok=True)
    os.makedirs(P["oof_dir"], exist_ok=True)
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- data ----
    labels = pd.read_csv(P["labels_csv"])
    folds = pd.read_csv(P["folds"])
    gold = load_gold(P["train_csv"])
    index = build_study_index(P["img_root"], P["train_series"], split="train",
                              cache_path=P["index_cache"])

    tr_ids = folds[folds.fold != args.fold][ID_COL].astype(str).tolist()
    va_ids = folds[folds.fold == args.fold][ID_COL].astype(str).tolist()
    if args.limit_studies:
        tr_ids = tr_ids[: args.limit_studies]
        va_ids = va_ids[: max(32, args.limit_studies // 4)]

    cfg = ModelConfig(backbone=args.backbone, pretrained=True, dropout=args.dropout)
    model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
    dc = model.data_config()
    mean, std = dc["mean"], dc["std"]
    print(f"[data-config] mean={mean} std={std} feat_dim={model.feat_dim}")

    tr_ds = StudyBagDataset(tr_ids, index, labels, k=args.k, train=True, mean=mean, std=std)
    va_ds = StudyBagDataset(va_ids, index, labels, k=args.k, train=False, mean=mean, std=std)
    print(f"[fold {args.fold}] train={len(tr_ds)} val={len(va_ds)} studies")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                           pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
                           prefetch_factor=4 if args.workers > 0 else None)
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                           pin_memory=True, persistent_workers=args.workers > 0,
                           prefetch_factor=4 if args.workers > 0 else None)

    # ---- val truth (binarized steven for OOF-AUC; real 0/1 for gold) ----
    lab_idx = labels.set_index(ID_COL)[TARGETS]
    va_present = [s for s in va_ids if s in index]
    y_oof = (lab_idx.loc[va_present].values >= 0.5).astype(int)
    y_oof_df = pd.DataFrame(y_oof, columns=TARGETS); y_oof_df.insert(0, ID_COL, va_present)

    # ---- pos_weight (mild rare-column upweight) ----
    tr_present = [s for s in tr_ids if s in index]
    ytr = (lab_idx.loc[tr_present].values >= 0.5).astype(float)
    pos = ytr.sum(0); neg = len(ytr) - pos
    pw = np.clip(neg / np.maximum(pos, 1.0), 1.0, args.max_pos_weight)
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
    print("[pos_weight]", {t: round(float(w), 2) for t, w in zip(TARGETS, pw)})
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(tr_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup = int(args.warmup_frac * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: warmup_cosine(s, total_steps, warmup))

    ckpt_last = os.path.join(P["models_dir"], f"{args.name}_fold{args.fold}_last.pt")
    ckpt_best = os.path.join(P["models_dir"], f"{args.name}_fold{args.fold}.pt")
    oof_path = os.path.join(P["oof_dir"], f"{args.name}_fold{args.fold}.parquet")

    start_epoch, best_auc, gstep = 0, -1.0, 0
    if args.resume and os.path.exists(ckpt_last):
        st = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); start_epoch = st["epoch"] + 1
        best_auc = st.get("best_auc", -1.0); gstep = st.get("gstep", 0)
        print(f"[resume] from epoch {start_epoch}, best_auc={best_auc:.4f}")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_t = time.time(); seen = 0; run_loss = 0.0
        for it, (x, y, _sid) in enumerate(tr_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            gstep += 1; seen += x.shape[0] * x.shape[1]; run_loss += loss.item()
            if it % 30 == 0:
                ips = seen / max(1e-6, time.time() - ep_t)
                print(f"  e{epoch} it{it}/{steps_per_epoch} loss={run_loss/(it+1):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {ips:.0f} img/s", flush=True)
        ep_time = time.time() - ep_t
        ips = seen / max(1e-6, ep_time)

        # ---- validate ----
        preds = predict(model, va_loader, device)
        preds = preds.set_index(ID_COL).loc[va_present].reset_index()
        oof_auc = macro_auc(y_oof, preds[TARGETS].values)
        g_auc, g_rep = _gold_on_fold(preds, gold)
        print(f"[epoch {epoch}] loss={run_loss/steps_per_epoch:.4f} "
              f"OOF_macroAUC={oof_auc:.4f} gold_foldAUC={g_auc:.4f} "
              f"epoch_time={ep_time:.0f}s {ips:.0f} img/s", flush=True)
        rep = auc_report(y_oof, preds[TARGETS].values)
        print("  OOF per-finding:", {t: round(float(rep[t]), 3) for t in TARGETS})

        # ---- checkpoint (resumable) ----
        fp = compute_fingerprint(model, device, k=args.k)
        state = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                     epoch=epoch, best_auc=best_auc, gstep=gstep, cfg=vars(cfg),
                     args=vars(args), mean=list(map(float, mean)), std=list(map(float, std)),
                     fingerprint=fp, targets=TARGETS, oof_auc=oof_auc)
        torch.save(state, ckpt_last)
        if oof_auc > best_auc:
            best_auc = oof_auc
            state["best_auc"] = best_auc
            torch.save(state, ckpt_best)
            preds.to_parquet(oof_path, index=False)
            print(f"  * new best OOF={best_auc:.4f} -> saved {ckpt_best} + OOF", flush=True)

        if args.max_train_minutes and (time.time() - t0) / 60.0 >= args.max_train_minutes:
            print(f"[time-box] stopping after {(time.time()-t0)/60:.1f} min", flush=True)
            break

    print(f"[done] fold {args.fold} best OOF_macroAUC={best_auc:.4f} "
          f"total {(time.time()-t0)/60:.1f} min")
    return best_auc


def _gold_on_fold(preds: pd.DataFrame, gold: pd.DataFrame):
    """Gold macro-AUC on the gold studies present in this fold's predictions."""
    m = gold.merge(preds, on=ID_COL, how="inner", suffixes=("_t", "_p"))
    if len(m) == 0:
        return float("nan"), None
    yt = m[[f"{t}_t" for t in TARGETS]].values
    yp = m[[f"{t}_p" for t in TARGETS]].values
    return macro_auc(yt, yp), per_column_auc(yt, yp)


if __name__ == "__main__":
    main()
