"""Train ONE fold of the 2.5D adjacent-slice arm — `train_hr.py`'s recipe, new input only.

Identical to `src/rsna/train_hr.py` in EVERY modelling respect: per-slice
`tf_efficientnetv2_s` -> gated-attention MIL -> 12 sigmoid heads, soft-BCE on the `steven_v4`
blend with the same capped rare-column `pos_weight`, AdamW + warmup-cosine with the same
defaults, bf16 autocast + channels_last on sm_80+, per-epoch resumable checkpoint,
best-by-OOF-macro-AUC checkpoint + OOF parquet, `--device cpu` smoke path, `--limit-studies`.
One variable changes, and it is the input tensor:

    HR arm   slot j  ->  [gray, gray, gray]            (the same slice three times)
    2.5D arm slot j  ->  [prev, gray, next]            (its anatomical neighbours)

The slot sampler is untouched, so slot j holds the SAME InstanceUID as in the 224/384 arms —
this is a pure "spend the wasted channels" experiment, not a different bag.

CHECKPOINT SCHEMA — baseline keys (model, opt, sched, epoch, best_auc, gstep, cfg, args,
mean, std, fingerprint, targets, oof_auc) plus `args["size"]` (as `train_hr`) and
`args["mode"] == "2.5d"`, which is how `infer_25d.load_members_25d` (and the multi-arm
notebook, see `reports/arm_25d.md` §6) tells this arm apart from the plain HR arm. That flag
is LOAD-BEARING: a 2.5D checkpoint fed `gray x 3` is silently wrong, not merely worse.

SLICE ORDER — the neighbours must be anatomical, and the InstanceUID filename order is not
(0/400 train series match it; see `order.py`). Training reads the order from
`data/series_order.parquet`, built once:

    .venv/bin/python -m src.rsna.order build --root data/raw --split train \
        --out data/series_order.parquet --workers 3          # ~6 min, 3 workers

Fold 0 on the GB10 (same shape as the 384 HR arm: ~2.8 h/fold):

    .venv/bin/python -m src.rsna.train_25d --fold 0 --size 384 --k 24 --batch 6 \
        --grad-accum 3 --workers 6 --epochs 10 --lr 3e-4 --cache-root data/cache384 \
        --order-parquet data/series_order.parquet --name effv2s_k24_384_25d

CPU smoke (no GPU, a handful of studies):

    .venv/bin/python -m src.rsna.train_25d --device cpu --size 224 --k 8 --batch 2 \
        --epochs 1 --limit-studies 8 --limit-val 4 --workers 2 \
        --cache-root data/processed_jpeg224 --models-dir /tmp/mini/models \
        --oof-dir /tmp/mini/oof
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
from .dataset_25d import MODE_TAG, StudyBagDataset25D, build_order_resolver
from .dataset_hr import build_hr_index, default_cache_root, resolve_series_csv
from .metric import auc_report, load_gold, macro_auc, per_column_auc
from .model import ModelConfig, build_model
from .order import DEFAULT_ORDER_MODE, DEFAULT_ORDER_PARQUET, ORDER_MODES
# Reused verbatim from the HR trainer -- identical recipe means identical code, not a copy.
from .train_hr import (
    autocast_ctx,
    compute_fingerprint,
    default_paths,
    predict,
    resolve_device,
    warmup_cosine,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _gold_on_fold(preds: pd.DataFrame, gold: pd.DataFrame):
    """Gold macro-AUC on the gold studies present in this fold's predictions."""
    m = gold.merge(preds, on=ID_COL, how="inner", suffixes=("_t", "_p"))
    if len(m) == 0:
        return float("nan"), None
    yt = m[[f"{t}_t" for t in TARGETS]].values
    yp = m[[f"{t}_p" for t in TARGETS]].values
    return macro_auc(yt, yp), per_column_auc(yt, yp)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--name", type=str, default="effv2s_k24_384_25d")
    ap.add_argument("--backbone", type=str, default="tf_efficientnetv2_s")
    ap.add_argument("--size", type=int, default=384, help="input edge (224/384/512); in the ckpt")
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-pos-weight", type=float, default=3.0,
                    help="cap on per-column pos_weight (neg/pos), for rare-column upweighting")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="optimizer step every N batches (keeps the effective batch at 384px)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", type=str, default="cuda", help="cuda | cpu")
    ap.add_argument("--cache-root", type=str, default=None,
                    help="image root; default data/cache<size> (224 -> data/processed_jpeg224)")
    ap.add_argument("--series-csv", type=str, default=None,
                    help="default <cache-root>/train_series.csv, else data/comp_csv/train_series.csv")
    ap.add_argument("--index-cache", type=str, default=None,
                    help="pickle for the study index; default <cache-root>/study_index_train.pkl")
    ap.add_argument("--models-dir", type=str, default=None)
    ap.add_argument("--oof-dir", type=str, default=None)
    ap.add_argument("--pretrained", type=int, default=1, help="0 to skip the timm download (smoke)")
    ap.add_argument("--max-train-minutes", type=float, default=0.0,
                    help="if >0, stop after this many minutes of wall-clock (time-box)")
    ap.add_argument("--limit-studies", type=int, default=0,
                    help="smoke: cap #train studies (applied AFTER intersecting with the cache)")
    ap.add_argument("--limit-val", type=int, default=0, help="smoke: cap #val studies")
    # ---- 2.5D-specific ----
    ap.add_argument("--mode", type=str, default=MODE_TAG,
                    help="recorded in the ckpt so inference picks the 2.5D loader; do not change")
    ap.add_argument("--order-parquet", type=str, default=DEFAULT_ORDER_PARQUET,
                    help="series slice-order map (python -m src.rsna.order build); "
                         "'' disables it and falls back to --dicom-root / lexicographic")
    ap.add_argument("--dicom-root", type=str, default=None,
                    help="raw DICOM mirror (e.g. data/raw) to read slice order from when the "
                         "parquet has no entry for a series")
    ap.add_argument("--order-mode", type=str, default=DEFAULT_ORDER_MODE, choices=list(ORDER_MODES))
    ap.add_argument("--neighbour-step", type=int, default=1,
                    help="channels are (cur-step, cur, cur+step) in anatomical order")
    ap.add_argument("--require-order", action="store_true",
                    help="abort instead of silently falling back to lexicographic order")
    args = ap.parse_args()

    if args.mode != MODE_TAG:
        print(f"[warn] --mode {args.mode!r} != {MODE_TAG!r}: inference will not route this "
              f"checkpoint to the 2.5D loader")

    P = default_paths()
    models_dir = args.models_dir or P["models_dir"]
    oof_dir = args.oof_dir or P["oof_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(oof_dir, exist_ok=True)

    device = resolve_device(args.device)
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cache_root = args.cache_root or default_cache_root(args.size)
    series_csv = args.series_csv or resolve_series_csv(cache_root, "train")
    index_cache = args.index_cache or os.path.join(cache_root, "study_index_train.pkl")
    print(f"[cfg] device={device} size={args.size} k={args.k} batch={args.batch} "
          f"grad_accum={args.grad_accum} workers={args.workers} mode={args.mode} "
          f"neighbour_step={args.neighbour_step}")
    print(f"[cfg] cache_root={cache_root}")
    print(f"[cfg] series_csv={series_csv}")

    # ---- data ----
    labels = pd.read_csv(P["labels_csv"])
    folds = pd.read_csv(P["folds"])
    gold = load_gold(P["train_csv"])
    index = build_hr_index(cache_root, split="train", series_csv=series_csv,
                           cache_path=index_cache)
    n_slices = sum(len(p) for e in index.values() for _s, _pl, p in e)
    print(f"[index] studies={len(index)} series={sum(len(e) for e in index.values())} "
          f"slices={n_slices}")
    if not index:
        raise SystemExit(f"ERROR: empty study index under {cache_root} — build the cache first")

    # ---- slice order (the 2.5D correctness prerequisite) ----
    # Built HERE, in the parent process, so the compact rank vectors are shared copy-on-write
    # with every DataLoader worker instead of being re-loaded per worker.
    order = build_order_resolver(parquet=(args.order_parquet or None),
                                 dicom_root=args.dicom_root, split="train",
                                 mode=args.order_mode, allow_lex=not args.require_order)
    if not order.ranks_map and not args.dicom_root and args.order_mode != "lex":
        msg = ("no slice-order source: build data/series_order.parquet "
               "(python -m src.rsna.order build --root data/raw --split train) or pass "
               "--dicom-root data/raw. Neighbours would be RANDOM slices.")
        if args.require_order:
            raise SystemExit(f"ERROR: {msg}")
        print(f"[warn] {msg}")

    # Intersect with the cache BEFORE limiting, so a partial cache still yields a usable split.
    tr_ids = [s for s in folds[folds.fold != args.fold][ID_COL].astype(str) if s in index]
    va_ids = [s for s in folds[folds.fold == args.fold][ID_COL].astype(str) if s in index]
    if args.limit_studies:
        tr_ids = tr_ids[: args.limit_studies]
        va_ids = va_ids[: (args.limit_val or max(1, args.limit_studies // 4))]
    elif args.limit_val:
        va_ids = va_ids[: args.limit_val]
    if not tr_ids or not va_ids:
        raise SystemExit(f"ERROR: empty split (train={len(tr_ids)} val={len(va_ids)}) for "
                         f"fold {args.fold} under {cache_root}")

    cfg = ModelConfig(backbone=args.backbone, pretrained=bool(args.pretrained), dropout=args.dropout)
    model = build_model(cfg).to(device)
    if cuda:
        model = model.to(memory_format=torch.channels_last)
    dc = model.data_config()
    mean, std = dc["mean"], dc["std"]
    print(f"[data-config] mean={mean} std={std} feat_dim={model.feat_dim}")

    ds_kw = dict(k=args.k, size=args.size, mean=mean, std=std, order=order,
                 neighbour_step=args.neighbour_step)
    tr_ds = StudyBagDataset25D(tr_ids, index, labels, train=True, **ds_kw)
    va_ds = StudyBagDataset25D(va_ids, index, labels, train=False, **ds_kw)
    print(f"[fold {args.fold}] train={len(tr_ds)} val={len(va_ds)} studies @ {args.size}px "
          f"(2.5D: 3 slices decoded per slot)")
    # One-line proof, per run, that the centre channel still IS the HR arm's slot.
    t0_trip = tr_ds.slot_triples(0)[:3]
    print("[2.5d] slot (prev, cur, next) for train item 0:")
    for a, b, c in t0_trip:
        print(f"   {a[-10:]}  <-  {b[-10:]}  ->  {c[-10:]}")

    drop_last = len(tr_ds) >= 2 * args.batch  # keep tiny smoke splits from emptying the loader
    loader_kw = dict(num_workers=args.workers, pin_memory=cuda,
                     persistent_workers=args.workers > 0,
                     prefetch_factor=2 if args.workers > 0 else None)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=drop_last,
                           **loader_kw)
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False, **loader_kw)

    # ---- val truth (binarized steven for OOF-AUC; real 0/1 for gold) ----
    lab_idx = labels.set_index(ID_COL)[TARGETS]
    va_present = [s for s in va_ids if s in lab_idx.index]
    y_oof = (lab_idx.loc[va_present].values >= 0.5).astype(int)

    # ---- pos_weight (mild rare-column upweight) ----
    tr_present = [s for s in tr_ids if s in lab_idx.index]
    ytr = (lab_idx.loc[tr_present].values >= 0.5).astype(float)
    pos = ytr.sum(0)
    neg = len(ytr) - pos
    pw = np.clip(neg / np.maximum(pos, 1.0), 1.0, args.max_pos_weight)
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
    print("[pos_weight]", {t: round(float(w), 2) for t, w in zip(TARGETS, pw)})
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, math.ceil(len(tr_loader) / max(1, args.grad_accum)))
    total_steps = steps_per_epoch * args.epochs
    warmup = int(args.warmup_frac * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: warmup_cosine(s, total_steps, warmup))

    ckpt_last = os.path.join(models_dir, f"{args.name}_fold{args.fold}_last.pt")
    ckpt_best = os.path.join(models_dir, f"{args.name}_fold{args.fold}.pt")
    oof_path = os.path.join(oof_dir, f"{args.name}_fold{args.fold}.parquet")

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
        ep_t = time.time(); seen = 0; run_loss = 0.0; nb = 0
        opt.zero_grad(set_to_none=True)
        for it, (x, y, _sid) in enumerate(tr_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_ctx(device):
                logits = model(x)
                loss = criterion(logits, y)
            (loss / max(1, args.grad_accum)).backward()
            if (it + 1) % max(1, args.grad_accum) == 0 or (it + 1) == len(tr_loader):
                opt.step(); opt.zero_grad(set_to_none=True); sched.step(); gstep += 1
            seen += x.shape[0] * x.shape[1]; run_loss += float(loss.item()); nb += 1
            if it % 10 == 0:
                ips = seen / max(1e-6, time.time() - ep_t)
                print(f"  e{epoch} it{it}/{len(tr_loader)} loss={run_loss/nb:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {ips:.1f} img/s", flush=True)
        ep_time = time.time() - ep_t
        ips = seen / max(1e-6, ep_time)

        # ---- validate ----
        preds = predict(model, va_loader, device)
        got = set(preds[ID_COL].astype(str))
        order_ids = [s for s in va_present if s in got]        # labelled AND predicted
        preds = preds.set_index(ID_COL).loc[order_ids].reset_index()
        y_val = y_oof[[va_present.index(s) for s in order_ids]] if order_ids else y_oof[:0]
        oof_auc = macro_auc(y_val, preds[TARGETS].values) if len(order_ids) > 1 else float("nan")
        g_auc, _g_rep = _gold_on_fold(preds, gold)
        print(f"[epoch {epoch}] loss={run_loss/max(1,nb):.4f} OOF_macroAUC={oof_auc:.4f} "
              f"gold_foldAUC={g_auc:.4f} epoch_time={ep_time:.0f}s {ips:.1f} img/s", flush=True)
        if len(order_ids) > 1:
            rep = auc_report(y_val, preds[TARGETS].values)
            print("  OOF per-finding:", {t: round(float(rep[t]), 3) for t in TARGETS})
        print(" ", order.report(), flush=True)

        # ---- checkpoint (resumable; baseline schema + args['size'] + args['mode']) ----
        fp = compute_fingerprint(model, device, k=args.k, size=args.size)
        state = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                     epoch=epoch, best_auc=best_auc, gstep=gstep, cfg=vars(cfg),
                     args=vars(args), mean=list(map(float, mean)), std=list(map(float, std)),
                     fingerprint=fp, targets=TARGETS, oof_auc=oof_auc)
        torch.save(state, ckpt_last)
        # NaN-safe: a degenerate val split (smoke, single class) yields NaN AUC — still emit a
        # best checkpoint so the artifact exists; real runs behave exactly like the baseline.
        improved = (not math.isnan(oof_auc) and oof_auc > best_auc) or not os.path.exists(ckpt_best)
        if improved:
            if not math.isnan(oof_auc):
                best_auc = oof_auc
            state["best_auc"] = best_auc
            torch.save(state, ckpt_best)
            preds.to_parquet(oof_path, index=False)
            print(f"  * saved best -> {ckpt_best} (OOF={oof_auc:.4f}) + {oof_path}", flush=True)

        if args.max_train_minutes and (time.time() - t0) / 60.0 >= args.max_train_minutes:
            print(f"[time-box] stopping after {(time.time()-t0)/60:.1f} min", flush=True)
            break

    print(f"[done] fold {args.fold} @{args.size}px 2.5D best OOF_macroAUC={best_auc:.4f} "
          f"total {(time.time()-t0)/60:.1f} min")
    return best_auc


if __name__ == "__main__":
    main()
