"""Train ONE fold of the Wave-3 "A3" PLANE-SPECIALISED arm.

Same recipe as `src/rsna/train.py` / `train_hr.py` — per-slice `tf_efficientnetv2_s` ->
gated-attention MIL -> 12 sigmoid heads, soft-BCE on the `steven_v4_blend` labels with a mild
rare-column pos_weight, AdamW + warmup-cosine, bf16 autocast on sm_80+, channels_last on
CUDA, resumable per-epoch checkpoint, best-by-OOF-macro-AUC checkpoint + OOF parquet — with
exactly ONE experimental variable changed:

    the study bag is split into THREE per-plane bags (Sagittal / Coronal / Axial), each with
    its own attention pool, and the 12 findings learn a softmax over the 3 plane embeddings.

With the default `--k-per-plane 8 8 8` the per-study slice budget is 24, i.e. IDENTICAL to
the 224 baseline (`effv2s_k24`, fold-0 OOF 0.838) -> a fold costs the same and the OOF delta
is attributable to plane-aware pooling alone.

Checkpoint schema = the baseline's keys (model, opt, sched, epoch, best_auc, gstep, cfg,
args, mean, std, fingerprint, targets, oof_auc) PLUS:
    args["size"]          input edge, so `infer_plane` decodes at the right resolution
    args["plane_mode"]    the fusion mode ("planewise" | "concat")
    args["k_per_plane"]   the per-plane slot allocation (args["k"] = its sum)
    plane_weights         [12, 3] learned per-finding plane weights (rows sum to 1)
    plane_order           ["Sagittal", "Coronal", "Axial"]

Reproduce (GB10, 224 first — clean attribution vs the 224 baseline):
    .venv/bin/python -m src.rsna.train_plane --fold 0 --size 224 --k-per-plane 8 8 8 \
        --batch 12 --workers 8 --epochs 10 --lr 3e-4 --name effv2s_plane_k8x3_224

CPU smoke (no GPU, a handful of studies):
    .venv/bin/python -m src.rsna.train_plane --device cpu --size 224 --k-per-plane 4 4 4 \
        --batch 2 --epochs 1 --limit-studies 8 --limit-val 4 --pretrained 0 \
        --models-dir /tmp/plane/models --oof-dir /tmp/plane/oof \
        --index-cache /tmp/plane/idx224.pkl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import TARGETS, ID_COL
from .dataset_plane import (
    DEFAULT_K_PER_PLANE,
    PLANE_ORDER,
    StudyPlaneBagDataset,
    build_plane_index,
    default_cache_root,
    plane_coverage,
    resolve_series_csv,
)
from .metric import auc_report, load_gold, macro_auc, per_column_auc
from .model_plane import PlaneModelConfig, build_plane_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_paths() -> Dict[str, str]:
    return dict(
        labels_csv=os.path.join(
            ROOT, "data", "labels", "stevenleehans_rsna-knee-llm-report-labels",
            "llm_labels_v4_blend.csv",
        ),
        folds=os.path.join(ROOT, "data", "folds.csv"),
        train_csv=os.path.join(ROOT, "data", "comp_csv", "train.csv"),
        models_dir=os.path.join(ROOT, "models"),
        oof_dir=os.path.join(ROOT, "data", "oof"),
    )


# --------------------------------------------------------------------------------------
# device / precision helpers (identical rules to train_hr.py)
# --------------------------------------------------------------------------------------
def resolve_device(want: str) -> str:
    if str(want).startswith("cuda") and not torch.cuda.is_available():
        print("[device] cuda requested but unavailable -> cpu")
        return "cpu"
    return want


def autocast_ctx(device: str):
    """bf16 autocast on Ampere+ (sm_80+); plain fp32 on older GPUs and on CPU."""
    if str(device).startswith("cuda"):
        try:
            major, _minor = torch.cuda.get_device_capability(0)
        except Exception:
            major = 0
        if major >= 8:
            return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def predict(model, loader, device: str) -> pd.DataFrame:
    model.eval()
    ids, probs = [], []
    for x, _y, mask, sid in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with autocast_ctx(device):
            logits = model(x, plane_mask=mask)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        ids.extend(list(sid))
    probs = np.concatenate(probs, axis=0) if probs else np.zeros((0, len(TARGETS)), np.float32)
    df = pd.DataFrame(probs, columns=TARGETS)
    df.insert(0, ID_COL, ids)
    return df


def compute_fingerprint(model, device: str, k_total: int, size: int) -> np.ndarray:
    """Deterministic output on a fixed synthetic [1,k_total,3,size,size] input (all planes present).

    True fp32 (TF32 disabled), NO autocast -> portable across GPUs/CPU, so a wrong-weights or
    wrong-preprocessing load is caught at inference time.
    """
    model.eval()
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        g = torch.Generator().manual_seed(1234)
        x = torch.randn(1, k_total, 3, size, size, generator=g).to(device)
        m = torch.ones(1, len(PLANE_ORDER), dtype=torch.bool, device=device)
        with torch.no_grad():
            out = torch.sigmoid(model(x, plane_mask=m).float()).cpu().numpy().ravel()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cudnn
    return out


def _gold_on_fold(preds: pd.DataFrame, gold: pd.DataFrame):
    """Gold macro-AUC on the gold studies present in this fold's predictions."""
    m = gold.merge(preds, on=ID_COL, how="inner", suffixes=("_t", "_p"))
    if len(m) == 0:
        return float("nan"), None
    yt = m[[f"{t}_t" for t in TARGETS]].values
    yp = m[[f"{t}_p" for t in TARGETS]].values
    return macro_auc(yt, yp), per_column_auc(yt, yp)


def _fmt_plane_table(report: Dict[str, Dict[str, float]]) -> str:
    """One-line-per-finding view of the learned routing (argmax plane flagged)."""
    lines = []
    for t, w in report.items():
        best = max(w, key=w.get)
        cells = "  ".join(f"{p[:3]}={w[p]:.3f}" for p in PLANE_ORDER)
        lines.append(f"    {t:<17s} {cells}   -> {best}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--name", type=str, default="effv2s_plane_k8x3_224")
    ap.add_argument("--backbone", type=str, default="tf_efficientnetv2_s")
    ap.add_argument("--size", type=int, default=224, help="input edge (224/384/512); recorded in the ckpt")
    ap.add_argument("--k-per-plane", type=int, nargs=3, default=list(DEFAULT_K_PER_PLANE),
                    metavar=("SAG", "COR", "AX"),
                    help="slots per plane; the default 8 8 8 = the baseline's 24-slice budget")
    ap.add_argument("--fusion", type=str, default="planewise", choices=["planewise", "concat"],
                    help="planewise = per-finding softmax over the 3 plane embeddings; "
                         "concat = Linear(3D,12) control")
    ap.add_argument("--plane-temp", type=float, default=1.0,
                    help="softmax temperature on the plane logits (>1 = softer routing)")
    ap.add_argument("--attn-dim", type=int, default=256)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--plane-lr-mult", type=float, default=10.0,
                    help="LR multiplier for the 12x3 plane logits (they are a tiny, slow-moving "
                         "parameter group; 1.0 to disable)")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-pos-weight", type=float, default=3.0,
                    help="cap on per-column pos_weight (neg/pos), for rare-column upweighting")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", type=str, default="cuda", help="cuda | cpu")
    ap.add_argument("--cache-root", type=str, default=None,
                    help="image root; default data/cache<size> (224 -> data/processed_jpeg224)")
    ap.add_argument("--series-csv", type=str, default=None,
                    help="default <cache-root>/train_series.csv, else data/comp_csv/train_series.csv")
    ap.add_argument("--index-cache", type=str, default=None,
                    help="pickle for the study index; default <cache-root>/study_index_train.pkl "
                         "(identical content to the one train_hr.py caches -- safe to share)")
    ap.add_argument("--models-dir", type=str, default=None)
    ap.add_argument("--oof-dir", type=str, default=None)
    ap.add_argument("--pretrained", type=int, default=1, help="0 to skip the timm download (smoke)")
    ap.add_argument("--max-train-minutes", type=float, default=0.0,
                    help="if >0, stop after this many minutes of wall-clock (time-box)")
    ap.add_argument("--limit-studies", type=int, default=0,
                    help="smoke: cap #train studies (applied AFTER intersecting with the cache)")
    ap.add_argument("--limit-val", type=int, default=0, help="smoke: cap #val studies")
    ap.add_argument("--plane-coverage", action="store_true",
                    help="print the corpus per-plane slice/missing statistics and exit")
    args = ap.parse_args()

    # Derived/recorded fields (checkpoint schema: baseline keys + size + plane_mode).
    args.k_per_plane = [int(v) for v in args.k_per_plane]
    args.k = int(sum(args.k_per_plane))          # total slices per study (baseline-comparable)
    args.plane_mode = args.fusion                 # recorded name for the fusion topology
    args.plane_order = list(PLANE_ORDER)

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
    print(f"[cfg] device={device} size={args.size} k_per_plane={args.k_per_plane} "
          f"(k={args.k}) fusion={args.fusion} batch={args.batch} "
          f"grad_accum={args.grad_accum} workers={args.workers}")
    print(f"[cfg] cache_root={cache_root}")
    print(f"[cfg] series_csv={series_csv}")

    # ---- data ----
    labels = pd.read_csv(P["labels_csv"])
    folds = pd.read_csv(P["folds"])
    gold = load_gold(P["train_csv"])
    index = build_plane_index(cache_root, split="train", series_csv=series_csv,
                              cache_path=index_cache)
    n_slices = sum(len(p) for e in index.values() for _s, _pl, p in e)
    print(f"[index] studies={len(index)} series={sum(len(e) for e in index.values())} "
          f"slices={n_slices}")
    if not index:
        raise SystemExit(f"ERROR: empty study index under {cache_root} — build the cache first")

    cov = plane_coverage(index)
    print("[planes] missing-plane studies: " + ", ".join(
        f"{p}={cov['missing_counts'][p]} ({100*cov['missing_frac'][p]:.1f}%)" for p in PLANE_ORDER))
    print("[planes] slices/plane (median): " + ", ".join(
        f"{p}={cov['slices_per_plane'][p]['median']:.0f}" for p in PLANE_ORDER))
    if args.plane_coverage:
        print(json.dumps(cov, indent=2))
        return 0.0

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

    cfg = PlaneModelConfig(backbone=args.backbone, pretrained=bool(args.pretrained),
                           attn_dim=args.attn_dim, dropout=args.dropout,
                           k_per_plane=tuple(args.k_per_plane), fusion=args.fusion,
                           plane_temp=args.plane_temp)
    model = build_plane_model(cfg).to(device)
    if cuda:
        model = model.to(memory_format=torch.channels_last)
    dc = model.data_config()
    mean, std = dc["mean"], dc["std"]
    print(f"[data-config] mean={mean} std={std} feat_dim={model.feat_dim}")

    tr_ds = StudyPlaneBagDataset(tr_ids, index, labels, k_per_plane=args.k_per_plane, train=True,
                                 size=args.size, mean=mean, std=std)
    va_ds = StudyPlaneBagDataset(va_ids, index, labels, k_per_plane=args.k_per_plane, train=False,
                                 size=args.size, mean=mean, std=std)
    print(f"[fold {args.fold}] train={len(tr_ds)} val={len(va_ds)} studies @ {args.size}px "
          f"({args.k} slices/study)")

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

    # The 12x3 routing table is a tiny parameter group fed by a softmax; at the backbone LR it
    # barely moves inside 10 epochs. Give it its own (higher) LR and no weight decay, so the
    # experiment actually tests plane routing rather than a frozen uniform average.
    plane_params = [p for n, p in model.named_parameters() if n == "plane_logits"]
    other_params = [p for n, p in model.named_parameters() if n != "plane_logits"]
    groups = [dict(params=other_params, lr=args.lr, weight_decay=args.wd)]
    if plane_params:
        groups.append(dict(params=plane_params, lr=args.lr * args.plane_lr_mult, weight_decay=0.0))
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)
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
        for it, (x, y, mask, _sid) in enumerate(tr_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with autocast_ctx(device):
                logits = model(x, plane_mask=mask)
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
        order = [s for s in va_present if s in got]            # labelled AND predicted
        preds = preds.set_index(ID_COL).loc[order].reset_index()
        y_val = y_oof[[va_present.index(s) for s in order]] if order else y_oof[:0]
        oof_auc = macro_auc(y_val, preds[TARGETS].values) if len(order) > 1 else float("nan")
        g_auc, _g_rep = _gold_on_fold(preds, gold)
        print(f"[epoch {epoch}] loss={run_loss/max(1,nb):.4f} OOF_macroAUC={oof_auc:.4f} "
              f"gold_foldAUC={g_auc:.4f} epoch_time={ep_time:.0f}s {ips:.1f} img/s", flush=True)
        if len(order) > 1:
            rep = auc_report(y_val, preds[TARGETS].values)
            print("  OOF per-finding:", {t: round(float(rep[t]), 3) for t in TARGETS})
        pw_report = model.plane_weight_report()
        print("  learned plane weights (per finding):")
        print(_fmt_plane_table(pw_report), flush=True)

        # ---- checkpoint (resumable; baseline schema + size/plane_mode/plane weights) ----
        fp = compute_fingerprint(model, device, k_total=args.k, size=args.size)
        state = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                     epoch=epoch, best_auc=best_auc, gstep=gstep, cfg=vars(cfg),
                     args=vars(args), mean=list(map(float, mean)), std=list(map(float, std)),
                     fingerprint=fp, targets=TARGETS, oof_auc=oof_auc,
                     plane_weights=model.plane_weight_table(), plane_order=list(PLANE_ORDER))
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

    print(f"[done] fold {args.fold} @{args.size}px plane={args.fusion} "
          f"best OOF_macroAUC={best_auc:.4f} total {(time.time()-t0)/60:.1f} min")
    return best_auc


if __name__ == "__main__":
    main()
