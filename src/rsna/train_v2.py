"""Train ONE fold of the Wave-4 "v2" arm — the input contract of the 0.92+ public solutions.

Same CLI shape, fold/OOF/checkpoint conventions, time-box and resume behaviour as
`train_hr.py`; everything that differs is a deliberate adoption from
`reports/public_solutions.md`:

  input     `dataset_v2.StudyBagDatasetV2` — 5 plane x fluid SLOTS, 64 slices, evenly spaced
            in anatomical rank over span 0.02-0.98, one chosen series per slot, laterality
            canonicalised to a right knee (§2c, §2d, §4 #2/#3). `--k-scale` shrinks the bag
            for training only; eval and inference always use the full 64.
  model     `model_v2.StudyModelV2` — any timm backbone (default the CoAtNet-384 that scores
            0.924 as a single public model), per-finding gated attention with the fixed
            anatomical slot prior (§2d.1), 12 linear heads.
  labels    `data/labels/consensus_v1.parquet` — soft targets + a PER-CELL weight (§2b),
            gold rows weight 3.0. Folds come from that file's own `fold` column, so the
            gold studies are held out one fold at a time instead of being validation-only.
  loss      BCE-with-logits on the soft targets x the per-cell weight, mean over cells. NO
            pos_weight by default (the confidence weights already do that job);
            `--pos-weight` restores the old rare-column scheme for an A/B.
  optim     layer-wise LR (backbone = lr x `--lr-backbone-mult`, default 0.1 — the public
            arms adapt a pretrained encoder rather than retraining it, §2a), AdamW,
            warmup-cosine, bf16 autocast on sm_80+, channels_last for CNNs, EMA 0.999 with
            timm's decay warmup. BOTH the EMA and the raw weights are evaluated every epoch
            and the better of the two is what gets checkpointed and written as the OOF.

Checkpoint schema is `train_hr`'s (model/opt/sched/epoch/best_auc/gstep/cfg/args/mean/std/
fingerprint/targets/oof_auc) plus `model_source` ("ema"|"raw" — which weights are in `model`),
`ema_updates`, and `args["mode"]="v2"`, `slots`, `crop_mm`, `pct`, `k_scale`, `slot_prior`,
`k` — so `infer_v2.is_v2_meta` routes it and `dev/slim_ckpt.py` slims it.

Reproduce (GPU, fold 0 — also in `dev/kaggle/next_v2_cmd`):
    .venv/bin/python -m src.rsna.train_v2 --fold 0 --size 384 \
        --backbone coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --batch 2 --grad-accum 8 \
        --k-scale 0.75 --workers 8 --epochs 10 --lr 2e-4 --lr-backbone-mult 0.1 \
        --cache-root data/cache_c140_384 --labels data/labels/consensus_v1.parquet \
        --side-parquet data/study_side.parquet --order-parquet data/series_order.parquet \
        --name v2_coat384 --max-train-minutes 300 --grad-ckpt

CPU smoke (no GPU, tiny):
    .venv/bin/python -m src.rsna.train_v2 --device cpu --backbone tf_efficientnetv2_s \
        --size 224 --k-scale 0.25 --batch 1 --epochs 1 --limit-studies 4 --workers 0 \
        --cache-root data/cache384 --models-dir /tmp/v2/models --oof-dir /tmp/v2/oof
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import TARGETS, ID_COL
from .dataset_v2 import (
    V2_CROP_MM,
    V2_JITTER,
    V2_PCT,
    MODE_TAG_V2,
    SLOTS,
    StudyBagDatasetV2,
    build_index_v2,
    default_labels_paths,
    fallback_report,
    load_labels_v2,
    load_side_map,
    scaled_slot_counts,
    slot_total,
)
from .metric import auc_report, load_gold, macro_auc
from .model_v2 import ModelV2Config, build_model_v2, compute_fingerprint_v2
from .order import DEFAULT_ORDER_PARQUET, SliceOrderResolver
from .train_hr import autocast_ctx, resolve_device, warmup_cosine  # reused verbatim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# --------------------------------------------------------------------------------------
# EMA
# --------------------------------------------------------------------------------------
class ModelEma:
    """Exponential moving average of the weights; what gets evaluated and checkpointed.

    `romantamrazov/...v4-train` uses EMA 0.995 on the head-only arm; 0.999 is the usual
    choice for a full fine-tune at this step count. `--ema 0` disables it and the raw
    weights are used everywhere instead.

    DECAY WARMUP (timm `ModelEmaV3`; on by default, `--ema-warmup 0` restores the old flat
    decay). The average starts as a copy of the RANDOM init, so a constant 0.999 keeps
    0.999^t of that init alive: 41 % after one 881-step epoch. Measured on fold 0, epoch 0 —
    EMA 0.463 macro-AUC vs 0.761 for the raw weights on the same 48 studies. The warmup
    decay `min(decay, (1 + t) / (10 + t))` is 0.10 at the first step and only approaches
    0.999 around t ~ 10k, so the average tracks the live weights while they still move fast
    and is a genuine average by the time they settle.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, warmup: bool = True):
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.updates = 0            # t: how many update() calls have happened
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def effective_decay(self, updates: Optional[int] = None) -> float:
        """Decay the NEXT `update()` will apply (t = updates done so far)."""
        t = self.updates if updates is None else int(updates)
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + t) / (10.0 + t))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.effective_decay()
        for ev, v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ev.dtype.is_floating_point:
                ev.mul_(d).add_(v.detach().to(ev.device), alpha=1.0 - d)
            else:
                ev.copy_(v)
        self.updates += 1

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)


@torch.no_grad()
def predict(model, loader, device: str) -> pd.DataFrame:
    model.eval()
    ids, probs = [], []
    for x, sid_t, pos, valid, _y, _w, sid in loader:
        x = x.to(device, non_blocking=True)
        with autocast_ctx(device):
            logits = model(x, sid_t.to(device), pos.to(device), valid.to(device))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        ids.extend(list(sid))
    arr = np.concatenate(probs, axis=0) if probs else np.zeros((0, len(TARGETS)), np.float32)
    df = pd.DataFrame(arr.astype(np.float32), columns=TARGETS)
    df.insert(0, ID_COL, ids)
    return df


def _gold_on_fold(preds: pd.DataFrame, gold: pd.DataFrame):
    m = gold.merge(preds, on=ID_COL, how="inner", suffixes=("_t", "_p"))
    if len(m) == 0:
        return float("nan"), 0
    yt = m[[f"{t}_t" for t in TARGETS]].values
    yp = m[[f"{t}_p" for t in TARGETS]].values
    return macro_auc(yt, yp), len(m)


def _evaluate(model, loader, device: str, va_present, y_oof, gold) -> dict:
    """One full val pass for ONE weight set: preds + OOF macro-AUC + gold-in-fold AUC."""
    preds = predict(model, loader, device)
    got = set(preds[ID_COL].astype(str))
    order = [s for s in va_present if s in got]
    oof_auc, y_val, pv = float("nan"), None, None
    if len(order) > 1:
        pv = preds.set_index(ID_COL).loc[order][TARGETS].values
        y_val = y_oof[[va_present.index(s) for s in order]]
        oof_auc = macro_auc(y_val, pv)
    g_auc, n_g = _gold_on_fold(preds, gold)
    return dict(preds=preds, oof=oof_auc, gold=g_auc, n_gold=n_g, y=y_val, p=pv)


def _pick_source(res: dict) -> str:
    """Which of the evaluated weight sets to checkpoint: better OOF (nan last, EMA on ties)."""
    def key(kv):
        auc = kv[1]["oof"]
        return (-1e9 if math.isnan(auc) else auc, kv[0] == "ema")
    return max(res.items(), key=key)[0]


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # --- the train_hr CLI, unchanged in shape -------------------------------------------
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--name", type=str, default="v2_coat384")
    ap.add_argument("--size", type=int, default=384, help="input edge; 384 cache, <384 resized")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--cache-root", type=str, default=os.path.join(ROOT, "data", "cache_c140_384"))
    ap.add_argument("--series-csv", type=str, default=None)
    ap.add_argument("--index-cache", type=str, default=None)
    ap.add_argument("--models-dir", type=str, default=os.path.join(ROOT, "models"))
    ap.add_argument("--oof-dir", type=str, default=os.path.join(ROOT, "data", "oof"))
    ap.add_argument("--device", type=str, default="cuda", help="cuda | cpu")
    ap.add_argument("--cpu-threads", type=int, default=0,
                    help="torch.set_num_threads on the CPU path (0 = torch's default). Torch's "
                         "native thread pool does NOT read OMP_NUM_THREADS, so a CPU smoke on a "
                         "loaded box needs this to stop 6 idle workers thrashing on futexes.")
    ap.add_argument("--pretrained", type=int, default=1, help="0 to skip the timm download (smoke)")
    ap.add_argument("--max-train-minutes", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit-studies", type=int, default=0, help="smoke: cap #train studies")
    ap.add_argument("--limit-val", type=int, default=0, help="smoke: cap #val studies")
    # --- v2 additions --------------------------------------------------------------------
    ap.add_argument("--backbone", type=str, default="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    ap.add_argument("--k-scale", type=float, default=1.0,
                    help="train-time scale on every slot count (0.75 -> 14/11/9/6/9 = 49); "
                         "eval and inference always use the full 64")
    ap.add_argument("--labels", type=str, default=default_labels_paths()["consensus"],
                    help="consensus_v1.parquet; falls back to a 1.0-weight stub if absent")
    ap.add_argument("--side-parquet", type=str, default=os.path.join(ROOT, "data", "study_side.parquet"))
    ap.add_argument("--order-parquet", type=str, default=DEFAULT_ORDER_PARQUET)
    ap.add_argument("--dicom-root", type=str, default=None,
                    help="raw DICOM mirror, for series whose order the parquet is missing")
    ap.add_argument("--slot-prior", type=float, default=0.55, help="0 disables the prior bias")
    ap.add_argument("--learn-slot-bias", type=int, default=0,
                    help="1 = also learn a per-finding x slot bias on top of the fixed prior")
    ap.add_argument("--grad-ckpt", action="store_true", help="checkpoint the backbone per chunk")
    ap.add_argument("--slice-chunk", type=int, default=0,
                    help="forward the B*K slices through the backbone this many at a time")
    ap.add_argument("--lr-backbone-mult", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999, help="0 disables EMA")
    ap.add_argument("--ema-warmup", type=int, default=1,
                    help="1 = timm ModelEmaV3 decay warmup min(decay,(1+t)/(10+t)); 0 = the old "
                         "flat decay, which leaves 41%% of the random init in the average after "
                         "one 881-step epoch")
    ap.add_argument("--eval-raw", type=int, default=1,
                    help="1 = also evaluate the RAW weights each epoch (a second pass over the "
                         "val loader) and checkpoint whichever of ema/raw scores better")
    ap.add_argument("--jitter", type=float, default=V2_JITTER,
                    help="train-time position jitter in slot spacings (1.0 = +/- 0.5 slot)")
    ap.add_argument("--pos-weight", action="store_true",
                    help="re-enable the old rare-column pos_weight scheme (off by default)")
    ap.add_argument("--max-pos-weight", type=float, default=3.0)
    ap.add_argument("--crop-mm", type=float, default=V2_CROP_MM,
                    help="recorded in the ckpt; the DICOM decoder at inference reads it")
    ap.add_argument("--pct", type=float, nargs=2, default=list(V2_PCT),
                    help="per-series normalisation percentiles, recorded in the ckpt")
    args = ap.parse_args()

    # metadata that must live in the checkpoint's `args` (routing + inference contract)
    # `--grad-ckpt` with ONE chunk saves nothing: checkpointing a single call over all B*K
    # slices stores no intermediates during the forward, but the backward recomputes the whole
    # segment at once, so the peak is the un-checkpointed peak. The chunking is what bounds it.
    if args.grad_ckpt and int(args.slice_chunk) <= 0:
        args.slice_chunk = 16
        print("[cfg] --grad-ckpt given without --slice-chunk -> using --slice-chunk 16 "
              "(a single chunk would checkpoint the whole bag and bound nothing)")

    args.mode = MODE_TAG_V2
    args.slots = [list(s) for s in SLOTS]
    args.slot_counts = list(scaled_slot_counts(args.k_scale))
    args.k = slot_total(args.k_scale)
    args.k_eval = slot_total(1.0)

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.oof_dir, exist_ok=True)
    device = resolve_device(args.device)
    cuda = str(device).startswith("cuda")
    if args.cpu_threads and not cuda:
        torch.set_num_threads(int(args.cpu_threads))
        print(f"[cfg] torch.set_num_threads({args.cpu_threads})")
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"[cfg] mode=v2 device={device} size={args.size} backbone={args.backbone}")
    print(f"[cfg] k_scale={args.k_scale} slots={args.slot_counts} K={args.k} (eval K={args.k_eval}) "
          f"batch={args.batch} grad_accum={args.grad_accum} workers={args.workers}")
    print(f"[cfg] cache_root={args.cache_root} crop_mm={args.crop_mm} pct={tuple(args.pct)}")

    # ---- data ------------------------------------------------------------------------
    labels = load_labels_v2(args.labels)
    gold = load_gold(default_labels_paths()["train_csv"])
    steven = pd.read_csv(default_labels_paths()["steven"]).set_index(ID_COL)[TARGETS]
    steven.index = steven.index.astype(str)
    side_map = load_side_map(args.side_parquet)

    # Built in the MAIN process so the compact rank vectors are shared copy-on-write.
    resolver = SliceOrderResolver.from_parquet(args.order_parquet, dicom_root=args.dicom_root,
                                               split="train")
    # A smoke asks for a handful of studies; indexing all 4,407 (and pickling ~190 MB of
    # paths) would dwarf the thing being smoked, so restrict the build to a superset of what
    # the run can possibly use. A filtered index is never cached.
    only = None
    if args.limit_studies:
        fo = labels["fold"].astype(int)
        take = max(4, args.limit_studies * 4)
        only = (list(fo.index[fo != args.fold].astype(str)[:take])
                + list(fo.index[fo == args.fold].astype(str)[:take]))
    index = build_index_v2(args.cache_root, split="train", series_csv=args.series_csv,
                           order=resolver, cache_path=args.index_cache, only=only)
    if not index:
        raise SystemExit(f"ERROR: empty v2 index under {args.cache_root} — build the cache first")

    fold_of = labels["fold"].astype(int)
    tr_ids = [s for s in fold_of.index[fold_of != args.fold].astype(str) if s in index]
    va_ids = [s for s in fold_of.index[fold_of == args.fold].astype(str) if s in index]
    if args.limit_studies:
        tr_ids = tr_ids[: args.limit_studies]
        va_ids = va_ids[: (args.limit_val or max(1, args.limit_studies // 2))]
    elif args.limit_val:
        va_ids = va_ids[: args.limit_val]
    if not tr_ids or not va_ids:
        raise SystemExit(f"ERROR: empty split (train={len(tr_ids)} val={len(va_ids)}) for fold "
                         f"{args.fold} under {args.cache_root}")

    cfg = ModelV2Config(backbone=args.backbone, pretrained=bool(args.pretrained), size=args.size,
                        dropout=args.dropout, slot_prior=args.slot_prior,
                        learn_slot_bias=bool(args.learn_slot_bias), grad_ckpt=bool(args.grad_ckpt),
                        slice_chunk=int(args.slice_chunk))
    model = build_model_v2(cfg).to(device)
    if cuda and model._channels_last:
        model = model.to(memory_format=torch.channels_last)
    dc = model.data_config()
    mean, std = dc["mean"], dc["std"]
    print(f"[data-config] mean={mean} std={std} feat_dim={model.feat_dim}")

    tr_ds = StudyBagDatasetV2(tr_ids, index, labels, size=args.size, train=True, mean=mean,
                              std=std, k_scale=args.k_scale, side_map=side_map,
                              jitter=args.jitter)
    va_ds = StudyBagDatasetV2(va_ids, index, labels, size=args.size, train=False, mean=mean,
                              std=std, k_scale=1.0, side_map=side_map)   # eval = full 64
    n_gold_val = int(labels.loc[[s for s in va_ids if s in labels.index], "is_gold"].sum())
    print(f"[fold {args.fold}] train={len(tr_ds)} (K={tr_ds.k}) val={len(va_ds)} (K={va_ds.k}) "
          f"studies @ {args.size}px, gold-in-fold={n_gold_val}")

    drop_last = len(tr_ds) >= 2 * args.batch
    loader_kw = dict(num_workers=args.workers, pin_memory=cuda,
                     persistent_workers=args.workers > 0,
                     prefetch_factor=2 if args.workers > 0 else None)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=drop_last,
                           **loader_kw)
    va_loader = DataLoader(va_ds, batch_size=max(1, args.batch), shuffle=False, **loader_kw)

    # ---- OOF truth: binarised steven blend (the ruler train_hr uses) -------------------
    va_present = [s for s in va_ids if s in steven.index]
    y_oof = (steven.loc[va_present].values >= 0.5).astype(int)

    # ---- loss -------------------------------------------------------------------------
    pos_weight = None
    if args.pos_weight:
        tr_present = [s for s in tr_ids if s in steven.index]
        ytr = (steven.loc[tr_present].values >= 0.5).astype(float)
        pos = ytr.sum(0)
        pw = np.clip((len(ytr) - pos) / np.maximum(pos, 1.0), 1.0, args.max_pos_weight)
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
        print("[pos_weight]", {t: round(float(v), 2) for t, v in zip(TARGETS, pw)})
    else:
        print("[loss] weighted soft-BCE, mean over cells, NO pos_weight "
              "(the per-cell confidence weights carry the imbalance)")

    def criterion(logits, y, w):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, weight=w, pos_weight=pos_weight, reduction="mean")
        return loss

    # ---- optimiser --------------------------------------------------------------------
    opt = torch.optim.AdamW(model.param_groups(args.lr, args.lr_backbone_mult),
                            lr=args.lr, weight_decay=args.wd)
    print(f"[optim] AdamW lr={args.lr:.2e} backbone_lr={args.lr * args.lr_backbone_mult:.2e} "
          f"wd={args.wd} ema={args.ema}")
    steps_per_epoch = max(1, math.ceil(len(tr_loader) / max(1, args.grad_accum)))
    total_steps = steps_per_epoch * args.epochs
    warmup = int(args.warmup_frac * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: warmup_cosine(s, total_steps, warmup))
    ema: Optional[ModelEma] = (ModelEma(model, args.ema, warmup=bool(args.ema_warmup))
                               if args.ema and args.ema > 0 else None)
    if ema is not None:
        print(f"[ema] decay={args.ema} warmup={'on' if args.ema_warmup else 'off'} "
              f"effective d(t=0)={ema.effective_decay(0):.4f} d(100)={ema.effective_decay(100):.4f} "
              f"d(1000)={ema.effective_decay(1000):.4f} | eval_raw={bool(args.eval_raw)}")

    ckpt_last = os.path.join(args.models_dir, f"{args.name}_fold{args.fold}_last.pt")
    ckpt_best = os.path.join(args.models_dir, f"{args.name}_fold{args.fold}.pt")
    oof_path = os.path.join(args.oof_dir, f"{args.name}_fold{args.fold}.parquet")

    start_epoch, best_auc, gstep = 0, -1.0, 0
    if args.resume and os.path.exists(ckpt_last):
        st = torch.load(ckpt_last, map_location=device, weights_only=False)
        # `model` holds the EMA (inference) weights; `raw_model` holds the live ones. Resume
        # from the live weights when they are there, so training continues where it stopped.
        model.load_state_dict(st.get("raw_model") or st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); start_epoch = st["epoch"] + 1
        best_auc = st.get("best_auc", -1.0); gstep = st.get("gstep", 0)
        if ema is not None and st.get("ema") is not None:
            ema.load_state_dict(st["ema"])
            # pre-warmup checkpoints have no counter; #optimizer steps == #ema updates.
            ema.updates = int(st.get("ema_updates", gstep))
        print(f"[resume] from epoch {start_epoch}, best_auc={best_auc:.4f}")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_t = time.time(); seen = 0; run_loss = 0.0; nb = 0
        opt.zero_grad(set_to_none=True)
        for it, (x, sid_t, pos_t, valid, y, w, _s) in enumerate(tr_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            with autocast_ctx(device):
                logits = model(x, sid_t.to(device), pos_t.to(device), valid.to(device))
                loss = criterion(logits.float(), y, w)
            (loss / max(1, args.grad_accum)).backward()
            if (it + 1) % max(1, args.grad_accum) == 0 or (it + 1) == len(tr_loader):
                opt.step(); opt.zero_grad(set_to_none=True); sched.step(); gstep += 1
                if ema is not None:
                    ema.update(model)
            seen += x.shape[0] * x.shape[1]; run_loss += float(loss.item()); nb += 1
            if it % 10 == 0:
                ips = seen / max(1e-6, time.time() - ep_t)
                dmsg = ""
                if ema is not None and (it == 0 or gstep in (100, 1000)):
                    dmsg = f" ema_decay={ema.effective_decay():.5f}(t={ema.updates})"
                print(f"  e{epoch} it{it}/{len(tr_loader)} loss={run_loss / nb:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {ips:.1f} img/s{dmsg}", flush=True)
        ep_time = time.time() - ep_t
        ips = seen / max(1e-6, ep_time)
        print(" ", fallback_report())

        # ---- validate BOTH weight sets -------------------------------------------------
        # The EMA alone is not a usable early-epoch signal (see ModelEma) — so the raw
        # weights get their own pass and the better of the two is what gets kept.
        res: dict = {}
        if ema is not None:
            res["ema"] = _evaluate(ema.module, va_loader, device, va_present, y_oof, gold)
        if ema is None or args.eval_raw:
            res["raw"] = _evaluate(model, va_loader, device, va_present, y_oof, gold)
        prim = res["ema"] if "ema" in res else res["raw"]   # the historic OOF/gold fields
        raw = res.get("raw")
        raw_oof = raw["oof"] if raw is not None else float("nan")
        raw_gold = raw["gold"] if raw is not None else float("nan")
        oof_auc, g_auc, n_g = prim["oof"], prim["gold"], prim["n_gold"]
        src = _pick_source(res)
        best = res[src]
        print(f"[epoch {epoch}] loss={run_loss / max(1, nb):.4f} OOF_macroAUC={oof_auc:.4f} "
              f"raw_OOF={raw_oof:.4f} gold_foldAUC={g_auc:.4f} raw_gold={raw_gold:.4f} "
              f"(n={n_g}) saved={src} epoch_time={ep_time:.0f}s {ips:.1f} img/s", flush=True)
        if best["p"] is not None:
            rep = auc_report(best["y"], best["p"])
            print(f"  OOF per-finding ({src}):", {t: round(float(rep[t]), 3) for t in TARGETS})

        # ---- checkpoint (train_hr schema + the v2 args) --------------------------------
        # `model` holds the weights named by `model_source`; `raw_model`/`ema` keep their old
        # meaning so `--resume` and every existing reader keep working. The OOF parquet is
        # written from the SAME source as `model`.
        sel_model = ema.module if src == "ema" else model
        sel_auc = best["oof"]
        fp = compute_fingerprint_v2(sel_model, device, k=va_ds.k, size=args.size)
        args_d = dict(vars(args)); args_d["model_source"] = src
        state = dict(model=(ema.state_dict() if src == "ema" else model.state_dict()),
                     raw_model=(model.state_dict() if ema is not None else None),
                     opt=opt.state_dict(), sched=sched.state_dict(),
                     ema=(ema.state_dict() if ema is not None else None),
                     ema_updates=(ema.updates if ema is not None else 0),
                     model_source=src,
                     epoch=epoch, best_auc=best_auc, gstep=gstep, cfg=vars(cfg),
                     args=args_d, mean=list(map(float, mean)), std=list(map(float, std)),
                     fingerprint=fp, targets=TARGETS, oof_auc=sel_auc, gold_auc=best["gold"],
                     ema_oof_auc=(res["ema"]["oof"] if "ema" in res else float("nan")),
                     raw_oof_auc=raw_oof)
        torch.save(state, ckpt_last)
        improved = (not math.isnan(sel_auc) and sel_auc > best_auc) or not os.path.exists(ckpt_best)
        if improved:
            if not math.isnan(sel_auc):
                best_auc = sel_auc
            state["best_auc"] = best_auc
            torch.save(state, ckpt_best)
            best["preds"].to_parquet(oof_path, index=False)
            print(f"  * saved best -> {ckpt_best} (OOF={sel_auc:.4f} from {src}) + {oof_path}",
                  flush=True)

        if args.max_train_minutes and (time.time() - t0) / 60.0 >= args.max_train_minutes:
            print(f"[time-box] stopping after {(time.time() - t0) / 60:.1f} min", flush=True)
            break

    print(f"[done] v2 fold {args.fold} @{args.size}px best OOF_macroAUC={best_auc:.4f} "
          f"total {(time.time() - t0) / 60:.1f} min")
    return best_auc


if __name__ == "__main__":
    main()
