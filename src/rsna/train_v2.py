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

RAPTOR-RECIPE KNOBS (2026-10-12, `reports/coat2_raptor.md`; ALL default off = the trainer exactly
as before): `--block-ckpt 1` (timm per-block grad checkpointing, so 8 studies x 12 slices = 96
images go through ONE backbone call), `--clip-grad G`, `--fused-adamw 1`, `--cudnn-benchmark 0`,
`--eval-batch B`, `--eval-from-epoch E`, `--swa-top N` (uniform weight average of the N best
validated epochs, kept iff its OOF >= the best single epoch; the pool lives in RAM, so a `--resume`
pools only the epochs it runs) and `--mem-log-every N`. The head/backbone LR split is the existing
`--lr` (head) x `--lr-backbone-mult` (backbone): `--lr 1e-3 --lr-backbone-mult 0.03` = 1e-3 / 3e-5.

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
    parse_cache_contract,
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


def _mem_avail_gb() -> float:
    """/proc/meminfo MemAvailable in GB (the only memory signal on the GB10's unified pool)."""
    try:
        with open("/proc/meminfo") as fh:
            return next(int(l.split()[1]) for l in fh if l.startswith("MemAvailable")) / 1048576
    except Exception:
        return float("nan")


def average_state_dicts(sds):
    """Uniform average of state dicts (SWA over epochs, as the public CoAtNet trainers do it).

    Floating-point tensors — parameters AND BatchNorm running statistics — are averaged in
    float64 and cast back to their dtype (no BN recalibration pass: mattiaangeli's
    `posthoc_swa` = "uniform_float_buffer_mean", `batchnorm_recalibration: False`). Integer
    buffers (`num_batches_tracked`) take the elementwise max. All inputs must share keys/shapes.
    """
    if not sds:
        raise ValueError("no state dicts to average")
    keys = list(sds[0].keys())
    for sd in sds[1:]:
        if list(sd.keys()) != keys:
            raise ValueError("state dicts disagree on keys")
    out = {}
    for k in keys:
        v0 = sds[0][k]
        if torch.is_floating_point(v0):
            acc = torch.zeros(v0.shape, dtype=torch.float64)
            for sd in sds:
                acc += sd[k].detach().to("cpu", torch.float64)
            out[k] = (acc / float(len(sds))).to(v0.dtype)
        else:
            out[k] = torch.stack([sd[k].detach().cpu() for sd in sds]).amax(dim=0).to(v0.dtype)
    return out


def _finalize_swa(pool, model, va_loader, device, va_present, y_oof, gold, best_auc,
                  ckpt_best, oof_path, args, cfg, mean, std, gstep, k_eval) -> float:
    """`--swa-top N`: average the pooled epochs, validate, keep it iff OOF >= best single epoch.

    The SWA weights are written as the best checkpoint (`model_source="swa"`, plus `swa_epochs`,
    `swa_oof_auc`, `best_single_oof_auc`) and the OOF parquet is rewritten from the SAME weights,
    so `infer_v2.load_members_v2` and every OOF reader see one consistent member. Otherwise the
    best single epoch stays exactly as saved. Returns the (possibly updated) best OOF.
    """
    if len(pool) < 2:
        print(f"[swa] only {len(pool)} validated epoch(s) in the pool -> nothing to average; "
              f"best single epoch kept (OOF={best_auc:.4f})", flush=True)
        return best_auc
    eps = [int(e["epoch"]) for e in pool]
    src = pool[0]["src"]
    swa = copy.deepcopy(model)
    swa.load_state_dict(average_state_dicts([e["sd"] for e in pool]))
    r = _evaluate(swa, va_loader, device, va_present, y_oof, gold)
    print(f"[swa] average of epochs {eps} ({src} weights; pool OOF "
          f"{[round(e['oof'], 4) for e in pool]}) -> OOF_macroAUC={r['oof']:.4f} "
          f"gold_foldAUC={r['gold']:.4f} (n={r['n_gold']}) vs best single epoch {best_auc:.4f}",
          flush=True)
    if r["p"] is not None:
        rep = auc_report(r["y"], r["p"])
        print("  OOF per-finding (swa):", {t: round(float(rep[t]), 3) for t in TARGETS})
    if math.isnan(r["oof"]) or r["oof"] < best_auc:
        print(f"[swa] KEPT the best single epoch ({best_auc:.4f} > swa {r['oof']:.4f}); "
              f"{ckpt_best} unchanged", flush=True)
        del swa
        return best_auc
    fp = compute_fingerprint_v2(swa, device, k=k_eval, size=args.size)
    args_d = dict(vars(args)); args_d["model_source"] = "swa"; args_d["swa_epochs"] = eps
    state = dict(model=swa.state_dict(), raw_model=None, opt=None, sched=None, ema=None,
                 ema_updates=0, model_source="swa", epoch=max(eps), best_auc=float(r["oof"]),
                 gstep=gstep, cfg=vars(cfg), args=args_d, mean=list(map(float, mean)),
                 std=list(map(float, std)), fingerprint=fp, targets=TARGETS,
                 oof_auc=float(r["oof"]), gold_auc=r["gold"], ema_oof_auc=float("nan"),
                 raw_oof_auc=float("nan"), swa_epochs=eps, swa_source=src,
                 swa_oof_auc=float(r["oof"]), best_single_oof_auc=float(best_auc))
    torch.save(state, ckpt_best)
    r["preds"].to_parquet(oof_path, index=False)
    print(f"  * saved best -> {ckpt_best} (OOF={r['oof']:.4f} from swa of epochs {eps}) + "
          f"{oof_path}", flush=True)
    del swa
    return float(r["oof"])


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
    ap.add_argument("--min-avail-gb", type=float, default=20.0,
                    help="abort at start if /proc/meminfo MemAvailable is below this (GB). The GB10 "
                         "shares one 128 GB pool between host and GPU and nvidia-smi reports N/A, so "
                         "this is the only memory check; a launch into a full pool hangs the box.")
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
    ap.add_argument("--crop-mm", type=float, default=None,
                    help=f"recorded in the ckpt; the DICOM decoder at inference reads it. "
                         f"Default: parsed from --cache-root (cache_c<crop>_<size>), else "
                         f"{V2_CROP_MM:g}")
    ap.add_argument("--pct", type=float, nargs=2, default=list(V2_PCT),
                    help="per-series normalisation percentiles, recorded in the ckpt")
    ap.add_argument("--slice-context", type=int, default=0,
                    help="2.5D slot context, recorded in the ckpt and read back by "
                         "infer_v2.MemberV2. 0 (default) = today's replicated grayscale; "
                         "1 = the 3 backbone channels become [prev, this, next] of the same "
                         "series in anatomical order (clamped at the ends, canonical "
                         "direction for a left knee). Only 0 or 1: the stem takes 3 channels.")
    # --- "Raptor recipe" knobs (reports/coat2_raptor.md). Every default reproduces the trainer
    # as it was before they existed; none of them is read unless set. ----------------------------
    ap.add_argument("--block-ckpt", type=int, default=0,
                    help="1 = timm PER-BLOCK gradient checkpointing (backbone.set_grad_checkpointing), "
                         "training forward only; lets ONE backbone call take all B*K slices "
                         "(the public CoAtNet trainers: 8 studies x 12 = 96). Not with --grad-ckpt.")
    ap.add_argument("--clip-grad", type=float, default=0.0,
                    help="> 0 = clip the global grad L2 norm to this before each optimizer step "
                         "(public CoAtNet trainers: 3.0-5.0). 0 = off (default).")
    ap.add_argument("--fused-adamw", type=int, default=0,
                    help="1 = torch.optim.AdamW(fused=True) on CUDA. 0 = the default implementation.")
    ap.add_argument("--cudnn-benchmark", type=int, default=1,
                    help="torch.backends.cudnn.benchmark on CUDA (default 1, as always)")
    ap.add_argument("--eval-batch", type=int, default=0,
                    help="studies per VALIDATION batch (each = 64 slices); 0 = --batch (default)")
    ap.add_argument("--eval-from-epoch", type=int, default=0,
                    help="> 0 = skip the validation pass on epochs 1..E-1 (epoch 0, every epoch >= E "
                         "and the last/time-boxed epoch are always validated; a skipped epoch still "
                         "writes the _last.pt resume checkpoint). 0 = validate every epoch (default).")
    ap.add_argument("--swa-top", type=int, default=0,
                    help=">= 2 = after training, average the weights of the N best VALIDATED epochs "
                         "(ranked by their OOF macro-AUC; raw weights when they were evaluated, else "
                         "EMA), validate the average, and keep it as the best checkpoint + OOF iff "
                         "its OOF >= the best single epoch's. BatchNorm buffers are averaged too "
                         "(no recalibration), integer buffers take the max. 0 = off (default).")
    ap.add_argument("--mem-log-every", type=int, default=0,
                    help="> 0 = print a [mem] line (MemAvailable, torch peak allocated) every N "
                         "training iterations. 0 = off (default).")
    args = ap.parse_args()

    if int(args.block_ckpt) and args.grad_ckpt:
        raise SystemExit("ERROR: --block-ckpt (timm per-block recompute) and --grad-ckpt (per-chunk "
                         "recompute) are mutually exclusive")
    if int(args.swa_top) == 1 or int(args.swa_top) < 0:
        raise SystemExit(f"ERROR: --swa-top must be 0 (off) or >= 2, got {args.swa_top}")

    if int(args.slice_context) not in (0, 1):
        raise SystemExit(f"ERROR: --slice-context must be 0 or 1 (the backbone stem takes 3 "
                         f"channels, so N neighbours each side means 2N+1 = 3); got "
                         f"{args.slice_context}")

    # ---- the crop contract, settled BEFORE anything is loaded ---------------------------
    # `args["crop_mm"]` is not a label: `infer_v2._predict_dicom` crops the Kaggle DICOMs with
    # it, so a checkpoint whose value disagrees with the cache it was trained on is fed a
    # different physical field of view at test time than at train time — silently, with
    # plausible numbers. That is exactly what happened to `v2_effv2s384_c110_fold{0..4}`:
    # launched off `data/cache_c110_384` without `--crop-mm 110`, they shipped saying 140.0.
    # So: derive the crop from the cache root's name, adopt it when the flag was not given,
    # and refuse to start when the two disagree.
    _root_crop, _root_size = parse_cache_contract(args.cache_root)
    if args.crop_mm is None:
        if _root_crop is not None:
            args.crop_mm = float(_root_crop)
            print(f"[cfg] crop_mm taken from cache root: {args.crop_mm:g}")
        else:
            args.crop_mm = float(V2_CROP_MM)
    elif _root_crop is not None and abs(float(args.crop_mm) - float(_root_crop)) > 1e-6:
        raise SystemExit(
            f"ERROR: --crop-mm {float(args.crop_mm):g} disagrees with the cache root "
            f"{args.cache_root}, whose name says {_root_crop:g} mm. The checkpoint's crop_mm "
            f"is what the DICOM decoder crops with at inference, so a wrong value feeds the "
            f"model a different physical field of view at test time than it saw in training. "
            f"Pass --crop-mm {_root_crop:g}, or drop the flag and let the cache root decide.")

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
        torch.backends.cudnn.benchmark = bool(int(args.cudnn_benchmark))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"[cfg] mode=v2 device={device} size={args.size} backbone={args.backbone}")
    print(f"[cfg] k_scale={args.k_scale} slots={args.slot_counts} K={args.k} (eval K={args.k_eval}) "
          f"batch={args.batch} grad_accum={args.grad_accum} workers={args.workers}")
    print(f"[cfg] cache_root={args.cache_root} crop_mm={args.crop_mm} pct={tuple(args.pct)} "
          f"slice_context={args.slice_context}"
          + ("  (channels = [prev, this, next] in anatomical order)"
             if args.slice_context else "  (grayscale replicated x3)"))
    _raptor = dict(block_ckpt=int(args.block_ckpt), clip_grad=float(args.clip_grad),
                   fused_adamw=int(args.fused_adamw), cudnn_benchmark=int(args.cudnn_benchmark),
                   eval_batch=int(args.eval_batch), eval_from_epoch=int(args.eval_from_epoch),
                   swa_top=int(args.swa_top), mem_log_every=int(args.mem_log_every))
    _raptor_default = dict(block_ckpt=0, clip_grad=0.0, fused_adamw=0, cudnn_benchmark=1,
                           eval_batch=0, eval_from_epoch=0, swa_top=0, mem_log_every=0)
    if _raptor != _raptor_default:          # printed only when a recipe knob is in use
        print("[cfg] recipe knobs: " + " ".join(f"{k}={v}" for k, v in _raptor.items())
              + f" | images/forward={args.batch * args.k} (batch {args.batch} x K {args.k})")

    # ---- memory guard (2026-09-19 box hang: 512 run launched with ~30 GB available) --------
    try:
        with open("/proc/meminfo") as fh:
            _avail_gb = next(int(l.split()[1]) for l in fh if l.startswith("MemAvailable")) / 1048576
    except Exception:
        _avail_gb = float("nan")
    print(f"[mem] MemAvailable={_avail_gb:.1f} GB (guard: abort below {args.min_avail_gb:.0f} GB)", flush=True)
    if not _avail_gb >= args.min_avail_gb:
        print(f"[mem] ABORT: only {_avail_gb:.1f} GB available < {args.min_avail_gb:.0f} GB "
              f"(is qwen-server / another job holding the unified memory?)", flush=True)
        raise SystemExit(3)

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
                        slice_chunk=int(args.slice_chunk), block_ckpt=bool(int(args.block_ckpt)))
    model = build_model_v2(cfg).to(device)
    if cuda and model._channels_last:
        model = model.to(memory_format=torch.channels_last)
    dc = model.data_config()
    mean, std = dc["mean"], dc["std"]
    print(f"[data-config] mean={mean} std={std} feat_dim={model.feat_dim}")

    tr_ds = StudyBagDatasetV2(tr_ids, index, labels, size=args.size, train=True, mean=mean,
                              std=std, k_scale=args.k_scale, side_map=side_map,
                              jitter=args.jitter, slice_context=args.slice_context)
    va_ds = StudyBagDatasetV2(va_ids, index, labels, size=args.size, train=False, mean=mean,
                              std=std, k_scale=1.0, side_map=side_map,
                              slice_context=args.slice_context)          # eval = full 64
    n_gold_val = int(labels.loc[[s for s in va_ids if s in labels.index], "is_gold"].sum())
    print(f"[fold {args.fold}] train={len(tr_ds)} (K={tr_ds.k}) val={len(va_ds)} (K={va_ds.k}) "
          f"studies @ {args.size}px, gold-in-fold={n_gold_val}")

    drop_last = len(tr_ds) >= 2 * args.batch
    # pin_memory OFF by default on the GB10 (2026-09-21): host and device share one memory pool, so pinning buys
    # nothing, and the per-batch cudaHostAlloc churn (~270 MB/batch) walked the page-allocator slow path once the
    # box was fragmented (108 -> 35 img/s mid-run, GPU 96 % util at 17 W). RSNA_PIN_MEMORY=1 restores it.
    pin = cuda and os.environ.get("RSNA_PIN_MEMORY", "0") == "1"
    loader_kw = dict(num_workers=args.workers, pin_memory=pin,
                     persistent_workers=args.workers > 0,
                     prefetch_factor=2 if args.workers > 0 else None)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=drop_last,
                           **loader_kw)
    va_loader = DataLoader(va_ds, batch_size=max(1, int(args.eval_batch) or args.batch),
                           shuffle=False, **loader_kw)

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
    _opt_kw = {"fused": True} if (int(args.fused_adamw) and cuda) else {}
    opt = torch.optim.AdamW(model.param_groups(args.lr, args.lr_backbone_mult),
                            lr=args.lr, weight_decay=args.wd, **_opt_kw)
    print(f"[optim] AdamW lr={args.lr:.2e} backbone_lr={args.lr * args.lr_backbone_mult:.2e} "
          f"wd={args.wd} ema={args.ema}"
          + (" fused=True" if _opt_kw else "")
          + (f" clip_grad={args.clip_grad:g}" if args.clip_grad and args.clip_grad > 0 else ""))
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

    clip = float(args.clip_grad) if args.clip_grad and args.clip_grad > 0 else 0.0
    mem_every = max(0, int(args.mem_log_every))
    swa_n = int(args.swa_top)
    swa_pool: list = []          # [{epoch, oof, src, sd(cpu)}], best first — only with --swa-top
    gnorm = None
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
                if clip:
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step(); opt.zero_grad(set_to_none=True); sched.step(); gstep += 1
                if ema is not None:
                    ema.update(model)
            seen += x.shape[0] * x.shape[1]; run_loss += float(loss.item()); nb += 1
            if it % 10 == 0:
                ips = seen / max(1e-6, time.time() - ep_t)
                dmsg = ""
                if ema is not None and (it == 0 or gstep in (100, 1000)):
                    dmsg = f" ema_decay={ema.effective_decay():.5f}(t={ema.updates})"
                if clip and gnorm is not None:
                    dmsg += f" gnorm={float(gnorm):.3f}"
                print(f"  e{epoch} it{it}/{len(tr_loader)} loss={run_loss / nb:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {ips:.1f} img/s{dmsg}", flush=True)
            if mem_every and it % mem_every == 0:
                _peak = (f" torch_peak_alloc={torch.cuda.max_memory_allocated() / 2**30:.1f}GB"
                         f" reserved={torch.cuda.memory_reserved() / 2**30:.1f}GB" if cuda else "")
                print(f"  [mem] e{epoch} it{it} MemAvailable={_mem_avail_gb():.1f}GB{_peak}", flush=True)
        ep_time = time.time() - ep_t
        ips = seen / max(1e-6, ep_time)
        print(" ", fallback_report())

        # ---- --eval-from-epoch: skip this epoch's validation (resume checkpoint only) ----------
        if int(args.eval_from_epoch) > 0 and 0 < epoch < int(args.eval_from_epoch) \
                and epoch != args.epochs - 1 \
                and not (args.max_train_minutes and (time.time() - t0) / 60.0 >= args.max_train_minutes):
            args_d = dict(vars(args)); args_d["model_source"] = "raw"
            torch.save(dict(model=model.state_dict(),
                            raw_model=(model.state_dict() if ema is not None else None),
                            opt=opt.state_dict(), sched=sched.state_dict(),
                            ema=(ema.state_dict() if ema is not None else None),
                            ema_updates=(ema.updates if ema is not None else 0),
                            model_source="raw", epoch=epoch, best_auc=best_auc, gstep=gstep,
                            cfg=vars(cfg), args=args_d, mean=list(map(float, mean)),
                            std=list(map(float, std)), fingerprint=None, targets=TARGETS,
                            oof_auc=float("nan"), gold_auc=float("nan"),
                            ema_oof_auc=float("nan"), raw_oof_auc=float("nan")), ckpt_last)
            print(f"[epoch {epoch}] loss={run_loss / max(1, nb):.4f} validation skipped "
                  f"(--eval-from-epoch {args.eval_from_epoch}) epoch_time={ep_time:.0f}s "
                  f"{ips:.1f} img/s", flush=True)
            continue

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

        # ---- --swa-top: keep the N best validated epochs' weights (CPU copies) ------------
        if swa_n >= 2:
            pk = "raw" if "raw" in res else "ema"
            p_auc = res[pk]["oof"]
            if not math.isnan(p_auc) and (len(swa_pool) < swa_n
                                          or p_auc > min(e["oof"] for e in swa_pool)):
                pm = model if pk == "raw" else ema.module
                swa_pool.append(dict(epoch=epoch, oof=float(p_auc), src=pk,
                                     sd={k_: v_.detach().to("cpu", copy=True)
                                         for k_, v_ in pm.state_dict().items()}))
                swa_pool.sort(key=lambda e: -e["oof"])
                del swa_pool[swa_n:]
            print(f"  [swa] pool ({pk} weights, top {swa_n} by OOF): "
                  + ", ".join(f"e{e['epoch']}={e['oof']:.4f}" for e in swa_pool), flush=True)

        if args.max_train_minutes and (time.time() - t0) / 60.0 >= args.max_train_minutes:
            print(f"[time-box] stopping after {(time.time() - t0) / 60:.1f} min", flush=True)
            break

    if swa_n >= 2:
        best_auc = _finalize_swa(swa_pool, model, va_loader, device, va_present, y_oof, gold,
                                 best_auc, ckpt_best, oof_path, args, cfg, mean, std, gstep,
                                 va_ds.k)

    print(f"[done] v2 fold {args.fold} @{args.size}px best OOF_macroAUC={best_auc:.4f} "
          f"total {(time.time() - t0) / 60:.1f} min")
    return best_auc


if __name__ == "__main__":
    main()
