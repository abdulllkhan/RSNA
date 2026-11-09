#!/usr/bin/env python
"""CoAtNet-2 "Raptor recipe" throughput bench on the GB10 — ONE config per process.

Mirrors `train_v2`'s training step exactly (same model class, same dataset on the real cache,
same param groups / loss / autocast / channels_last), so img/s here is what the trainer will
see: real JPEG bags from `data/cache_c110_384` through 8 DataLoader workers, bf16 autocast,
weighted soft-BCE, backward, grad clip, AdamW step. Memory is measured the only way that works
on the unified-memory GB10: /proc/meminfo MemAvailable sampled every 0.25 s by a thread, as the
drop from the value read when the process started (CUDA context + model + optimizer + workers +
activations), plus torch's own peak allocated / reserved.

    # training step (one config)
    python dev/bench_coat2.py --mode train --batch 8 --k-scale 0.1875 --block-ckpt 1 --time-s 45
    # eval-mode forward only (K=64 bags, the Kaggle shape), CoAtNet-2 and effv2s, bf16 + fp16
    python dev/bench_coat2.py --mode infer
Each run prints one JSON line (and appends it to --out). Run under the box lock (dev/bench_coat2.sh).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

COAT = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
T_PROC = time.time()


def mem_avail_gb() -> float:
    with open("/proc/meminfo") as fh:
        return next(int(l.split()[1]) for l in fh if l.startswith("MemAvailable")) / 1048576


class MemSampler(threading.Thread):
    def __init__(self, period: float = 0.25):
        super().__init__(daemon=True)
        self.period, self.min_gb, self.stop_ev = period, mem_avail_gb(), threading.Event()

    def run(self):
        while not self.stop_ev.is_set():
            self.min_gb = min(self.min_gb, mem_avail_gb())
            time.sleep(self.period)


def run_train(a, base_gb: float, sampler: MemSampler) -> dict:
    import pandas as pd
    from torch.utils.data import DataLoader
    from src.rsna import TARGETS, ID_COL
    from src.rsna.dataset_v2 import StudyBagDatasetV2, build_index_v2, load_labels_v2, load_side_map
    from src.rsna.model_v2 import ModelV2Config, build_model_v2
    from src.rsna.order import SliceOrderResolver
    from src.rsna.train_hr import autocast_ctx

    dev = "cuda"
    torch.backends.cudnn.benchmark = bool(a.cudnn_benchmark)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    labels = load_labels_v2(os.path.join(REPO, "data/labels/consensus_v1.parquet"))
    resolver = SliceOrderResolver.from_parquet(os.path.join(REPO, "data/series_order.parquet"),
                                               split="train")
    index = build_index_v2(os.path.join(REPO, a.cache_root), split="train", order=resolver)
    side = load_side_map(os.path.join(REPO, "data/study_side.parquet"))
    fo = labels["fold"].astype(int)
    tr_ids = [s for s in fo.index[fo != 0].astype(str) if s in index]
    cfg = ModelV2Config(backbone=a.backbone, pretrained=True, size=384, dropout=0.2,
                        block_ckpt=bool(a.block_ckpt), slice_chunk=int(a.slice_chunk))
    model = build_model_v2(cfg).to(dev)
    if model._channels_last:
        model = model.to(memory_format=torch.channels_last)
    dc = model.data_config()
    ds = StudyBagDatasetV2(tr_ids, index, labels, size=384, train=True, mean=dc["mean"],
                           std=dc["std"], k_scale=a.k_scale, side_map=side)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                        num_workers=a.workers, pin_memory=False, persistent_workers=False,
                        prefetch_factor=2)
    kw = {"fused": True} if a.fused else {}
    opt = torch.optim.AdamW(model.param_groups(1e-3, 0.03), lr=1e-3, weight_decay=0.02, **kw)
    model.train()
    print(f"[bench] setup {time.time() - T_PROC:.1f}s since process start; K={ds.k} "
          f"images/forward={a.batch * ds.k}", flush=True)
    t_start = time.time(); steps = 0; timed_steps = 0; t_timed = None; losses = []
    t_prev = time.time()
    for x, sid_t, pos_t, valid, y, w, _s in loader:
        x = x.to(dev, non_blocking=True); y = y.to(dev); w = w.to(dev)
        with autocast_ctx(dev):
            logits = model(x, sid_t.to(dev), pos_t.to(dev), valid.to(dev))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), y, weight=w, reduction="mean")
        loss.backward()
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step(); opt.zero_grad(set_to_none=True)
        steps += 1
        torch.cuda.synchronize()
        print(f"[bench] step {steps} {time.time() - t_prev:.2f}s loss={float(loss.item()):.4f} "
              f"MemAvailable={mem_avail_gb():.1f}GB", flush=True)
        t_prev = time.time()
        if steps == a.warmup:
            torch.cuda.synchronize(); t_timed = time.time()
        elif steps > a.warmup:
            timed_steps += 1
            losses.append(float(loss.item()))
            if time.time() - t_timed >= a.time_s:
                break
        if time.time() - t_start > a.time_s + 240:          # hard stop (slow autotune etc.)
            break
    torch.cuda.synchronize()
    el = time.time() - (t_timed or t_start)
    imgs = timed_steps * a.batch * ds.k
    return dict(mode="train", backbone=a.backbone, batch=a.batch, k=ds.k,
                slots=list(map(int, np.bincount(ds._slot_id))), images_per_forward=a.batch * ds.k,
                block_ckpt=int(a.block_ckpt), slice_chunk=int(a.slice_chunk),
                cudnn_benchmark=int(a.cudnn_benchmark), fused=int(a.fused), clip=a.clip,
                warmup_steps=a.warmup, timed_steps=timed_steps, timed_s=round(el, 1),
                img_per_s=round(imgs / max(1e-6, el), 2),
                s_per_step=round(el / max(1, timed_steps), 2),
                startup_s=round((t_timed or time.time()) - t_start, 1),
                mean_loss=round(float(np.mean(losses)), 4) if losses else None,
                torch_peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                torch_peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                mem_avail_start_gb=round(base_gb, 1), mem_avail_min_gb=round(sampler.min_gb, 1),
                mem_drop_gb=round(base_gb - sampler.min_gb, 1))


@torch.no_grad()
def run_infer(a, base_gb: float, sampler: MemSampler) -> dict:
    from src.rsna.model_v2 import ModelV2Config, build_model_v2
    dev = "cuda"
    torch.backends.cudnn.benchmark = bool(a.cudnn_benchmark)
    out = dict(mode="infer", batch_studies=a.infer_batch, k_eval=64, rows=[])
    g = torch.Generator().manual_seed(0)
    x = torch.randn(a.infer_batch, 64, 3, 384, 384, generator=g).to(dev)
    for name, chunk in [(COAT, 96), (COAT, 0), ("tf_efficientnetv2_s", 0)]:
        cfg = ModelV2Config(backbone=name, pretrained=False, size=384, slice_chunk=chunk)
        m = build_model_v2(cfg, verbose=False).to(dev).to(memory_format=torch.channels_last).eval()
        for dt in (torch.float16, torch.bfloat16):
            for _ in range(2):
                with torch.autocast("cuda", dtype=dt):
                    m(x)
            torch.cuda.synchronize(); t = time.time(); n = 0
            while n < a.infer_iters:
                with torch.autocast("cuda", dtype=dt):
                    lo = m(x)
                n += 1
            torch.cuda.synchronize(); el = time.time() - t
            out["rows"].append(dict(backbone=name, slice_chunk=chunk, dtype=str(dt).split(".")[-1],
                                    ms_per_study_variant=round(1000 * el / (n * a.infer_batch), 1),
                                    img_per_s=round(n * a.infer_batch * 64 / el, 1),
                                    finite=bool(torch.isfinite(lo).all())))
        del m; torch.cuda.empty_cache()
    out.update(torch_peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               mem_avail_start_gb=round(base_gb, 1), mem_avail_min_gb=round(sampler.min_gb, 1),
               mem_drop_gb=round(base_gb - sampler.min_gb, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("train", "infer"), default="train")
    ap.add_argument("--tag", default="")
    ap.add_argument("--backbone", default=COAT)
    ap.add_argument("--cache-root", default="data/cache_c110_384")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--k-scale", type=float, default=0.1875)
    ap.add_argument("--block-ckpt", type=int, default=1)
    ap.add_argument("--slice-chunk", type=int, default=0)
    ap.add_argument("--cudnn-benchmark", type=int, default=1)
    ap.add_argument("--fused", type=int, default=1)
    ap.add_argument("--clip", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--time-s", type=float, default=45.0)
    ap.add_argument("--infer-batch", type=int, default=2)
    ap.add_argument("--infer-iters", type=int, default=6)
    ap.add_argument("--min-avail-gb", type=float, default=60.0)
    ap.add_argument("--out", default=os.path.join(REPO, "reports", "coat2_bench.jsonl"))
    a = ap.parse_args()
    base = mem_avail_gb()
    if base < a.min_avail_gb:
        raise SystemExit(f"[bench] ABORT: MemAvailable {base:.1f} GB < {a.min_avail_gb:.0f} GB")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    s = MemSampler(); s.start()
    t0 = time.time()
    try:
        rec = run_train(a, base, s) if a.mode == "train" else run_infer(a, base, s)
        rec["status"] = "ok"
    except Exception as e:                       # an OOM must still leave a record
        rec = dict(mode=a.mode, batch=a.batch, block_ckpt=a.block_ckpt, status="FAILED",
                   error=f"{type(e).__name__}: {str(e)[:200]}",
                   mem_avail_start_gb=round(base, 1), mem_avail_min_gb=round(s.min_gb, 1))
    s.stop_ev.set(); s.join(timeout=2)
    rec["mem_avail_min_gb"] = round(min(s.min_gb, rec.get("mem_avail_min_gb", 1e9)), 1)
    rec["mem_drop_gb"] = round(base - rec["mem_avail_min_gb"], 1)
    rec.update(tag=a.tag, wall_s=round(time.time() - t0, 1),
               when=time.strftime("%Y-%m-%d %H:%M:%S"))
    line = json.dumps(rec)
    print(line, flush=True)
    with open(a.out, "a") as fh:
        fh.write(line + "\n")


if __name__ == "__main__":
    main()
