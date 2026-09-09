#!/usr/bin/env python
"""Backbone fwd+bwd throughput bench for the v2 arm (why is CoAtNet slow on sm_121?).

Mirrors exactly how `src/rsna/train_v2.py` + `model_v2.py` call the backbone:
  * `timm.create_model(name, pretrained=..., num_classes=0)` (+ the img_size cascade)
  * `.to(memory_format=torch.channels_last)` for CNNs, plain contiguous for ViTs
  * `torch.autocast('cuda', dtype=torch.bfloat16)`
  * `torch.backends.cudnn.benchmark = True`
one AdamW step per iteration, 5 warmup iters then N timed iters (N shrinks to fit a
per-repeat wall budget so a slow config cannot eat the whole GPU allowance).

Results are appended to a JSONL as they finish, so a crash/OOM keeps what ran.

    python dev/bench_backbones.py --list
    python dev/bench_backbones.py --only ref,coat_a --out reports/bench.jsonl
    python dev/bench_backbones.py --table reports/bench.jsonl      # render markdown
    python dev/bench_backbones.py --flops-only > reports/backbone_flops.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO = Path(__file__).resolve().parents[1]
DINO_LOG = REPO / "logs" / "v2_dino336_fold0.log"
MEM_CAP_GB = 20.0


# ---------------------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------------------
@dataclass
class Cfg:
    key: str
    name: str
    size: int = 384
    batch: int = 16
    channels_last: bool = True
    amp: str = "bf16"            # bf16 | fp16 | fp32
    tf32: bool = False           # torch.backends.cuda.matmul.allow_tf32
    compile: bool = False
    pretrained: bool = False
    vit: bool = False            # try img_size + dynamic_img_size first
    note: str = ""

    @property
    def layout(self) -> str:
        return "channels_last" if self.channels_last else "contiguous"


COAT = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"

CONFIGS: List[Cfg] = [
    # 1. reference
    Cfg("ref", "tf_efficientnetv2_s", 384, 32, note="reference (trainer settings)"),
    # 2. coatnet variants
    Cfg("coat_a", COAT, 384, 16, note="as the trainer does it"),
    Cfg("coat_b", COAT, 384, 16, channels_last=False, note="no channels_last"),
    Cfg("coat_c", COAT, 384, 16, amp="fp16", note="fp16 autocast"),
    Cfg("coat_d", COAT, 384, 16, amp="fp32", note="fp32, no autocast"),
    Cfg("coat_e", COAT, 384, 16, tf32=True, note="bf16 + matmul tf32 allowed"),
    Cfg("coat_f", COAT, 384, 16, compile=True, note="torch.compile"),
    # fp32 activations at B=16 exceed the 20 GB self-imposed cap, so the fp32-vs-bf16
    # comparison is run at B=8 against a B=8 bf16 control.
    Cfg("coat_a8", COAT, 384, 8, note="bf16 control for the fp32 run"),
    Cfg("coat_d8", COAT, 384, 8, amp="fp32", note="fp32, no autocast (B=8)"),
    Cfg("coat_pt", COAT, 384, 16, pretrained=True, note="pretrained=True sanity"),
    # 3. candidates
    Cfg("cnxt_s", "convnext_small.fb_in22k_ft_in1k_384", 384, 16),
    Cfg("cnxt_b", "convnext_base.fb_in22k_ft_in1k_384", 384, 16),
    Cfg("effv2_m", "tf_efficientnetv2_m.in21k_ft_in1k", 384, 16),
    Cfg("effv2_l", "tf_efficientnetv2_l.in21k_ft_in1k", 384, 16),
    Cfg("coat1", "coatnet_1_rw_224.sw_in1k", 384, 16),
    Cfg("maxvit_s", "maxvit_small_tf_384.in1k", 384, 16),
    Cfg("swin_b", "swin_base_patch4_window12_384.ms_in22k_ft_in1k", 384, 16),
    # NOTE: the asked-for tag `eva02_small_patch14_336.mim_in22k` does not exist in
    # timm 1.0.29 (only the 224 variant carries a bare `mim_in22k`); the 336 model's
    # only tag is `.mim_in22k_ft_in1k`, which is the same architecture.
    Cfg("eva02_s", "eva02_small_patch14_336.mim_in22k_ft_in1k", 336, 16,
        channels_last=False, vit=True),
    Cfg("dinov2_b", "vit_base_patch14_dinov2.lvd142m", 336, 16,
        channels_last=False, vit=True),
]
BY_KEY = {c.key: c for c in CONFIGS}


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def dino_rate() -> Optional[float]:
    """img/s of the concurrent DINOv2 training job, as a GPU-load indicator."""
    try:
        txt = DINO_LOG.read_text(errors="ignore")
    except OSError:
        return None
    hits = re.findall(r"([\d.]+) img/s", txt)
    return float(hits[-1]) if hits else None


def build(cfg: Cfg) -> torch.nn.Module:
    """The `model_v2._create_backbone` cascade: img_size+dynamic -> img_size -> plain."""
    base = dict(pretrained=cfg.pretrained, num_classes=0)
    attempts: List[dict] = []
    if cfg.vit:
        attempts.append(dict(img_size=cfg.size, dynamic_img_size=True))
    attempts.append(dict(img_size=cfg.size))
    attempts.append({})
    import timm
    last = None
    for extra in attempts:
        try:
            m = timm.create_model(cfg.name, **base, **extra)
            m._built_with = extra  # noqa: SLF001
            return m
        except (TypeError, RuntimeError, ValueError, KeyError) as e:
            last = e
    raise RuntimeError(f"could not build {cfg.name!r}: {last}")


def gflops_per_img(cfg: Cfg) -> Optional[float]:
    """MACs*2 via torch's built-in FlopCounterMode on meta tensors (no GPU, no weights)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        import dataclasses
        cfg = dataclasses.replace(cfg, pretrained=False)   # never fetch weights for FLOPs
        with torch.device("meta"):
            m = build(cfg)
            x = torch.randn(1, 3, cfg.size, cfg.size)
        fc = FlopCounterMode(display=False)
        with fc:
            m(x)
        return fc.get_total_flops() / 2 / 1e9
    except Exception:
        return None


def amp_ctx(amp: str):
    if amp == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if amp == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    from contextlib import nullcontext
    return nullcontext()


def run_cfg(cfg: Cfg, warmup: int, max_iters: int, budget_s: float,
            repeats: int, profile: bool) -> Dict:
    """Build once, warm up once, then time `repeats` measurement passes; report the max."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)

    rec: Dict = {**asdict(cfg), "layout": cfg.layout, "ok": False,
                 "dino_img_s_before": dino_rate()}
    t_build = time.time()
    model = build(cfg).cuda().train()
    rec["built_with"] = str(getattr(model, "_built_with", {}))
    rec["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if cfg.compile:
        model = torch.compile(model)
    rec["build_s"] = round(time.time() - t_build, 1)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(cfg.batch, 3, cfg.size, cfg.size, device="cuda")
    if cfg.channels_last:
        x = x.to(memory_format=torch.channels_last)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp == "fp16"))

    def step():
        opt.zero_grad(set_to_none=True)
        with amp_ctx(cfg.amp):
            out = model(x)
            loss = out.float().square().mean()
        if cfg.amp == "fp16":
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    per_iter = (time.time() - t0) / max(1, warmup)
    rec["warmup_s"] = round(time.time() - t0, 1)

    n = max(4, min(max_iters, int(budget_s / max(per_iter, 1e-6))))
    rec["iters"] = n
    rates = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            step()
        torch.cuda.synchronize()
        rates.append(cfg.batch * n / (time.time() - t))
    rec["img_s"] = round(max(rates), 2)
    rec["img_s_all"] = [round(r, 2) for r in rates]
    rec["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    rec["dino_img_s_after"] = dino_rate()
    rec["ok"] = True

    if profile:
        rec["profile"] = profile_kernels(step)

    del model, opt, x
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def profile_kernels(step, iters: int = 3, topk: int = 8) -> Dict:
    """Top CUDA kernels for one config, at two levels.

    `kernels` = the real device kernels (what actually ran, so a slow fallback shows up
    by name); `ops`   = the ATen operators they were dispatched from (what to blame in
    the model). Both are ranked by self device time.
    """
    from torch.profiler import profile, ProfilerActivity
    from torch.autograd import DeviceType
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
    evts = list(prof.key_averages())
    attr = "self_device_time_total" if hasattr(evts[0], "self_device_time_total") \
        else "self_cuda_time_total"

    def top(pool, total):
        rows = []
        for e in sorted(pool, key=lambda e: getattr(e, attr, 0.0), reverse=True)[:topk]:
            t = getattr(e, attr, 0.0)
            if t <= 0:
                break
            rows.append({"name": e.key[:100], "cuda_ms": round(t / 1e3, 1),
                         "pct": round(100 * t / total, 1), "calls": int(e.count)})
        return rows

    kern = [e for e in evts if getattr(e, "device_type", None) == DeviceType.CUDA]
    ops = [e for e in evts if getattr(e, "device_type", None) == DeviceType.CPU]
    k_tot = sum(max(0.0, getattr(e, attr, 0.0)) for e in kern) or 1.0
    o_tot = sum(max(0.0, getattr(e, attr, 0.0)) for e in ops) or 1.0
    return {"iters": iters, "kernel_cuda_ms_total": round(k_tot / 1e3, 1),
            "kernels": top(kern, k_tot), "ops": top(ops, o_tot)}


# ---------------------------------------------------------------------------------------
# table rendering
# ---------------------------------------------------------------------------------------
def load_flops(path: Path) -> Dict[str, Optional[float]]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["key"]] = d.get("gflops")
    return out


def render_table(recs: List[Dict], flops: Optional[Dict] = None) -> str:
    """Markdown table, best (max img/s) record per config key.

    `TFLOP/s` is the achieved rate: img/s x GFLOPs/img x 3, the 3 being the usual
    fwd+bwd multiplier. It is the column that says whether a backbone is running its
    arithmetic well or falling off a cliff -- raw img/s only says how big it is.
    """
    flops = flops or {}
    ref_g = flops.get("ref") or 1.0
    best: Dict[str, Dict] = {}
    for r in recs:
        k = r["key"]
        if k not in best or (r.get("ok") and r["img_s"] > best[k].get("img_s", -1)):
            best[k] = r
    lines = [
        "| key | backbone | size | B | dtype / layout | img/s | peak GB | GFLOPs/img "
        "| xEffv2s | params M | TFLOP/s | note |",
        "|" + "---|" * 12,
    ]
    for k, r in best.items():
        nm = f"`{r['name']}`"
        g = flops.get(k)
        if not r.get("ok"):
            err = (r.get("error") or "")[:70]
            lines.append(f"| {k} | {nm} | {r.get('size','')} | {r.get('batch','')} | "
                         f"{r.get('amp','')} / {r.get('layout','')} | n/a | n/a | "
                         f"{g:.1f} | {g / ref_g:.2f} | - | - | FAILED: {err} |"
                         if g else
                         f"| {k} | {nm} | - | - | - | n/a | n/a | n/a | - | - | - | "
                         f"FAILED: {err} |")
            continue
        mode = f"{r['amp']} / {r['layout']}"
        if r.get("compile"):
            mode += " / compiled"
        if r.get("tf32"):
            mode += " / tf32"
        gs = f"{g:.1f}" if g else "n/a"
        xs = f"{g / ref_g:.2f}" if g else "-"
        tf = f"{r['img_s'] * g * 3 / 1000:.2f}" if g else "-"
        lines.append(
            f"| {k} | {nm} | {r['size']} | {r['batch']} | {mode} | **{r['img_s']:.1f}** "
            f"| {r['peak_gb']:.2f} | {gs} | {xs} | {r.get('params_M', 0):.1f} | {tf} "
            f"| {r.get('note', '')} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated config keys")
    ap.add_argument("--out", default=str(REPO / "reports" / "backbone_bench.jsonl"))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--budget", type=float, default=8.0,
                    help="seconds per timed repeat (iters shrink to fit)")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--profile", default="", help="config keys to also profile")
    ap.add_argument("--flops-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--table", default="", help="render the markdown table from a JSONL")
    ap.add_argument("--flops", default=str(REPO / "reports" / "backbone_flops.jsonl"))
    args = ap.parse_args()

    if args.table:
        recs = [json.loads(l) for l in Path(args.table).read_text().splitlines() if l.strip()]
        print(render_table(recs, load_flops(Path(args.flops))))
        return 0

    if args.list:
        for c in CONFIGS:
            print(f"{c.key:10s} {c.name:52s} {c.size} B={c.batch} {c.amp} {c.layout} {c.note}")
        return 0

    keys = [k for k in args.only.split(",") if k] or [c.key for c in CONFIGS]
    prof_keys = set(k for k in args.profile.split(",") if k)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.flops_only:
        for k in keys:
            cfg = BY_KEY[k]
            g = gflops_per_img(cfg)
            print(json.dumps({"key": k, "name": cfg.name, "size": cfg.size, "gflops": g}),
                  flush=True)
        return 0

    # Hard cap our own allocator so a fat config OOMs cleanly instead of crowding out
    # the concurrent training job that owns this GPU.
    tot_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, MEM_CAP_GB / tot_gb))
    print(f"[bench] memory cap {MEM_CAP_GB} GB of {tot_gb:.0f} GB; "
          f"dino={dino_rate()} img/s", flush=True)

    total_gpu = 0.0
    for k in keys:
        cfg = BY_KEY[k]
        t0 = time.time()
        try:
            rec = run_cfg(cfg, args.warmup, args.iters, args.budget, args.repeats,
                          profile=(k in prof_keys))
        except Exception as e:  # build failure / OOM / anything
            rec = {**asdict(cfg), "layout": cfg.layout, "ok": False,
                   "error": f"{type(e).__name__}: {e}"[:400]}
            gc.collect()
            torch.cuda.empty_cache()
        rec["wall_s"] = round(time.time() - t0, 1)
        total_gpu += rec["wall_s"]
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        msg = (f"{rec['img_s']:8.1f} img/s  {rec['peak_gb']:5.2f} GB  "
               f"(dino {rec.get('dino_img_s_after')})" if rec.get("ok")
               else f"  FAILED  {rec.get('error','')[:120]}")
        print(f"[{k:9s}] {cfg.name[:46]:46s} {msg}  [{rec['wall_s']}s, cum {total_gpu:.0f}s]",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
