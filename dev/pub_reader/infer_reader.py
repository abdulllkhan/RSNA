"""Run goodpjw2008's public 2.5D ConvNeXt-T reader on OUR raw DICOMs, one prediction table per checkpoint.

Exactly the released inference (models/public/gpjw_reader/infer.py, via the verbatim port gpjw_port.py):
  * canonical volume per series: 0.4 mm/px, 384 px (153.6 mm FOV) centre crop / zero pad, in-plane orientation
    from ImageOrientationPatient (patient LPS frame, no laterality flip), slices sorted along the normal,
    <= 64 slices (even subsampling), per-series 1-99.5 percentile window -> uint8;
  * 6 slots = plane x Fat_Suppression (CSV), per slot the candidate with most files (capped at 64; ties by CSV
    order), slots without a series are masked;
  * k = 12 triplets [j-1, j, j+1] per series, centres evenly over 4-96 % of the stack;
  * centred 0.92 FOV resized to 256 px, ImageNet norm, ConvNeXt-T -> 2-layer study transformer -> 12 attention
    pools; fp16 autocast; sigmoid.
Variants (--variants):
  k12   = the released infer.py (PRIMARY, the configuration the author's own OOF used: k_eval = 12);
  k16fb = the author's 0.954 notebook (k = 16 + a missing fat-sat/non-fat-sat slot filled from its counterpart).
Both variants share one decode per series.

Writes <out>/pred_<variant>_fold<f>.parquet (index StudyInstanceUID, our 12 columns in our order, sigmoid
probabilities of checkpoint cnxt_v0_fold<f>.pt alone) and <out>/meta.parquet (per-study decode seconds,
slots present, slices decoded) and <out>/timing.json.

Usage (repo root; GPU job -> under the box lock):
  flock -w 14400 logs/box.lock env PYTHONPATH=$PWD nice -n 10 .venv/bin/python -P dev/pub_reader/infer_reader.py \
      --out outputs/pub_reader/run_all --variants k12,k16fb --workers 12
  timing sample:  ... --out outputs/pub_reader/timing --sample 40 --workers 0 --bs 1 --variants k12
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dev.pub_reader import gpjw_port as G

ROOT = Path(__file__).resolve().parents[2]
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
VARIANTS = {"k12": (12, G.build_study_table), "k16fb": (16, G.build_study_table_fallback)}


def mem_avail_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1048576
    return float("nan")


class ReaderStudies(torch.utils.data.Dataset):
    """One item = one study, windows for every requested variant, each chosen series decoded once."""

    def __init__(self, studies, root, tables, variants):
        self.studies, self.root, self.tables, self.variants = list(studies), root, tables, variants

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        t0 = time.perf_counter()
        memo = {}

        def vol_of(sr):
            if sr not in memo:
                try:
                    memo[sr] = G.load_series(f"{self.root}/{st}/{sr}")[0]
                except Exception as e:  # remembered so a second variant fails the same way, never re-decodes
                    memo[sr] = e
            v = memo[sr]
            if isinstance(v, Exception):
                raise v
            return v

        out = []
        for v in self.variants:
            k, _ = VARIANTS[v]
            slots = self.tables[v].get(st, [[] for _ in range(G.N_SLOTS)])
            x, mask, pos = G.study_windows(vol_of, st, slots, k)
            out.append((torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos)))
        dt = time.perf_counter() - t0
        n_sl = sum(len(v) for v in memo.values() if not isinstance(v, Exception))
        n_err = sum(isinstance(v, Exception) for v in memo.values())
        return st, out, dt, len(memo), n_sl, n_err


def collate(items):
    sts = [it[0] for it in items]
    per_v = []
    for vi in range(len(items[0][1])):
        per_v.append(tuple(torch.stack([it[1][vi][j] for it in items]) for j in range(3)))
    meta = [(it[0], it[2], it[3], it[4], it[5]) for it in items]
    return sts, per_v, meta


def worker_init(_):
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="k12")
    ap.add_argument("--series-csv", default=str(ROOT / "data/comp_csv/train_series.csv"))
    ap.add_argument("--study-csv", default=str(ROOT / "data/comp_csv/train.csv"))
    ap.add_argument("--root", default=str(ROOT / "data/raw/train_series"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "models/public/gpjw_reader"))
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--sample", type=int, default=0, help="random subset of this many studies (timing)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--min-avail-gb", type=float, default=60.0)
    ap.add_argument("--floor-gb", type=float, default=20.0, help="abort if MemAvailable falls below this")
    ap.add_argument("--save-every", type=int, default=400)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    avail = mem_avail_gb()
    print(f"[mem] MemAvailable {avail:.1f} GB (need >= {a.min_avail_gb})", flush=True)
    if avail < a.min_avail_gb:
        raise SystemExit(f"MemAvailable {avail:.1f} GB < {a.min_avail_gb} GB: refusing to start")
    stop = threading.Event()

    def watchdog():
        strikes = 0
        while not stop.wait(10):
            strikes = strikes + 1 if mem_avail_gb() < a.floor_gb else 0
            if strikes >= 2:
                print(f"[mem] WATCHDOG: MemAvailable < {a.floor_gb} GB twice -> aborting", flush=True)
                os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    assert all(v in VARIANTS for v in variants), variants
    tr = pd.read_csv(a.study_csv)
    assert G.LABELS == T, "reader label order differs from ours"
    if set(T) <= set(tr.columns):
        assert list(tr.columns[2:14]) == T, "train.csv label columns differ from ours"
    studies = tr.StudyInstanceUID.tolist()
    if a.sample:
        studies = pd.Series(studies).sample(a.sample, random_state=a.seed).tolist()
    se = pd.read_csv(a.series_csv)
    se = se[se.StudyInstanceUID.isin(set(studies))]
    t0 = time.time()
    n_files = G.series_file_counts(se, a.root)
    tables = {v: VARIANTS[v][1](se, n_files) for v in variants}
    print(f"[plan] {len(studies)} studies, {len(se)} series, file counts in {time.time() - t0:.1f}s; "
          f"variants {variants}", flush=True)

    folds = [int(f) for f in a.folds.split(",")]
    models = G.load_models([f"{a.ckpt_dir}/cnxt_v0_fold{f}.pt" for f in folds])
    for f, (_, res, args) in zip(folds, models):
        assert args["fold"] == f and res == 256 and args["k_eval"] == 12, args
        print(f"[model] fold{f}: {args['backbone']} res {res} labels {args['labels']} epochs {args['epochs']}",
              flush=True)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):  # warm-up, excluded from the timing
        kw = VARIANTS[variants[0]][0]
        xw = torch.zeros((1, G.N_SLOTS, kw, 3, G.SIZE, G.SIZE), dtype=torch.uint8, device="cuda")
        mw, pw = torch.ones((1, G.N_SLOTS), dtype=torch.bool, device="cuda"), torch.zeros((1, G.N_SLOTS, kw), device="cuda")
        for m, res, _ in models:
            m(xw, mw, pw, res=res)
        torch.cuda.synchronize()
        del xw
    ds = ReaderStudies(studies, a.root, tables, variants)
    dl_kw = dict(batch_size=a.bs, num_workers=a.workers, collate_fn=collate, pin_memory=True)
    if a.workers:
        dl_kw.update(worker_init_fn=worker_init, prefetch_factor=2)
    dl = torch.utils.data.DataLoader(ds, **dl_kw)

    preds = {(v, f): [] for v in variants for f in folds}
    ids, meta_rows = [], []
    gpu_s = {(v, f): 0.0 for v in variants for f in folds}
    n_img = {v: 0 for v in variants}
    n_slots = {v: 0 for v in variants}
    wait_s, t_start, t_last = 0.0, time.time(), time.time()
    it = iter(dl)
    done = 0
    while True:
        tw = time.perf_counter()
        try:
            sts, per_v, meta = next(it)
        except StopIteration:
            break
        wait_s += time.perf_counter() - tw
        for vi, v in enumerate(variants):
            x, mask, pos = per_v[vi]
            x, mask, pos = x.cuda(non_blocking=True), mask.cuda(), pos.cuda()
            n_img[v] += int(mask.sum()) * x.shape[2]
            n_slots[v] += int(mask.sum())
            for f, (m, res, _) in zip(folds, models):
                torch.cuda.synchronize()
                tg = time.perf_counter()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    p = m(x, mask, pos, res=res).float().sigmoid()
                torch.cuda.synchronize()
                gpu_s[(v, f)] += time.perf_counter() - tg
                preds[(v, f)].append(p.cpu().numpy())
        ids.extend(sts)
        meta_rows.extend(meta)
        done += len(sts)
        if time.time() - t_last > 60 or done == len(studies):
            el = time.time() - t_start
            print(f"[run] {done}/{len(studies)} studies {el:.0f}s ({done / max(el, 1e-9):.2f} st/s) "
                  f"dl-wait {wait_s:.0f}s mem {mem_avail_gb():.0f} GB", flush=True)
            t_last = time.time()
        if a.save_every and done % a.save_every < a.bs and done < len(studies):
            save(out, variants, folds, preds, ids, meta_rows, partial=True)
    stop.set()
    save(out, variants, folds, preds, ids, meta_rows, partial=False)
    wall = time.time() - t_start
    meta = pd.DataFrame(meta_rows, columns=["StudyInstanceUID", "decode_s", "n_series", "n_slices", "n_err"])
    timing = {
        "studies": len(ids), "workers": a.workers, "bs": a.bs, "variants": variants, "folds": folds,
        "wall_s": wall, "dataloader_wait_s": wait_s,
        "gpu_s": {f"{v}_fold{f}": s for (v, f), s in gpu_s.items()},
        "gpu_s_per_study_per_ckpt": {v: float(np.mean([gpu_s[(v, f)] for f in folds]) / max(len(ids), 1))
                                     for v in variants},
        "images_per_variant": n_img,
        "img_per_s_per_ckpt": {v: n_img[v] / max(np.mean([gpu_s[(v, f)] for f in folds]), 1e-9) for v in variants},
        "decode_s_per_study_mean": float(meta.decode_s.mean()), "decode_s_per_study_median": float(meta.decode_s.median()),
        "series_per_study_mean": float(meta.n_series.mean()), "slices_per_study_mean": float(meta.n_slices.mean()),
        "series_errors": int(meta.n_err.sum()),
        "slots_present_mean": {v: n_slots[v] / max(len(ids), 1) for v in variants},
    }
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    print("[timing]", json.dumps(timing), flush=True)


def save(out, variants, folds, preds, ids, meta_rows, partial):
    suffix = "_partial" if partial else ""
    idx = pd.Index(ids, name="StudyInstanceUID")
    for v in variants:
        for f in folds:
            arr = np.concatenate(preds[(v, f)]) if preds[(v, f)] else np.zeros((0, 12), np.float32)
            pd.DataFrame(arr.astype(np.float32), index=idx, columns=T).to_parquet(out / f"pred_{v}_fold{f}{suffix}.parquet")
    pd.DataFrame(meta_rows, columns=["StudyInstanceUID", "decode_s", "n_series", "n_slices", "n_err"]) \
        .set_index("StudyInstanceUID").to_parquet(out / f"meta{suffix}.parquet")
    if not partial:
        for p in out.glob("*_partial.parquet"):
            p.unlink()
    print(f"[save] {len(ids)} studies -> {out} ({'partial' if partial else 'final'})", flush=True)


if __name__ == "__main__":
    main()
