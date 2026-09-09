#!/usr/bin/env python
"""Build the v2 image cache from the raw competition DICOM: physical-mm crop + per-series
percentile normalisation (CPU only).

This is `dev/decode_raw.py` with the v2 rendering contract (`src/rsna/decode.read_series_v2`,
see `reports/public_solutions.md` §2(c) / §4 #1) instead of the v1 squish:

    v1  per-slice min-max -> resize the WHOLE frame to size x size   (data/cache384)
    v2  per-SERIES 1-99 percentile -> centre crop a fixed 140 mm box -> size x size

The unit of work is therefore a SERIES, not a slice: the percentile needs the whole stack.

    <out>/cache_c<crop_mm>_<size>/<split>_series/<StudyUID>/<SeriesUID>/<InstanceUID>.jpg

The tree is mirrored exactly (same `<split>_series` name, same UIDs, filenames still
`<InstanceUID>.jpg`) and the split CSVs are copied in, so `dataset_hr.build_hr_index` and the
slot sampler work on the new root with no code change.

It also writes the per-slice header metadata it already parsed:

    <out>/cache_c<crop_mm>_<size>/slice_meta_<split>.parquet

(one row per slice: geometry, PixelSpacing, Rows/Columns, Laterality tag, FOV) which is what
`dev/laterality_audit.py` consumes — so laterality never costs a second pass over 247 GB.
Series skipped by the resume logic still contribute their metadata via a headers-only read
(`stop_before_pixels`, ~0.5 ms/slice), so the parquet is complete after any run.

Properties (unchanged from decode_raw.py): resumable, atomic writes, per-series try/except,
`nice`d workers, `cv2.setNumThreads(0)`.

Examples:
    # smoke: 8 series into a scratch dir
    .venv/bin/python dev/decode_raw_v2.py data/raw --out /tmp/c2test --limit-series 8 --workers 4

    # the real run
    nohup .venv/bin/python dev/decode_raw_v2.py data/raw --out data \
        --size 384 --crop-mm 140 --pct 1 99 --workers 12 --nice 10 > logs/decode_v2.log 2>&1 &
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rsna.decode import (  # noqa: E402
    DEFAULT_CROP_MM,
    DEFAULT_PCT,
    encode_jpeg,
    read_series_v2,
    slice_meta_v2,
)

SPLITS = ("train", "test")

# Header tags the meta-only path needs (superset of order.ORDER_TAGS).
META_TAGS = [
    (0x0020, 0x0013),  # InstanceNumber
    (0x0020, 0x0032),  # ImagePositionPatient
    (0x0020, 0x0037),  # ImageOrientationPatient
    (0x0028, 0x0010),  # Rows
    (0x0028, 0x0011),  # Columns
    (0x0028, 0x0030),  # PixelSpacing
    (0x0018, 0x1164),  # ImagerPixelSpacing
    (0x0020, 0x0060),  # Laterality
    (0x0020, 0x0062),  # ImageLaterality
]

META_COLS = [
    "split", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "instance_number",
    "rows", "cols", "row_mm", "col_mm", "spacing_source",
    "ipp_x", "ipp_y", "ipp_z", "iop0", "iop1", "iop2", "iop3", "iop4", "iop5",
    "laterality_tag", "fov_row_mm", "fov_col_mm",
]

_CFG: Dict = {}


def cache_dir_name(crop_mm: float, size: int) -> str:
    """`cache_c140_384` — crop in mm, output edge in px, so v1 and v2 caches never collide."""
    return f"cache_c{int(round(float(crop_mm)))}_{int(size)}"


# ------------------------------------------------------------------------------------------
# worker
# ------------------------------------------------------------------------------------------
def _init_worker(cfg: Dict) -> None:
    _CFG.update(cfg)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        if cfg.get("nice"):
            os.nice(int(cfg["nice"]))
    except Exception:
        pass


def _write_atomic(path: str, blob: bytes) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, path)


def _meta_only(path: str) -> Optional[dict]:
    """`slice_meta_v2` from a headers-only read — the resume path, no pixel data touched."""
    try:
        import pydicom

        ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=META_TAGS)
    except Exception:
        return None
    try:
        return slice_meta_v2(ds)
    except Exception:
        return None


def _meta_row(split: str, study: str, series: str, stem: str, m: Optional[dict]) -> List:
    if not m:
        return [split, study, series, stem] + [None] * (len(META_COLS) - 4)
    ipp = m.get("ipp")
    iop = m.get("iop")
    ipp = [None] * 3 if ipp is None else [float(v) for v in ipp]
    iop = [None] * 6 if iop is None else [float(v) for v in iop]
    ps = m.get("pixel_spacing", (None, None))
    fov = m.get("fov_mm", (None, None))
    return [split, study, series, stem, m.get("instance_number"),
            m.get("rows"), m.get("cols"), ps[0], ps[1], m.get("spacing_source"),
            ipp[0], ipp[1], ipp[2], iop[0], iop[1], iop[2], iop[3], iop[4], iop[5],
            m.get("laterality_tag"), fov[0], fov[1]]


def _decode_series(task: Tuple[str, str, str]) -> Dict:
    """Render one series with the v2 contract. Never raises."""
    split, study, series = task
    raw_root, out_root = _CFG["raw_root"], _CFG["out_root"]
    size, crop_mm, pct = _CFG["size"], _CFG["crop_mm"], _CFG["pct"]
    quality, overwrite, want_meta = _CFG["quality"], _CFG["overwrite"], _CFG["meta"]

    res = {"series": series, "n_files": 0, "written": 0, "skipped": 0, "failed": 0,
           "bytes": 0, "errors": [], "meta": [], "series_done": 0, "series_skipped": 0}
    src_dir = os.path.join(raw_root, f"{split}_series", study, series)
    try:
        names = sorted(fn for fn in os.listdir(src_dir) if fn.lower().endswith(".dcm"))
    except Exception as e:
        res["errors"].append(f"{src_dir}\tLISTDIR\t{type(e).__name__}: {e}")
        return res
    res["n_files"] = len(names)
    if not names:
        return res
    stems = [fn[:-4] for fn in names]
    dst_dir = os.path.join(out_root, cache_dir_name(crop_mm, size), f"{split}_series", study, series)

    # --- resume: a series whose JPEG count already matches its DICOM count is done ---
    if not overwrite:
        try:
            have = sum(1 for fn in os.listdir(dst_dir)
                       if fn.endswith(".jpg") and os.path.getsize(os.path.join(dst_dir, fn)) > 0)
        except OSError:
            have = 0
        if have >= len(names):
            res["skipped"] = len(names)
            res["series_skipped"] = 1
            if want_meta:
                for fn, stem in zip(names, stems):
                    res["meta"].append(_meta_row(split, study, series,
                                                 stem, _meta_only(os.path.join(src_dir, fn))))
            return res

    try:
        os.makedirs(dst_dir, exist_ok=True)
    except Exception as e:
        res["errors"].append(f"{dst_dir}\tMKDIR\t{type(e).__name__}: {e}")
        return res

    paths = [os.path.join(src_dir, fn) for fn in names]
    try:
        vol, metas = read_series_v2(paths, size=size, crop_mm=crop_mm, pct=pct, verbose=False)
    except Exception as e:  # a whole-series failure must not kill the run
        res["errors"].append(f"{src_dir}\tSERIES\t{type(e).__name__}: {e}")
        res["failed"] += len(names)
        if want_meta:
            for stem in stems:
                res["meta"].append(_meta_row(split, study, series, stem, None))
        return res

    for i, (stem, m) in enumerate(zip(stems, metas)):
        if want_meta:
            res["meta"].append(_meta_row(split, study, series, stem, m))
        if m is None:  # undecodable slice -> read_series_v2 left zeros; do not write it
            res["failed"] += 1
            res["errors"].append(f"{paths[i]}\tDECODE\tundecodable slice")
            continue
        try:
            blob = encode_jpeg(vol[i], quality)
            if blob is None:
                res["failed"] += 1
                res["errors"].append(f"{paths[i]}\tENCODE\timencode failed")
                continue
            _write_atomic(os.path.join(dst_dir, stem + ".jpg"), blob)
            res["written"] += 1
            res["bytes"] += len(blob)
        except Exception as e:
            res["failed"] += 1
            res["errors"].append(f"{paths[i]}\tERROR\t{type(e).__name__}: {e}")
    res["series_done"] = 1
    return res


# ------------------------------------------------------------------------------------------
# enumeration / plumbing
# ------------------------------------------------------------------------------------------
def enumerate_series(raw_root: str, splits, limit_series: Optional[int]) -> List[Tuple[str, str, str]]:
    tasks: List[Tuple[str, str, str]] = []
    for split in splits:
        base = os.path.join(raw_root, f"{split}_series")
        if not os.path.isdir(base):
            continue
        for study in sorted(os.listdir(base)):
            sdir = os.path.join(base, study)
            if not os.path.isdir(sdir):
                continue
            for series in sorted(os.listdir(sdir)):
                if os.path.isdir(os.path.join(sdir, series)):
                    tasks.append((split, study, series))
                    if limit_series is not None and len(tasks) >= limit_series:
                        return tasks
    return tasks


def copy_csvs(raw_root: str, dst_root: str) -> None:
    """Mirror the split CSVs into the cache root so `build_hr_index` resolves them there."""
    wanted = [fn for fn in os.listdir(raw_root) if fn.lower().endswith(".csv")]
    os.makedirs(dst_root, exist_ok=True)
    for fn in wanted:
        src, dst = os.path.join(raw_root, fn), os.path.join(dst_root, fn)
        if not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"[csv] WARN could not copy {fn}: {e}")
    print(f"[csv] {len(wanted)} CSV(s) in {dst_root}: {sorted(wanted)}" if wanted
          else f"[csv] WARN no CSV found in {raw_root}")


def _fmt(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec // 3600:d}h{(sec % 3600) // 60:02d}m{sec % 60:02d}s"


def _meta_to_parquet(csv_paths: Dict[str, str], out_root: str) -> None:
    """CSV shards (streamed, so the parent never holds 800k rows) -> one parquet per split."""
    try:
        import pandas as pd
    except Exception as e:
        print(f"[meta] WARN pandas unavailable ({e}) — leaving the CSV shards in place")
        return
    for split, path in csv_paths.items():
        try:
            df = pd.read_csv(path)
            dst = os.path.join(out_root, f"slice_meta_{split}.parquet")
            df.to_parquet(dst, index=False)
            print(f"[meta] {split}: {len(df)} slice rows -> {dst}")
            os.remove(path)
        except Exception as e:
            print(f"[meta] WARN {split}: {type(e).__name__}: {e} (CSV kept at {path})")


# ------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_root", help="dir containing train_series/ and/or test_series/")
    ap.add_argument("--out", default="data", help="cache_c<mm>_<size>/ is created here")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--crop-mm", type=float, default=DEFAULT_CROP_MM)
    ap.add_argument("--pct", type=float, nargs=2, default=list(DEFAULT_PCT),
                    metavar=("LO", "HI"), help="per-series percentiles (default 1 99)")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--limit-series", type=int, default=None, help="stop after N series (smoke)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-meta", action="store_true", help="skip the slice-meta parquet")
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--log", default=None, help="failure log (default <cache_root>/decode_v2_failures.log)")
    ap.add_argument("--chunksize", type=int, default=None)
    ap.add_argument("--progress-every", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw_root, out_root = os.path.abspath(args.raw_root), os.path.abspath(args.out)
    if not os.path.isdir(raw_root):
        print(f"ERROR: raw_root not found: {raw_root}")
        return 2
    cache_root = os.path.join(out_root, cache_dir_name(args.crop_mm, args.size))
    log_path = args.log or os.path.join(cache_root, "decode_v2_failures.log")
    pct = (float(args.pct[0]), float(args.pct[1]))

    print(f"=== decode_raw_v2 start {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"[cfg] raw_root={raw_root}")
    print(f"[cfg] cache_root={cache_root}  size={args.size}  crop_mm={args.crop_mm}  pct={pct}")
    print(f"[cfg] splits={args.splits}  workers={args.workers}  nice={args.nice}  "
          f"quality={args.quality}  meta={not args.no_meta}  overwrite={args.overwrite}")

    t_enum = time.time()
    tasks = enumerate_series(raw_root, args.splits, args.limit_series)
    print(f"[enum] {len(tasks)} series in {time.time() - t_enum:.1f}s")
    if not tasks:
        print("ERROR: nothing to do")
        return 2
    if args.dry_run:
        return 0

    os.makedirs(cache_root, exist_ok=True)
    copy_csvs(raw_root, cache_root)

    cfg = {"raw_root": raw_root, "out_root": out_root, "size": int(args.size),
           "crop_mm": float(args.crop_mm), "pct": pct, "quality": int(args.quality),
           "overwrite": bool(args.overwrite), "nice": args.nice, "meta": not args.no_meta}

    tot = {"n_files": 0, "written": 0, "skipped": 0, "failed": 0, "bytes": 0,
           "series_done": 0, "series_skipped": 0}
    n_series = 0
    t0 = t_last = time.time()
    logf = open(log_path, "a")
    logf.write(f"# decode_raw_v2 {time.strftime('%Y-%m-%d %H:%M:%S')} size={args.size} "
               f"crop_mm={args.crop_mm} pct={pct}\n")

    meta_files: Dict[str, object] = {}
    meta_writers: Dict[str, object] = {}
    meta_paths: Dict[str, str] = {}
    if not args.no_meta:
        for split in args.splits:
            p = os.path.join(cache_root, f"slice_meta_{split}.csv")
            meta_paths[split] = p
            meta_files[split] = open(p, "w", newline="")
            meta_writers[split] = csv.writer(meta_files[split])
            meta_writers[split].writerow(META_COLS)

    nproc = max(1, args.workers)
    chunksize = args.chunksize or max(1, min(4, len(tasks) // (nproc * 4)))
    print(f"[pool] {nproc} workers, chunksize {chunksize}", flush=True)
    ctx = mp.get_context("fork")
    pool = ctx.Pool(processes=nproc, initializer=_init_worker, initargs=(cfg,))
    try:
        for res in pool.imap_unordered(_decode_series, tasks, chunksize=chunksize):
            n_series += 1
            for k in tot:
                tot[k] += res[k]
            for line in res["errors"]:
                logf.write(line + "\n")
            if res["errors"]:
                logf.flush()
            for row in res["meta"]:
                w = meta_writers.get(row[0])
                if w is not None:
                    w.writerow(row)
            now = time.time()
            if now - t_last >= args.progress_every or n_series == len(tasks):
                el = now - t0
                frac = n_series / len(tasks)
                print(f"[{_fmt(el)}] series {n_series}/{len(tasks)} ({100*frac:.1f}%) | "
                      f"slices {tot['n_files']} ({tot['n_files']/max(el,1e-9):.1f}/s) | "
                      f"written {tot['written']} skipped {tot['skipped']} failed {tot['failed']} | "
                      f"{tot['bytes']/1e9:.2f} GB | ETA {_fmt(el*(1-frac)/max(frac,1e-9))}", flush=True)
                t_last = now
        pool.close()
    except KeyboardInterrupt:
        print("\n[interrupt] terminating workers (resumable — re-run the same command)")
        pool.terminate()
    finally:
        pool.join()
        logf.close()
        for f in meta_files.values():
            f.close()

    if meta_paths:
        _meta_to_parquet(meta_paths, cache_root)

    el = time.time() - t0
    print("--- summary ---")
    print(f"series processed : {n_series}/{len(tasks)}  (decoded {tot['series_done']}, "
          f"resume-skipped {tot['series_skipped']})")
    print(f"slices seen      : {tot['n_files']}")
    print(f"jpegs written    : {tot['written']}")
    print(f"jpegs skipped    : {tot['skipped']} (already present)")
    print(f"failures         : {tot['failed']}   -> {log_path}")
    print(f"bytes written    : {tot['bytes']/1e9:.2f} GB")
    print(f"elapsed          : {_fmt(el)}  ({tot['n_files']/max(el,1e-9):.1f} slices/s)")
    print(f"cache root       : {cache_root}")
    print(f"=== decode_raw_v2 done {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
