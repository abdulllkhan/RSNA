#!/usr/bin/env python
"""Build higher-resolution image caches from the raw competition DICOM (CPU only).

Walks `<raw_root>/{train,test}_series/<StudyUID>/<SeriesUID>/<InstanceUID>.dcm`, decodes
every slice ONCE with the validated recipe (`src/rsna/decode.read_dicom_slice` — the same
function inference uses, so train/infer stay consistent), and writes one grayscale JPEG per
slice per requested size:

    <out>/cache<size>/<split>_series/<StudyUID>/<SeriesUID>/<InstanceUID>.jpg

The tree is mirrored exactly (same UIDs, same `<split>_series` directory name, filenames
still `<InstanceUID>.<ext>`), so `dataset.build_study_index` + the slot sampler work on a
cache root unchanged — point them at `<out>/cache384` instead of `data/processed_jpeg224`.
The split CSVs are copied into each cache root for the same reason.

Properties:
  * one DICOM read per slice, resized to every requested size (I/O is the bottleneck);
  * resumable — an existing non-empty output is skipped, so re-running after an interrupt
    costs only a stat() per file;
  * robust — a bad slice is logged and skipped, never aborts the run;
  * atomic — JPEGs are written to a temp file then renamed, so a kill leaves no truncated
    file that a resume would treat as done;
  * polite — workers run at low priority (`--nice`) and OpenCV threading is off, so it can
    share the box; still: do NOT run it while GPU training is using the page cache.

Examples:
    # smoke test (200 slices, both sizes) into a scratch dir
    .venv/bin/python dev/decode_raw.py /tmp/dicom_mount --out /tmp/cachetest \
        --sizes 384 512 --workers 6 --limit 200

    # the real run
    .venv/bin/python dev/decode_raw.py data/raw --out data --sizes 384 512 --workers 6
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rsna.decode import encode_jpeg, read_dicom_slices  # noqa: E402

SPLITS = ("train", "test")

# Set once per worker by _init_worker.
_CFG: Dict = {}


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


def _decode_series(task: Tuple[str, str, str, Optional[int]]) -> Dict:
    """Decode every slice of one series at every requested size. Never raises."""
    split, study, series, max_files = task
    raw_root: str = _CFG["raw_root"]
    out_root: str = _CFG["out_root"]
    sizes: List[int] = _CFG["sizes"]
    quality: int = _CFG["quality"]
    overwrite: bool = _CFG["overwrite"]

    res = {"series": series, "n_files": 0, "written": 0, "skipped": 0, "failed": 0,
           "bytes": 0, "errors": []}
    src_dir = os.path.join(raw_root, f"{split}_series", study, series)
    try:
        names = sorted(fn for fn in os.listdir(src_dir) if fn.lower().endswith(".dcm"))
    except Exception as e:
        res["errors"].append(f"{src_dir}\tLISTDIR\t{type(e).__name__}: {e}")
        return res
    if max_files is not None:
        names = names[:max_files]
    res["n_files"] = len(names)
    if not names:
        return res

    dst_dirs = {}
    for size in sizes:
        d = os.path.join(out_root, f"cache{size}", f"{split}_series", study, series)
        try:
            os.makedirs(d, exist_ok=True)
        except Exception as e:
            res["errors"].append(f"{d}\tMKDIR\t{type(e).__name__}: {e}")
        dst_dirs[size] = d

    for fn in names:
        stem = fn[:-4]
        todo = []
        for size in sizes:
            dst = os.path.join(dst_dirs[size], stem + ".jpg")
            if not overwrite:
                try:
                    if os.path.getsize(dst) > 0:
                        continue
                except OSError:
                    pass
            todo.append((size, dst))
        if not todo:
            res["skipped"] += len(sizes)
            continue

        src = os.path.join(src_dir, fn)
        try:  # ONE parse+normalise shared by every requested size (the dominant cost)
            imgs = read_dicom_slices(src, [s for s, _ in todo], verbose=False)
        except Exception as e:
            imgs = {}
            res["errors"].append(f"{src}\tERROR\t{type(e).__name__}: {e}")
        for size, dst in todo:
            try:
                img = imgs.get(size)
                if img is None:
                    res["failed"] += 1
                    res["errors"].append(f"{src}\tDECODE\tNone at size {size}")
                    continue
                blob = encode_jpeg(img, quality)
                if blob is None:
                    res["failed"] += 1
                    res["errors"].append(f"{src}\tENCODE\timencode failed at size {size}")
                    continue
                _write_atomic(dst, blob)
                res["written"] += 1
                res["bytes"] += len(blob)
            except Exception as e:  # never let one bad slice kill the run
                res["failed"] += 1
                res["errors"].append(f"{src}\tERROR\t{type(e).__name__}: {e}")
        res["skipped"] += len(sizes) - len(todo)
    return res


# ------------------------------------------------------------------------------------------
# enumeration
# ------------------------------------------------------------------------------------------
def enumerate_series(raw_root: str, splits, limit: Optional[int]) -> List[Tuple[str, str, str, Optional[int]]]:
    """List (split, study, series, max_files) tasks. Only dirs are listed (cheap) unless
    --limit is given, in which case files are counted so the cap is exact."""
    tasks: List[Tuple[str, str, str, Optional[int]]] = []
    budget = limit
    for split in splits:
        base = os.path.join(raw_root, f"{split}_series")
        if not os.path.isdir(base):
            continue
        for study in sorted(os.listdir(base)):
            sdir = os.path.join(base, study)
            if not os.path.isdir(sdir):
                continue
            for series in sorted(os.listdir(sdir)):
                if not os.path.isdir(os.path.join(sdir, series)):
                    continue
                if budget is None:
                    tasks.append((split, study, series, None))
                    continue
                if budget <= 0:
                    return tasks
                try:
                    n = sum(1 for fn in os.listdir(os.path.join(sdir, series))
                            if fn.lower().endswith(".dcm"))
                except OSError:
                    n = 0
                if n == 0:
                    continue
                take = min(n, budget)
                tasks.append((split, study, series, take))
                budget -= take
    return tasks


def copy_csvs(raw_root: str, out_root: str, sizes) -> None:
    """Mirror the split CSVs into every cache root so build_study_index works there."""
    wanted = [fn for fn in os.listdir(raw_root) if fn.lower().endswith(".csv")]
    for size in sizes:
        dst_root = os.path.join(out_root, f"cache{size}")
        os.makedirs(dst_root, exist_ok=True)
        for fn in wanted:
            src, dst = os.path.join(raw_root, fn), os.path.join(dst_root, fn)
            if not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"[csv] WARN could not copy {fn}: {e}")
    if wanted:
        print(f"[csv] copied {len(wanted)} CSV(s) into {len(sizes)} cache root(s): {sorted(wanted)}")
    else:
        print(f"[csv] WARN no CSV found in {raw_root}")


def _fmt(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec // 3600:d}h{(sec % 3600) // 60:02d}m{sec % 60:02d}s"


# ------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_root", help="dir containing train_series/ and/or test_series/")
    ap.add_argument("--out", default="data", help="cache<size>/ dirs are created here (default: data)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[384, 512])
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N slices (smoke test)")
    ap.add_argument("--overwrite", action="store_true", help="re-decode even if output exists")
    ap.add_argument("--nice", type=int, default=10, help="worker niceness (default 10)")
    ap.add_argument("--log", default=None, help="failure log (default <out>/decode_raw_failures.log)")
    ap.add_argument("--chunksize", type=int, default=None,
                    help="pool chunksize (default: adaptive, 1..4 by task count)")
    ap.add_argument("--progress-every", type=float, default=30.0, help="seconds between progress lines")
    ap.add_argument("--dry-run", action="store_true", help="enumerate and exit")
    args = ap.parse_args()

    raw_root = os.path.abspath(args.raw_root)
    out_root = os.path.abspath(args.out)
    if not os.path.isdir(raw_root):
        print(f"ERROR: raw_root not found: {raw_root}")
        return 2
    sizes = sorted(set(int(s) for s in args.sizes))
    log_path = args.log or os.path.join(out_root, "decode_raw_failures.log")

    print(f"=== decode_raw start {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"[cfg] raw_root={raw_root}")
    print(f"[cfg] out_root={out_root}  sizes={sizes}  quality={args.quality}")
    print(f"[cfg] splits={args.splits}  workers={args.workers}  nice={args.nice}  "
          f"limit={args.limit}  overwrite={args.overwrite}")

    t_enum = time.time()
    tasks = enumerate_series(raw_root, args.splits, args.limit)
    print(f"[enum] {len(tasks)} series in {time.time() - t_enum:.1f}s")
    if not tasks:
        print("ERROR: nothing to do (no <split>_series/<study>/<series> dirs found)")
        return 2
    if args.dry_run:
        return 0

    os.makedirs(out_root, exist_ok=True)
    copy_csvs(raw_root, out_root, sizes)

    cfg = {"raw_root": raw_root, "out_root": out_root, "sizes": sizes,
           "quality": args.quality, "overwrite": args.overwrite, "nice": args.nice}

    tot = {"n_files": 0, "written": 0, "skipped": 0, "failed": 0, "bytes": 0}
    n_series = 0
    t0 = time.time()
    t_last = t0
    logf = open(log_path, "a")
    logf.write(f"# decode_raw {time.strftime('%Y-%m-%d %H:%M:%S')} raw_root={raw_root} sizes={sizes}\n")

    nproc = max(1, args.workers)
    # Small task lists must not be chunked, or some workers sit idle.
    chunksize = args.chunksize or max(1, min(4, len(tasks) // (nproc * 4)))
    print(f"[pool] {nproc} workers, chunksize {chunksize}")
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
            now = time.time()
            if now - t_last >= args.progress_every or n_series == len(tasks):
                el = now - t0
                done_img = tot["written"] + tot["skipped"]
                rate_sl = tot["n_files"] / max(el, 1e-9)
                frac = n_series / len(tasks)
                eta = el * (1 - frac) / max(frac, 1e-9)
                print(f"[{_fmt(el)}] series {n_series}/{len(tasks)} ({100*frac:.1f}%) | "
                      f"slices {tot['n_files']} ({rate_sl:.1f}/s) | jpegs written {tot['written']} "
                      f"skipped {tot['skipped']} failed {tot['failed']} | "
                      f"{tot['bytes']/1e9:.2f} GB | ETA {_fmt(eta)}", flush=True)
                t_last = now
        pool.close()
    except KeyboardInterrupt:
        print("\n[interrupt] terminating workers (run is resumable — just re-run)")
        pool.terminate()
    finally:
        pool.join()
        logf.close()

    el = time.time() - t0
    per_size = tot["written"] / max(len(sizes), 1)
    print("--- summary ---")
    print(f"series processed : {n_series}/{len(tasks)}")
    print(f"slices seen      : {tot['n_files']}")
    print(f"jpegs written    : {tot['written']}  (~{per_size:.0f} per size)")
    print(f"jpegs skipped    : {tot['skipped']} (already present)")
    print(f"failures         : {tot['failed']}   -> {log_path}")
    print(f"bytes written    : {tot['bytes']/1e9:.2f} GB")
    print(f"elapsed          : {_fmt(el)}  ({tot['n_files']/max(el,1e-9):.1f} slices/s "
          f"read, {tot['written']/max(el,1e-9):.1f} jpegs/s written)")
    for size in sizes:
        print(f"cache root       : {os.path.join(out_root, f'cache{size}')}")
    print(f"=== decode_raw done {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
