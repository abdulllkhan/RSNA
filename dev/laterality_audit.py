#!/usr/bin/env python
"""Decide (and audit) which knee each study images, from DICOM headers only.

Produces `data/study_side.parquet` — `StudyInstanceUID, side, source, x_centre_mm, tag_side` —
the table every v2 dataset/inference path reads to canonicalise a study to a RIGHT knee
(`src/rsna/laterality.canonical_ops`). See `reports/public_solutions.md` §4 #3: four of the
twelve targets are side-specific, `Laterality` is empty on most studies, and the geometry rule
only works with the image-CENTRE correction (`sadamtorres`: 96.9 % vs 58.8 % agreement with
the populated tag, with vs without it).

It prints exactly the numbers that decide whether the rule is safe to apply:

  * tag coverage — % of studies with a populated, self-consistent Laterality/ImageLaterality;
  * tag-vs-geometry agreement, centre-corrected AND with the naive corner `IPP[0]`, each
    reported for BOTH sign conventions (+x = patient left, the LPS standard, and its inverse)
    so the convention is measured rather than assumed;
  * how many studies land in the |x| < dead-zone band (where a no-tag study defaults to 'R');
  * the L/R distribution.

INPUT — no second pass over the 247 GB of DICOM. It reads the per-slice header table that
`dev/decode_raw_v2.py` writes while it decodes (`<cache>/slice_meta_<split>.parquet`). With
`--from-dicom` (or when that file is missing) it falls back to a headers-only walk of the raw
tree (`stop_before_pixels` + `specific_tags`, ~0.5 ms/slice), which is also how the hidden
test set would be handled at inference time.

Examples:
    .venv/bin/python dev/laterality_audit.py                       # both splits, from the cache meta
    .venv/bin/python dev/laterality_audit.py --splits test --from-dicom --workers 4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decode_raw_v2 import META_COLS, _meta_only, _meta_row, cache_dir_name  # noqa: E402
from src.rsna import laterality as lat  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------------------------------
# metadata sources
# ------------------------------------------------------------------------------------------
def meta_from_parquet(cache_root: str, split: str):
    import pandas as pd

    for ext in ("parquet", "csv"):
        p = os.path.join(cache_root, f"slice_meta_{split}.{ext}")
        if os.path.exists(p):
            df = pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
            if len(df):
                print(f"[meta] {split}: {len(df)} slice rows from {p}")
                return df
    return None


def _series_meta_rows(job: Tuple[str, str, str, str]) -> List[List]:
    split, study, series, sdir = job
    rows = []
    try:
        names = sorted(fn for fn in os.listdir(sdir) if fn.lower().endswith(".dcm"))
    except OSError:
        return rows
    for fn in names:
        rows.append(_meta_row(split, study, series, fn[:-4], _meta_only(os.path.join(sdir, fn))))
    return rows


def meta_from_dicom(raw_root: str, split: str, workers: int, limit: Optional[int]):
    """Headers-only walk of `<raw_root>/<split>_series` — the fallback / inference path."""
    import pandas as pd

    base = os.path.join(raw_root, f"{split}_series")
    if not os.path.isdir(base):
        print(f"[meta] {split}: {base} not found")
        return None
    jobs = []
    for study in sorted(os.listdir(base)):
        sdir = os.path.join(base, study)
        if not os.path.isdir(sdir):
            continue
        for series in sorted(os.listdir(sdir)):
            d = os.path.join(sdir, series)
            if os.path.isdir(d):
                jobs.append((split, study, series, d))
        if limit is not None and len({j[1] for j in jobs}) >= limit:
            break
    print(f"[meta] {split}: reading headers of {len(jobs)} series with {workers} workers ...", flush=True)
    t0 = time.time()
    rows: List[List] = []
    if workers <= 1:
        for j in jobs:
            rows += _series_meta_rows(j)
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            for r in pool.imap_unordered(_series_meta_rows, jobs, chunksize=8):
                rows += r
    print(f"[meta] {split}: {len(rows)} slice rows in {time.time()-t0:.1f}s")
    return pd.DataFrame(rows, columns=META_COLS) if rows else None


# ------------------------------------------------------------------------------------------
# per-study decision
# ------------------------------------------------------------------------------------------
def _row_to_meta(r) -> dict:
    """One dataframe row -> the `decode.slice_meta_v2` dict `laterality.py` consumes."""
    def _f(v):
        try:
            v = float(v)
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    ipp = [_f(r.ipp_x), _f(r.ipp_y), _f(r.ipp_z)]
    iop = [_f(getattr(r, f"iop{i}")) for i in range(6)]
    tag = r.laterality_tag
    tag = None if tag is None or (isinstance(tag, float) and np.isnan(tag)) else str(tag)
    return {
        "ipp": None if any(v is None for v in ipp) else np.asarray(ipp, dtype=np.float64),
        "iop": None if any(v is None for v in iop) else np.asarray(iop, dtype=np.float64),
        "pixel_spacing": (_f(r.row_mm) or 1.0, _f(r.col_mm) or 1.0),
        "rows": int(_f(r.rows) or 0),
        "cols": int(_f(r.cols) or 0),
        "laterality_tag": tag if tag in ("L", "R") else None,
    }


def audit(df, dead_zone_mm: float) -> Tuple[List[dict], Dict]:
    """One row per study + the aggregate counters printed by `report`."""
    out: List[dict] = []
    agg = {"n": 0, "tagged": 0, "tag_conflict": 0, "no_geom": 0, "dead_zone": 0,
           "agree_centre": 0, "agree_corner": 0, "cmp": 0, "L": 0, "R": 0,
           "src": {"tag": 0, "geometry": 0, "default": 0}}
    for study, g in df.groupby("StudyInstanceUID", sort=True):
        series_metas = [[_row_to_meta(r) for r in gs.itertuples()]
                        for _, gs in g.groupby("SeriesInstanceUID", sort=True)]
        side, source, x_centre = lat.side_from_series_meta(series_metas, dead_zone_mm=dead_zone_mm)
        tside, n_tag, consistent = lat.tag_side(series_metas)
        x_corner = lat.study_x_centre(series_metas, corner=True)

        agg["n"] += 1
        agg["src"][source] += 1
        agg[side] += 1
        if tside is not None:
            agg["tagged"] += 1
            if not consistent:
                agg["tag_conflict"] += 1
        if not np.isfinite(x_centre):
            agg["no_geom"] += 1
        elif abs(x_centre) < dead_zone_mm:
            agg["dead_zone"] += 1
        if tside is not None and consistent:
            gc = lat.side_from_x(x_centre, dead_zone_mm=0.0)      # sign only, no dead zone,
            gn = lat.side_from_x(x_corner, dead_zone_mm=0.0)      # so the comparison is fair
            if gc is not None and gn is not None:
                agg["cmp"] += 1
                agg["agree_centre"] += int(gc == tside)
                agg["agree_corner"] += int(gn == tside)
        out.append({"StudyInstanceUID": study, "side": side, "source": source,
                    "x_centre_mm": float(x_centre), "tag_side": tside if consistent else None})
    return out, agg


def report(split: str, agg: Dict, dead_zone_mm: float) -> None:
    n, c = max(agg["n"], 1), max(agg["cmp"], 1)
    pc = 100.0 * agg["agree_centre"] / c
    pn = 100.0 * agg["agree_corner"] / c
    print(f"\n--- laterality audit: {split} ({agg['n']} studies) ---")
    print(f"tag coverage           : {agg['tagged']}/{agg['n']} = {100*agg['tagged']/n:.1f}% "
          f"(inconsistent within study: {agg['tag_conflict']})")
    print(f"studies comparable     : {agg['cmp']} (consistent tag AND usable geometry)")
    print(f"agreement, CENTRE-corrected, +x = patient LEFT : {agg['agree_centre']}/{agg['cmp']} = {pc:.1f}%")
    print(f"agreement, CENTRE-corrected, +x = patient RIGHT: {agg['cmp']-agg['agree_centre']}/{agg['cmp']} = {100-pc:.1f}%")
    print(f"agreement, naive corner IPP[0], +x = LEFT      : {agg['agree_corner']}/{agg['cmp']} = {pn:.1f}%")
    print(f"agreement, naive corner IPP[0], +x = RIGHT     : {agg['cmp']-agg['agree_corner']}/{agg['cmp']} = {100-pn:.1f}%")
    print(f"centre correction is worth                     : {pc-pn:+.1f} pp")
    print(f"dead zone |x| < {dead_zone_mm:g} mm : {agg['dead_zone']} studies "
          f"({100*agg['dead_zone']/n:.1f}%);  no usable geometry: {agg['no_geom']}")
    print(f"side distribution      : L {agg['L']} ({100*agg['L']/n:.1f}%)  R {agg['R']} ({100*agg['R']/n:.1f}%)")
    print(f"decided by             : tag {agg['src']['tag']}  geometry {agg['src']['geometry']}  "
          f"default(R) {agg['src']['default']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-root", default=None,
                    help="cache root holding slice_meta_<split>.parquet (default data/cache_c140_384)")
    ap.add_argument("--raw-root", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    ap.add_argument("--dead-zone-mm", type=float, default=lat.DEAD_ZONE_MM)
    ap.add_argument("--from-dicom", action="store_true", help="ignore the cache meta, read headers")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-studies", type=int, default=None, help="--from-dicom smoke cap")
    args = ap.parse_args()

    cache_root = args.cache_root or os.path.join(ROOT, "data", cache_dir_name(140.0, 384))
    print(f"=== laterality_audit {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"[cfg] cache_root={cache_root}  raw_root={args.raw_root}  "
          f"dead_zone={args.dead_zone_mm} mm  LEFT_IS_POSITIVE_X={lat.LEFT_IS_POSITIVE_X}")

    rc = 0
    for split in args.splits:
        df = None if args.from_dicom else meta_from_parquet(cache_root, split)
        if df is None:
            df = meta_from_dicom(args.raw_root, split, args.workers, args.limit_studies)
        if df is None or not len(df):
            print(f"[{split}] no metadata available — skipped")
            rc = 1
            continue
        t0 = time.time()
        rows, agg = audit(df, args.dead_zone_mm)
        report(split, agg, args.dead_zone_mm)
        import pandas as pd

        out = os.path.join(args.out_dir, "study_side.parquet" if split == "train"
                           else f"study_side_{split}.parquet")
        sdf = pd.DataFrame(rows, columns=["StudyInstanceUID", "side", "source",
                                          "x_centre_mm", "tag_side"])
        sdf.to_parquet(out, index=False)
        print(f"[{split}] {len(sdf)} studies -> {out}   ({time.time()-t0:.1f}s)")
        if split == "test":  # inference-path smoke: show the 3 visible studies in full
            print(sdf.to_string(index=False, max_colwidth=28))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
