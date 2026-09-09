"""Anatomical slice ORDER for a DICOM series — the prerequisite for any 2.5D input.

WHY THIS FILE EXISTS
--------------------
`dataset.build_study_index` lists each series directory with `sorted(os.listdir(...))`, i.e.
**lexicographic by InstanceUID** (the filename stem in every tree we use: raw `.dcm`,
`data/cache384/…jpg`, `data/processed_jpeg224/…jpeg`). That is fine for the existing arms —
they treat a series as an unordered bag and only ever need *reproducible* slot positions —
but it is catastrophic for a 2.5D model, which needs the slice that is physically ADJACENT.

Measured on this competition's data (`python -m src.rsna.order check`, numbers in
`reports/arm_25d.md`):

    lexicographic InstanceUID order == anatomical order :  0 / 400 train series (0.0%)
                                                           0 /  15 test  series (0.0%)
    mean |Spearman(lex order, anatomical order)|        :  0.16   (median 0.12)
    InstanceNumber present + unique                     :  400/400 (100%)
    ImagePositionPatient + ImageOrientationPatient       :  400/400 (100%)
    |Spearman(InstanceNumber, IPP-along-normal)| == 1.0  :  400/400 (100%)

These UIDs are randomised hashes: sorting by them shuffles the stack. "slice i+1 in the
sorted list" is a RANDOM other slice, so a naive [i-1, i, i+1] triple would feed the network
three unrelated images. Hence: order comes from the DICOM headers, never from the filename.

ORDERING RULES (`mode=`)
------------------------
* ``"patient"`` (default) — sort by the patient-space coordinate along the dominant axis of
  the slice normal ``n = row x col`` (from ImageOrientationPatient), i.e. by
  ``ImagePositionPatient[argmax|n|]``, ascending, InstanceNumber as tie-break. DICOM patient
  coordinates are LPS, so this is a *fixed* geometric direction for every study
  (sagittal → increasing Left, coronal → increasing Posterior, axial → increasing Superior)
  and does not inherit the ±sign of the IOP convention, which does vary between series.
* ``"instance"`` — InstanceNumber ascending (acquisition order, the MRNet convention). It is
  a perfect monotone map of the geometry here (100% of probed series), but its *direction*
  is scanner-dependent: InstanceNumber runs along +normal in 64% of series and along −normal
  in the other 36%. For a symmetric [prev, cur, next] triple that only swaps channels 0 and 2,
  which is why either mode trains; ``"patient"`` is the deterministic one, so it is default.
* ``"lex"`` — the current (wrong for 2.5D) filename order. Only as an explicit escape hatch;
  it is also the last-resort fallback, and it warns once when it is used implicitly.

WHERE THE ORDER COMES FROM AT RUN TIME
--------------------------------------
1. **A cached parquet** (`data/series_order.parquet`, built once by ``build`` below from the
   raw DICOM headers under `data/raw`). Needed for TRAINING, because the cache trees hold
   JPEGs with no headers at all — but the filename stems are the same InstanceUIDs, so the
   map transfers to `data/cache384` / `data/processed_jpeg224` unchanged.
2. **The DICOM headers themselves** when the tree being read is `.dcm` — this is the Kaggle
   inference path, where no parquet can exist for the hidden test. Only three tags are read
   (`specific_tags` + `stop_before_pixels`), ~0.5 ms/slice, cached per series.
3. **`dicom_root`** — a JPEG cache tree can borrow headers from a parallel raw tree
   (`data/raw`) when no parquet was built; same cost as (2), cached per series.
4. Lexicographic, with a warning. Never silent.

MEMORY
------
The parquet is a per-instance table (~730k rows). Loading it as ``{series: {stem: rank}}``
would cost ~150 MB per DataLoader worker, so the default loader
(:func:`load_series_ranks`) returns the COMPACT form instead: one ``int16`` rank vector per
series, in the lexicographic stem order the dataset index itself uses, plus a CRC32 of that
stem list so a mismatched/partial tree is DETECTED rather than silently mis-ordered
(→ falls back to headers). ~24k series ⇒ ~6 MB.

CLI
---
    # one-off: build the train map from the raw DICOM headers (~10 min, 3 workers)
    .venv/bin/python -m src.rsna.order build --root data/raw --split train \
        --out data/series_order.parquet --workers 3

    # audit: is lexicographic order anatomical? (the numbers quoted above)
    .venv/bin/python -m src.rsna.order check --root data/raw --split train --limit 400
    .venv/bin/python -m src.rsna.order check --root /tmp/dicom_mount --split test
"""
from __future__ import annotations

import os
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ORDER_MODES = ("patient", "instance", "lex")
DEFAULT_ORDER_MODE = "patient"

# InstanceNumber, ImagePositionPatient, ImageOrientationPatient — the only tags we read.
TAG_INSTANCE_NUMBER = (0x0020, 0x0013)
TAG_IMAGE_POSITION = (0x0020, 0x0032)
TAG_IMAGE_ORIENTATION = (0x0020, 0x0037)
ORDER_TAGS = (TAG_INSTANCE_NUMBER, TAG_IMAGE_POSITION, TAG_IMAGE_ORIENTATION)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ORDER_PARQUET = os.path.join(ROOT, "data", "series_order.parquet")

# one-time warnings, so a fallback is loud once and silent afterwards
_WARNED: Dict[str, bool] = {}


def _warn_once(key: str, msg: str) -> None:
    if not _WARNED.get(key):
        _WARNED[key] = True
        print(f"[order] {msg}")


def stem_of(path: str) -> str:
    """`.../<InstanceUID>.dcm|.jpg|.jpeg|.png` -> `<InstanceUID>`."""
    return os.path.splitext(os.path.basename(path))[0]


def stems_checksum(stems: Sequence[str]) -> int:
    """Stable CRC32 over a stem list — detects a partial/different tree at load time.

    `hash()` is salted per process and would differ between the builder and the reader;
    crc32 is fixed by the standard library.
    """
    return int(zlib.crc32("\n".join(stems).encode("utf-8")) & 0xFFFFFFFF)


# --------------------------------------------------------------------------------------
# DICOM header reading
# --------------------------------------------------------------------------------------
def read_slice_meta(path: str) -> Optional[dict]:
    """Read ONLY the three ordering tags from one DICOM. `None` if unreadable.

    `stop_before_pixels=True` + `specific_tags=...` keeps this at ~0.5 ms/slice: no pixel
    data is touched and the parser skips the rest of the header.
    """
    try:
        import pydicom  # lazy: JPEG-only training paths must not need pydicom

        ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=list(ORDER_TAGS))
    except Exception:
        return None
    out: dict = {"instance_number": None, "ipp": None, "iop": None}
    try:
        v = getattr(ds, "InstanceNumber", None)
        out["instance_number"] = None if v is None else int(v)
    except Exception:
        pass
    try:
        v = getattr(ds, "ImagePositionPatient", None)
        if v is not None and len(v) == 3:
            out["ipp"] = np.asarray([float(x) for x in v], dtype=np.float64)
    except Exception:
        pass
    try:
        v = getattr(ds, "ImageOrientationPatient", None)
        if v is not None and len(v) == 6:
            out["iop"] = np.asarray([float(x) for x in v], dtype=np.float64)
    except Exception:
        pass
    return out


def dominant_axis(iop: np.ndarray) -> int:
    """Index (0=x/L, 1=y/P, 2=z/S) of the patient axis the slice normal is closest to."""
    n = np.cross(np.asarray(iop[:3], dtype=np.float64), np.asarray(iop[3:], dtype=np.float64))
    return int(np.argmax(np.abs(n)))


def _ranks_from_keys(primary: Sequence[float], secondary: Sequence[float]) -> np.ndarray:
    """Dense rank of `(primary, secondary)` — `ranks[i]` = anatomical position of item i."""
    order = np.lexsort((np.asarray(secondary, dtype=np.float64),
                        np.asarray(primary, dtype=np.float64)))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order), dtype=np.int64)
    return ranks


def series_ranks_from_dicom(paths: Sequence[str], mode: str = DEFAULT_ORDER_MODE
                            ) -> Tuple[np.ndarray, str]:
    """Anatomical ranks for one series read straight from its DICOM headers.

    Returns `(ranks, method)` where `ranks[i]` is the anatomical position of `paths[i]` and
    `method` records which rule actually applied ("patient" / "instance" / "lex").

    Degrades per series, never per slice: if ANY slice is missing the geometry the whole
    series falls back to InstanceNumber, and if any is missing that too, to lexicographic —
    mixing rules inside one series would be worse than either rule alone.
    """
    n = len(paths)
    if n == 0:
        return np.zeros(0, dtype=np.int64), "lex"
    metas = [read_slice_meta(p) for p in paths]
    inums = [None if m is None else m.get("instance_number") for m in metas]
    have_inum = all(v is not None for v in inums)
    sec = [float(v) if v is not None else 0.0 for v in inums]

    if mode == "patient":
        ipps = [None if m is None else m.get("ipp") for m in metas]
        iops = [None if m is None else m.get("iop") for m in metas]
        if all(p is not None for p in ipps) and all(o is not None for o in iops):
            ax = dominant_axis(iops[0])
            return _ranks_from_keys([float(p[ax]) for p in ipps], sec), "patient"
        _warn_once("patient_missing",
                   "ImagePositionPatient/ImageOrientationPatient missing in a series "
                   "-> falling back to InstanceNumber for it")
    if mode in ("patient", "instance") and have_inum:
        return _ranks_from_keys(sec, np.zeros(n)), "instance"
    if mode != "lex":
        _warn_once("lex_fallback",
                   "no usable InstanceNumber/geometry in a series -> LEXICOGRAPHIC order "
                   "(2.5D neighbours are NOT anatomical for it)")
    return np.arange(n, dtype=np.int64), "lex"


# --------------------------------------------------------------------------------------
# Neighbours
# --------------------------------------------------------------------------------------
def neighbour_indices(ranks: np.ndarray, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """`(prev_idx, next_idx)` in the SAME indexing as `ranks` (i.e. the caller's path list).

    `prev_idx[i]` is the index of the slice `step` positions before `paths[i]` in anatomical
    order, clamped at the ends (edge slices replicate themselves, the standard 2.5D choice).
    """
    n = int(len(ranks))
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    order = np.argsort(ranks, kind="stable")          # order[r] = index at anatomical pos r
    r = np.arange(n)
    prev_idx = np.empty(n, dtype=np.int64)
    next_idx = np.empty(n, dtype=np.int64)
    prev_idx[order] = order[np.clip(r - step, 0, n - 1)]
    next_idx[order] = order[np.clip(r + step, 0, n - 1)]
    return prev_idx, next_idx


# --------------------------------------------------------------------------------------
# Parquet build / load
# --------------------------------------------------------------------------------------
def iter_series_dirs(root: str, split: str = "train") -> List[Tuple[str, str, str]]:
    """`[(study, series, dir), ...]` under `<root>/<split>_series` (sorted, deterministic)."""
    base = os.path.join(root, f"{split}_series")
    out: List[Tuple[str, str, str]] = []
    if not os.path.isdir(base):
        return out
    for study in sorted(os.listdir(base)):
        sd = os.path.join(base, study)
        if not os.path.isdir(sd):
            continue
        for series in sorted(os.listdir(sd)):
            p = os.path.join(sd, series)
            if os.path.isdir(p):
                out.append((study, series, p))
    return out


def _series_rows(job: Tuple[str, str, str, str]) -> List[Tuple[str, str, str, int, str]]:
    """Worker: one series dir -> `[(study, series, stem, order_index, method), ...]`."""
    study, series, sdir, mode = job
    try:
        files = sorted(fn for fn in os.listdir(sdir) if fn.lower().endswith(".dcm"))
    except OSError:
        return []
    if not files:
        return []
    paths = [os.path.join(sdir, fn) for fn in files]
    ranks, method = series_ranks_from_dicom(paths, mode=mode)
    return [(study, series, stem_of(p), int(r), method) for p, r in zip(paths, ranks)]


def build_series_order_table(
    root: str,
    split: str = "train",
    mode: str = DEFAULT_ORDER_MODE,
    workers: int = 3,
    limit: Optional[int] = None,
    progress_every: int = 500,
):
    """Scan a raw DICOM tree and return the per-instance order table as a DataFrame.

    Columns: `StudyInstanceUID, SeriesInstanceUID, InstanceUID, order_index, method`.
    Keep `workers` small (default 3): this is 700k+ header reads over a 247 GB tree and must
    not starve a concurrent GPU training of I/O.
    """
    import pandas as pd

    dirs = iter_series_dirs(root, split)
    if limit:
        dirs = dirs[:limit]
    jobs = [(st, se, d, mode) for st, se, d in dirs]
    print(f"[order] scanning {len(jobs)} series under {root}/{split}_series "
          f"(mode={mode}, workers={workers})", flush=True)
    rows: List[Tuple[str, str, str, int, str]] = []
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            for i, res in enumerate(ex.map(_series_rows, jobs, chunksize=8)):
                rows.extend(res)
                if progress_every and (i + 1) % progress_every == 0:
                    print(f"  [order] {i + 1}/{len(jobs)} series, {len(rows)} slices",
                          flush=True)
    else:
        for i, job in enumerate(jobs):
            rows.extend(_series_rows(job))
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  [order] {i + 1}/{len(jobs)} series, {len(rows)} slices", flush=True)
    df = pd.DataFrame(rows, columns=["StudyInstanceUID", "SeriesInstanceUID",
                                     "InstanceUID", "order_index", "method"])
    df["order_index"] = df["order_index"].astype("int32")
    return df


def write_series_order_parquet(df, out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[order] wrote {out_path}: {len(df)} slices, "
          f"{df['SeriesInstanceUID'].nunique()} series, "
          f"{os.path.getsize(out_path) / 1e6:.1f} MB")
    return out_path


def load_series_ranks(path: str) -> Dict[str, Tuple[int, np.ndarray]]:
    """COMPACT loader: `{series_uid: (crc32_of_lex_stems, int16 ranks in lex stem order)}`.

    The dataset index lists a series with `sorted(os.listdir(...))`, so the rank vector lines
    up positionally with what the dataset sees — no per-stem dict, no 150 MB of strings. The
    CRC guards the assumption: a partial cache (a slice that failed to decode) changes it and
    the resolver falls back to headers instead of silently mis-pairing every slice.
    """
    import pandas as pd

    df = pd.read_parquet(path, columns=["SeriesInstanceUID", "InstanceUID", "order_index"])
    out: Dict[str, Tuple[int, np.ndarray]] = {}
    for series, g in df.groupby("SeriesInstanceUID", sort=False):
        g = g.sort_values("InstanceUID", kind="stable")           # lexicographic stem order
        stems = g["InstanceUID"].tolist()
        ranks = g["order_index"].to_numpy()
        # re-densify: order_index was built over the full series, which is what we have here
        dense = np.empty(len(ranks), dtype=np.int32)
        dense[np.argsort(ranks, kind="stable")] = np.arange(len(ranks), dtype=np.int32)
        dtype = np.int16 if len(dense) < 32767 else np.int32
        out[str(series)] = (stems_checksum(stems), dense.astype(dtype))
    print(f"[order] loaded {len(out)} series from {path} (compact rank vectors)")
    return out


def load_series_order(path: str) -> Dict[str, Dict[str, int]]:
    """EXACT loader: `{series_uid: {instance_uid: order_index}}`. ~150 MB on the full train
    map — use it for analysis/debugging, not inside DataLoader workers."""
    import pandas as pd

    df = pd.read_parquet(path, columns=["SeriesInstanceUID", "InstanceUID", "order_index"])
    out: Dict[str, Dict[str, int]] = {}
    for series, g in df.groupby("SeriesInstanceUID", sort=False):
        out[str(series)] = dict(zip(g["InstanceUID"].astype(str), g["order_index"].astype(int)))
    return out


# --------------------------------------------------------------------------------------
# Resolver
# --------------------------------------------------------------------------------------
class SliceOrderResolver:
    """Turns a series' (lexicographic) path list into anatomical ranks + neighbour indices.

    Sources, tried in order:
      1. `ranks_map` (from `load_series_ranks`) when its CRC matches the observed stem list;
      2. the DICOM headers of the paths themselves, when they are `.dcm`;
      3. the DICOM headers of the mirror tree at `dicom_root`
         (`<dicom_root>/<split>_series/<study>/<series>/<stem>.dcm`), for JPEG caches;
      4. lexicographic order — WARNED once, and counted in `self.stats`.

    Results are memoised per series (`{series_uid: ranks}`), so a header source costs one
    read per series per worker per run, not one per sampled slice.
    """

    def __init__(
        self,
        ranks_map: Optional[Dict[str, Tuple[int, np.ndarray]]] = None,
        dicom_root: Optional[str] = None,
        split: str = "train",
        mode: str = DEFAULT_ORDER_MODE,
        allow_lex: bool = True,
    ):
        if mode not in ORDER_MODES:
            raise ValueError(f"mode must be one of {ORDER_MODES}, got {mode!r}")
        self.ranks_map = ranks_map or {}
        self.dicom_root = dicom_root
        self.split = split
        self.mode = mode
        self.allow_lex = allow_lex
        self._cache: Dict[str, np.ndarray] = {}
        self.stats: Dict[str, int] = {"map": 0, "dicom": 0, "mirror": 0, "lex": 0, "crc_miss": 0}

    # -- construction helpers ----------------------------------------------------------
    @classmethod
    def from_parquet(cls, path: Optional[str], **kw) -> "SliceOrderResolver":
        """Build from a parquet path; a missing/None path yields a header-only resolver."""
        ranks_map = None
        if path and os.path.exists(path):
            ranks_map = load_series_ranks(path)
        elif path:
            print(f"[order] parquet {path} not found -> header/lex fallback")
        return cls(ranks_map=ranks_map, **kw)

    # -- core --------------------------------------------------------------------------
    def _mirror_paths(self, paths: Sequence[str]) -> Optional[List[str]]:
        """Same slices under `dicom_root`, as `.dcm` (None if that tree lacks the series)."""
        if not self.dicom_root or not paths:
            return None
        d = os.path.dirname(paths[0])
        study, series = os.path.basename(os.path.dirname(d)), os.path.basename(d)
        sdir = os.path.join(self.dicom_root, f"{self.split}_series", study, series)
        if not os.path.isdir(sdir):
            return None
        out = [os.path.join(sdir, stem_of(p) + ".dcm") for p in paths]
        return out if all(os.path.exists(p) for p in out) else None

    def ranks(self, series_uid: str, paths: Sequence[str]) -> np.ndarray:
        """`ranks[i]` = anatomical position of `paths[i]` (0 = first slice of the stack)."""
        cached = self._cache.get(series_uid)
        if cached is not None and len(cached) == len(paths):
            return cached
        ranks = self._resolve(series_uid, paths)
        self._cache[series_uid] = ranks
        return ranks

    def _resolve(self, series_uid: str, paths: Sequence[str]) -> np.ndarray:
        n = len(paths)
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        if self.mode == "lex":
            self.stats["lex"] += 1
            return np.arange(n, dtype=np.int64)

        entry = self.ranks_map.get(series_uid)
        if entry is not None:
            crc, vec = entry
            if len(vec) == n and crc == stems_checksum([stem_of(p) for p in paths]):
                self.stats["map"] += 1
                return vec.astype(np.int64)
            self.stats["crc_miss"] += 1
            _warn_once(
                "crc_miss",
                f"series {series_uid[-12:]}: cached order does not match the files on disk "
                f"({len(vec)} mapped vs {n} present) -> re-reading headers for it")

        if str(paths[0]).lower().endswith(".dcm"):
            ranks, method = series_ranks_from_dicom(paths, mode=self.mode)
            self.stats["dicom" if method != "lex" else "lex"] += 1
            return ranks

        mirror = self._mirror_paths(paths)
        if mirror is not None:
            ranks, method = series_ranks_from_dicom(mirror, mode=self.mode)
            self.stats["mirror" if method != "lex" else "lex"] += 1
            return ranks

        if not self.allow_lex:
            raise RuntimeError(
                f"no anatomical order available for series {series_uid} (no parquet entry, "
                f"no DICOM headers, allow_lex=False)")
        self.stats["lex"] += 1
        _warn_once("resolver_lex",
                   "falling back to LEXICOGRAPHIC slice order — 2.5D neighbours will NOT be "
                   "anatomical. Build data/series_order.parquet "
                   "(python -m src.rsna.order build) or pass --dicom-root.")
        return np.arange(n, dtype=np.int64)

    def neighbours(self, series_uid: str, paths: Sequence[str], step: int = 1
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """`(prev_idx, next_idx)` into `paths` — clamped at the stack ends."""
        return neighbour_indices(self.ranks(series_uid, paths), step=step)

    def report(self) -> str:
        """Where the orders came from IN THIS PROCESS.

        A DataLoader with `num_workers > 0` forks a copy of the resolver per worker, and the
        counters live in the fork — so the parent's line reads near-zero during training even
        though every series was resolved. Run with `--workers 0` to see the real tally; the
        thing to check in a normal run is that `lex` and `crc_miss` are 0 there.
        """
        s = self.stats
        return (f"[order] sources (this process only; DataLoader workers count their own): "
                f"parquet={s['map']} dicom={s['dicom']} mirror={s['mirror']} "
                f"lex={s['lex']} crc_miss={s['crc_miss']} (mode={self.mode})")


# --------------------------------------------------------------------------------------
# Audit ("is lexicographic order anatomical?")
# --------------------------------------------------------------------------------------
def _audit_series(job: Tuple[str, str, str, str]) -> Optional[dict]:
    study, series, sdir, mode = job
    try:
        files = sorted(fn for fn in os.listdir(sdir) if fn.lower().endswith(".dcm"))
    except OSError:
        return None
    if len(files) < 3:
        return None
    paths = [os.path.join(sdir, fn) for fn in files]
    metas = [read_slice_meta(p) for p in paths]
    n = len(paths)
    inums = [None if m is None else m.get("instance_number") for m in metas]
    ipps = [None if m is None else m.get("ipp") for m in metas]
    iops = [None if m is None else m.get("iop") for m in metas]
    r = {"n": n, "has_inum": all(v is not None for v in inums),
         "has_geom": all(p is not None for p in ipps) and all(o is not None for o in iops)}
    ranks, method = series_ranks_from_dicom(paths, mode=mode)
    lex = np.arange(n)
    r["method"] = method
    r["lex_eq_anat"] = bool(np.array_equal(ranks, lex))
    r["lex_eq_anat_rev"] = bool(np.array_equal(ranks, lex[::-1]))
    r["spearman_lex_anat"] = float(abs(np.corrcoef(lex, ranks)[0, 1])) if n > 1 else 1.0
    if r["has_inum"]:
        iv = np.asarray([int(v) for v in inums], dtype=np.float64)
        r["inum_unique"] = bool(len(set(iv.tolist())) == n)
        ir = _ranks_from_keys(iv, np.zeros(n))
        r["inum_eq_anat"] = bool(np.array_equal(ir, ranks))
        r["inum_eq_anat_rev"] = bool(np.array_equal(ir, (n - 1) - ranks))
        r["spearman_inum_anat"] = float(abs(np.corrcoef(ir, ranks)[0, 1])) if n > 1 else 1.0
    return r


def audit_ordering(root: str, split: str = "train", limit: int = 400,
                   mode: str = DEFAULT_ORDER_MODE, workers: int = 3, seed: int = 0) -> dict:
    """Sample series, compare lexicographic vs anatomical order, print + return the stats."""
    import random

    dirs = iter_series_dirs(root, split)
    if limit and len(dirs) > limit:
        rnd = random.Random(seed)
        # sample whole studies' worth in a stable way: shuffle the series list, take `limit`
        dirs = rnd.sample(dirs, limit)
    jobs = [(st, se, d, mode) for st, se, d in dirs]
    res: List[dict] = []
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            res = [r for r in ex.map(_audit_series, jobs, chunksize=4) if r]
    else:
        res = [r for r in (_audit_series(j) for j in jobs) if r]
    if not res:
        print(f"[order] audit: no usable series under {root}/{split}_series")
        return {}

    def pct(key):
        vals = [bool(r[key]) for r in res if key in r]
        return 100.0 * sum(vals) / max(1, len(vals)), len(vals)

    sp = np.asarray([r["spearman_lex_anat"] for r in res])
    out = {
        "root": root, "split": split, "mode": mode, "series": len(res),
        "slices": int(sum(r["n"] for r in res)),
        "lex_eq_anat_pct": pct("lex_eq_anat")[0],
        "lex_eq_anat_rev_pct": pct("lex_eq_anat_rev")[0],
        "spearman_lex_anat_mean": float(sp.mean()),
        "spearman_lex_anat_median": float(np.median(sp)),
        "spearman_lex_anat_max": float(sp.max()),
        "has_instance_number_pct": pct("has_inum")[0],
        "has_geometry_pct": pct("has_geom")[0],
        "instance_number_unique_pct": pct("inum_unique")[0],
        "inum_eq_anat_pct": pct("inum_eq_anat")[0],
        "inum_eq_anat_rev_pct": pct("inum_eq_anat_rev")[0],
        "methods": {m: sum(1 for r in res if r["method"] == m) for m in set(r["method"] for r in res)},
    }
    print(f"[order] AUDIT {root}/{split}_series — {out['series']} series, {out['slices']} slices, "
          f"mode={mode}")
    print(f"  lexicographic == anatomical      : {out['lex_eq_anat_pct']:.1f}% "
          f"(reversed: {out['lex_eq_anat_rev_pct']:.1f}%)")
    print(f"  |spearman(lex, anatomical)|      : mean {out['spearman_lex_anat_mean']:.3f} "
          f"median {out['spearman_lex_anat_median']:.3f} max {out['spearman_lex_anat_max']:.3f}")
    print(f"  InstanceNumber present / unique  : {out['has_instance_number_pct']:.1f}% / "
          f"{out['instance_number_unique_pct']:.1f}%")
    print(f"  IPP+IOP present                  : {out['has_geometry_pct']:.1f}%")
    print(f"  InstanceNumber order == geometry : {out['inum_eq_anat_pct']:.1f}% forward, "
          f"{out['inum_eq_anat_rev_pct']:.1f}% reversed")
    print(f"  ordering method used             : {out['methods']}")
    return out


# --------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="scan a raw DICOM tree -> series order parquet")
    b.add_argument("--root", default=os.path.join(ROOT, "data", "raw"))
    b.add_argument("--split", default="train")
    b.add_argument("--out", default=DEFAULT_ORDER_PARQUET)
    b.add_argument("--mode", default=DEFAULT_ORDER_MODE, choices=list(ORDER_MODES))
    b.add_argument("--workers", type=int, default=3)
    b.add_argument("--limit", type=int, default=0, help="smoke: only the first N series")

    c = sub.add_parser("check", help="audit lexicographic vs anatomical order")
    c.add_argument("--root", default=os.path.join(ROOT, "data", "raw"))
    c.add_argument("--split", default="train")
    c.add_argument("--limit", type=int, default=400)
    c.add_argument("--mode", default=DEFAULT_ORDER_MODE, choices=list(ORDER_MODES))
    c.add_argument("--workers", type=int, default=3)
    c.add_argument("--json", default=None, help="also dump the stats here")

    a = ap.parse_args(argv)
    if a.cmd == "build":
        df = build_series_order_table(a.root, a.split, mode=a.mode, workers=a.workers,
                                      limit=a.limit or None)
        if df.empty:
            print("[order] nothing scanned — wrong --root/--split?")
            return 1
        write_series_order_parquet(df, a.out)
        print("[order] method counts:", df.groupby("method").size().to_dict())
        return 0
    stats = audit_ordering(a.root, a.split, limit=a.limit, mode=a.mode, workers=a.workers)
    if a.json and stats:
        with open(a.json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[order] wrote {a.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ORDER_MODES",
    "DEFAULT_ORDER_MODE",
    "DEFAULT_ORDER_PARQUET",
    "SliceOrderResolver",
    "audit_ordering",
    "build_series_order_table",
    "dominant_axis",
    "iter_series_dirs",
    "load_series_order",
    "load_series_ranks",
    "neighbour_indices",
    "read_slice_meta",
    "series_ranks_from_dicom",
    "stem_of",
    "stems_checksum",
    "write_series_order_parquet",
]
