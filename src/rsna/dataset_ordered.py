"""Anatomically-ORDERED slot sampler: the same K slots, chosen in true slice order.

THE IDEA (a "boring lever" at zero extra compute)
-------------------------------------------------
`dataset.sample_study_slices` splits K slots across planes, then across each plane's series,
then takes "evenly-spaced, middle-biased" positions **inside the series' LEXICOGRAPHIC file
list**. The filename stem is the InstanceUID — a randomised hash. Measured
(`python -m src.rsna.order check`, quoted in `reports/arm_25d.md` §1):

    lexicographic InstanceUID order == anatomical order :  0/400 train series (0.0%)
    mean |Spearman(lex order, anatomical order)|        :  0.152

So "evenly spaced over the lexicographic list" is, anatomically, **a random subset of the
stack**: the K slices land wherever the hash sort put them, a different (arbitrary) subset for
every series, and nothing guarantees the joint centre is covered at all.

This module changes exactly ONE thing: the chosen positions are evenly spaced in **anatomical
rank** (from `order.py`: `ImagePositionPatient` along the slice normal, cached in
`data/series_order.parquet` for the JPEG caches, read live from `.dcm` headers on Kaggle).
Everything else — backbone, K, the plane split (8/8/8 at K=24 across three planes), the
series allocation, the per-series slot counts, the train-time series permutation, the jitter
policy, the augmentation (small affine + intensity, **NO flips**), the normalisation, the RNG
seeding — is the HR arm's own code, imported, not re-implemented.

    HR arm    slot j  ->  paths_lex[ floor(((j+0.5)/n) * len) ]      (random anatomical rank)
    this arm  slot j  ->  paths_anat[ floor(((j+0.5)/n) * len) ]     (rank (j+0.5)/n of the stack)

Hypothesis: systematic anatomical coverage — and, in particular, *consistently* including the
central slices where the menisci and the cruciates live — beats an arbitrary subset. The
columns to watch are the ones the 384 HR arm is weakest on (MCL 0.801, Contusion 0.816,
PF-OA 0.840, Lat-OA 0.846 — EXPERIMENTS.md A2-hi; fold-0 macro-AUC bar = 0.8603).

COVERAGE MODES (`coverage=`)
----------------------------
* ``"even"`` (default) — the whole stack, positions ``(j+0.5)/n`` of the anatomical range.
  With ``coverage="even"`` the position arithmetic is `dataset._pick_positions` itself
  (imported, same jitter, same number of RNG draws), so this arm differs from the HR arm in
  the *index space* only.
* ``"center"`` — the same evenly-spaced positions squeezed into the central
  ``center_frac`` (default 0.60) of the anatomical range, i.e. ranks
  ``0.20*len .. 0.80*len``. In a knee MR stack the peripheral slices are mostly soft tissue /
  air; the joint (menisci, cruciates, cartilage, trochlea) is central. Same K, same cost — it
  just spends the budget where the findings are.

RNG DISCIPLINE (why the comparison is clean)
--------------------------------------------
The train-mode series permutation and the per-position jitter draw the SAME number of values
from the SAME generator in the SAME order as `sample_study_slices`. So for a given study and
item index, this arm allocates slots to (plane, series) identically — only the index *within*
each series changes — and the six augmentation draws per slot stay aligned with the HR arm's.
`assert_same_allocation()` checks that claim on real studies.

USE
---
    from src.rsna.dataset_ordered import StudyBagDatasetOrdered, build_order_resolver
    from src.rsna.dataset_hr import build_hr_index
    index = build_hr_index("data/cache384", split="train")
    res = build_order_resolver("data/series_order.parquet", dicom_root="data/raw")
    ds = StudyBagDatasetOrdered(ids, index, labels, k=24, train=True, size=384,
                                mean=m, std=s, order=res, coverage="even")
"""
from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

from . import TARGETS, ID_COL  # noqa: F401  (re-exported for callers)
from .dataset import (  # reused verbatim -- the allocation logic is NOT re-implemented
    IMAGENET_MEAN,
    IMAGENET_STD,
    PLANE_ORDER,
    _pick_positions,
    _split_evenly,
    sample_study_slices,
)
from .dataset_hr import (  # noqa: F401  (re-exported)
    StudyBagDatasetHR,
    build_hr_index,
    default_cache_root,
    read_slice_hr,
    resolve_series_csv,
    slot_instance_uids,
)
from .order import (
    DEFAULT_ORDER_MODE,
    DEFAULT_ORDER_PARQUET,
    ORDER_MODES,
    SliceOrderResolver,
    stem_of,
)

cv2.setNumThreads(0)  # avoid oversubscription inside DataLoader workers

MODE_TAG = "ordered"          # written into the ckpt args by train_ordered; read by infer_ordered
COVERAGE_MODES = ("even", "center")
DEFAULT_COVERAGE = "even"
DEFAULT_CENTER_FRAC = 0.60     # `coverage="center"` keeps anatomical ranks 20%..80%


def build_order_resolver(
    parquet: Optional[str] = DEFAULT_ORDER_PARQUET,
    dicom_root: Optional[str] = None,
    split: str = "train",
    mode: str = DEFAULT_ORDER_MODE,
    allow_lex: bool = True,
) -> SliceOrderResolver:
    """Parquet first, DICOM headers second, lexicographic last (warned once, and counted).

    Build it in the MAIN process, before the DataLoader forks: the compact int16 rank vectors
    are then shared copy-on-write by every worker instead of being loaded once per worker.
    """
    return SliceOrderResolver.from_parquet(parquet, dicom_root=dicom_root, split=split,
                                           mode=mode, allow_lex=allow_lex)


# --------------------------------------------------------------------------------------
# Position picking
# --------------------------------------------------------------------------------------
class SlotInfo(NamedTuple):
    """One chosen slot, with everything needed to prove what the sampler did."""
    plane: str
    series: str
    rank: int          # anatomical position within the series (0 = first slice of the stack)
    lex_index: int     # index of the same slice in the lexicographic file list
    n_slices: int      # slices in that series
    path: str


def _positions(length: int, n: int, jitter: float, rng: Optional[np.random.Generator],
               coverage: str, center_frac: float) -> List[int]:
    """`n` evenly-spaced, middle-biased positions in `range(length)` — optionally windowed.

    `coverage="even"` delegates to `dataset._pick_positions` VERBATIM (identical arithmetic,
    identical RNG consumption). `coverage="center"` maps the same fractions into the central
    `center_frac` of the range, drawing the same one random value per position so the
    downstream augmentation draws stay aligned with the HR arm's.
    """
    if coverage == "even":
        return _pick_positions(length, n, jitter=jitter, rng=rng)
    if length <= 0 or n <= 0:
        return []
    span = min(1.0, max(1e-3, float(center_frac)))
    lo = (1.0 - span) / 2.0
    out: List[int] = []
    for i in range(n):
        frac = (i + 0.5) / n
        if jitter and rng is not None:
            frac += (rng.random() - 0.5) * jitter / n   # same draw as _pick_positions
        idx = int(math.floor((lo + frac * span) * length))
        out.append(min(length - 1, max(0, idx)))
    return out


def _anat_order(ranks: Optional[np.ndarray], n: int) -> np.ndarray:
    """`anat[r]` = index (into the lexicographic path list) of the slice at anatomical rank r."""
    if ranks is None or len(ranks) != n:
        return np.arange(n, dtype=np.int64)          # lexicographic fallback (resolver warned)
    return np.argsort(np.asarray(ranks), kind="stable").astype(np.int64)


# --------------------------------------------------------------------------------------
# The sampler
# --------------------------------------------------------------------------------------
def sample_study_slices_ordered(
    entry: List[Tuple[str, str, List[str]]],
    k: int,
    train: bool = False,
    rng: Optional[np.random.Generator] = None,
    ranks_fn=None,
    coverage: str = DEFAULT_COVERAGE,
    center_frac: float = DEFAULT_CENTER_FRAC,
    detail: bool = False,
) -> Union[List[str], Tuple[List[str], List[SlotInfo]]]:
    """`dataset.sample_study_slices` with the positions taken in ANATOMICAL order.

    Structurally line-for-line the parent: same plane grouping in `PLANE_ORDER`, same
    `_split_evenly` slot allocation across planes then series, same train-time series
    permutation, same jitter, same pad-by-repeat / truncate. The ONLY change is that a chosen
    position `p` indexes the series' anatomically sorted list instead of its lexicographic one.

    `ranks_fn(series_uid, paths) -> ranks` supplies the order (normally
    `SliceOrderResolver.ranks`); `None`, or a length mismatch, degrades that series to
    lexicographic — i.e. exactly the HR arm's behaviour, never a crash.
    """
    if not entry:
        return ([], []) if detail else []

    # group (series, paths) by plane, in canonical order -- the parent drops the series UID,
    # we keep it because the order resolver is keyed on it.
    by_plane: Dict[str, List[Tuple[str, List[str]]]] = {}
    for series, plane, paths in entry:
        by_plane.setdefault(plane, []).append((series, paths))
    planes = [p for p in PLANE_ORDER if p in by_plane]
    for p in by_plane:                      # any unexpected plane label, appended in dict order
        if p not in planes:
            planes.append(p)

    jitter = 1.0 if train else 0.0
    plane_slots = _split_evenly(k, len(planes))
    chosen: List[str] = []
    recs: List[SlotInfo] = []
    for plane, slots in zip(planes, plane_slots):
        series_list = by_plane[plane]
        if train and rng is not None and len(series_list) > 1:
            order = list(rng.permutation(len(series_list)))     # same draw as the parent
            series_list = [series_list[j] for j in order]
        series_slots = _split_evenly(slots, len(series_list))
        for (series, paths), s in zip(series_list, series_slots):
            n = len(paths)
            ranks = None
            if ranks_fn is not None and n:
                try:
                    ranks = np.asarray(ranks_fn(series, paths))
                except Exception as e:      # a bad header must never kill a training step
                    print(f"[ordered] rank lookup failed for {series[-12:]}: "
                          f"{type(e).__name__}: {e} -> lexicographic for it")
                    ranks = None
            anat = _anat_order(ranks, n)
            for pos in _positions(n, s, jitter, rng, coverage, center_frac):
                li = int(anat[pos])
                chosen.append(paths[li])
                if detail:
                    recs.append(SlotInfo(plane, series, int(pos), li, n, paths[li]))

    if not chosen:                          # degenerate; fall back to all paths flat (parent)
        for series, plane, paths in entry:
            for li, p in enumerate(paths):
                chosen.append(p)
                if detail:
                    recs.append(SlotInfo(plane, series, li, li, len(paths), p))
    if len(chosen) < k:
        reps = (k + len(chosen) - 1) // len(chosen)
        chosen = (chosen * reps)[:k]
        if detail:
            recs = (recs * reps)[:k]
    elif len(chosen) > k:
        chosen = chosen[:k]
        if detail:
            recs = recs[:k]
    return (chosen, recs) if detail else chosen


# --------------------------------------------------------------------------------------
# Allocation identity check (the "only the positions differ" claim)
# --------------------------------------------------------------------------------------
def _series_of_path(entry: List[Tuple[str, str, List[str]]]) -> Dict[str, Tuple[str, str]]:
    return {p: (plane, series) for series, plane, paths in entry for p in paths}


def allocation_of(entry, k: int, i: int = 0, seed: int = 42, train: bool = False,
                  ordered: bool = False, ranks_fn=None, coverage: str = DEFAULT_COVERAGE,
                  center_frac: float = DEFAULT_CENTER_FRAC) -> List[Tuple[str, str]]:
    """The `(plane, series)` each of the K slots was allocated to, for eval item `i`."""
    rng = np.random.default_rng(seed + i)
    if ordered:
        _paths, recs = sample_study_slices_ordered(entry, k, train=train, rng=rng,
                                                   ranks_fn=ranks_fn, coverage=coverage,
                                                   center_frac=center_frac, detail=True)
        return [(r.plane, r.series) for r in recs]
    paths = sample_study_slices(entry, k, train=train, rng=rng)
    lut = _series_of_path(entry)
    return [lut[p] for p in paths]


def assert_same_allocation(entry, k: int, i: int = 0, seed: int = 42, ranks_fn=None,
                           coverage: str = DEFAULT_COVERAGE,
                           center_frac: float = DEFAULT_CENTER_FRAC) -> bool:
    """True iff the ordered sampler allocates its K slots to the same (plane, series) as HR."""
    return allocation_of(entry, k, i, seed, train=False, ordered=False) == \
        allocation_of(entry, k, i, seed, train=False, ordered=True, ranks_fn=ranks_fn,
                      coverage=coverage, center_frac=center_frac)


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class StudyBagDatasetOrdered(StudyBagDatasetHR):
    """`StudyBagDatasetHR` with the K slot positions taken in ANATOMICAL order.

    Arguments beyond `StudyBagDatasetHR`:
        order:        a `SliceOrderResolver` (build it once in the parent process with
                      `build_order_resolver`), OR an ordering-mode string
                      ("patient" / "instance" / "lex") for a header-only resolver — which
                      WARNS on a JPEG tree, so always pass a real resolver for training.
        coverage:     "even"   -> evenly spaced over the whole anatomical range;
                      "center" -> the same spacing squeezed into the central `center_frac`.
        center_frac:  fraction of the stack `coverage="center"` keeps (default 0.60).

    Identical to the HR arm in everything else: bag shape `[K, 3, size, size]`, the label map,
    the RNG seeding, the per-slice reader, the augmentation (affine + intensity, NO flips) and
    the normalisation are all the parent's code.
    """

    def __init__(
        self,
        study_ids: Sequence[str],
        index: Dict[str, List[Tuple[str, str, List[str]]]],
        labels=None,
        k: int = 24,
        train: bool = False,
        size: int = 384,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        seed: int = 42,
        order: Union[str, SliceOrderResolver, None] = DEFAULT_ORDER_MODE,
        coverage: str = DEFAULT_COVERAGE,
        center_frac: float = DEFAULT_CENTER_FRAC,
    ):
        super().__init__(study_ids, index, labels=labels, k=k, train=train, size=size,
                         mean=mean, std=std, seed=seed)
        if isinstance(order, SliceOrderResolver):
            self.order = order
        else:
            mode = DEFAULT_ORDER_MODE if order is None else str(order)
            if mode not in ORDER_MODES:
                raise ValueError(f"order must be one of {ORDER_MODES} or a SliceOrderResolver,"
                                 f" got {order!r}")
            self.order = SliceOrderResolver(mode=mode)   # headers/lex only
        self.order_mode = self.order.mode
        if str(coverage) not in COVERAGE_MODES:
            raise ValueError(f"coverage must be one of {COVERAGE_MODES}, got {coverage!r}")
        self.coverage = str(coverage)
        self.center_frac = float(center_frac)

    # ---------------------------------------------------------------------------------
    # order plumbing
    # ---------------------------------------------------------------------------------
    def _ranks_fn(self):
        return lambda series, paths: self.order.ranks(series, paths)

    def order_report(self) -> str:
        """Tally of where the slice orders came from IN THIS PROCESS (see `order.report`)."""
        return self.order.report()

    # ---------------------------------------------------------------------------------
    # inspection / proof helpers (deterministic eval sampling)
    # ---------------------------------------------------------------------------------
    def sampled_paths(self, i: int, train: bool = False) -> List[str]:
        """The paths this dataset picks for item `i` (eval branch: `default_rng(seed + i)`)."""
        rng = np.random.default_rng(self.seed + i)
        return sample_study_slices_ordered(self.index[self.study_ids[i]], self.k, train=train,
                                           rng=rng, ranks_fn=self._ranks_fn(),
                                           coverage=self.coverage,
                                           center_frac=self.center_frac)

    def slot_details(self, i: int, train: bool = False) -> List[SlotInfo]:
        """Per-slot `(plane, series, anatomical rank, lex index, n_slices, path)`."""
        rng = np.random.default_rng(self.seed + i)
        _paths, recs = sample_study_slices_ordered(
            self.index[self.study_ids[i]], self.k, train=train, rng=rng,
            ranks_fn=self._ranks_fn(), coverage=self.coverage,
            center_frac=self.center_frac, detail=True)
        return recs

    def slot_instance_uids(self, i: int) -> List[str]:
        """THIS arm's slot InstanceUIDs (the parent's method would report the HR arm's)."""
        return [stem_of(p) for p in self.sampled_paths(i)]

    def hr_slot_instance_uids(self, i: int) -> List[str]:
        """What the 224/384 HR arms would have picked — for the side-by-side comparison."""
        return slot_instance_uids(self.index[self.study_ids[i]], self.k, i, self.seed, False)

    def hr_slot_ranks(self, i: int) -> List[SlotInfo]:
        """The HR arm's slots, annotated with the anatomical rank each of them landed on."""
        sid = self.study_ids[i]
        entry = self.index[sid]
        rng = np.random.default_rng(self.seed + i)
        paths = sample_study_slices(entry, self.k, train=False, rng=rng)
        out: List[SlotInfo] = []
        lut = {p: (series, plane, sp) for series, plane, sp in entry for p in sp}
        for p in paths:
            hit = lut.get(p)
            if hit is None:
                continue
            series, plane, sp = hit
            ranks = self.order.ranks(series, sp)
            li = sp.index(p)
            rank = int(ranks[li]) if len(ranks) == len(sp) else li
            out.append(SlotInfo(plane, series, rank, li, len(sp), p))
        return out

    # ---------------------------------------------------------------------------------
    def __getitem__(self, i: int):
        sid = self.study_ids[i]
        # RNG seeding copied EXACTLY from StudyBagDataset.__getitem__: same generator, same
        # draw sequence -> the same (plane, series) allocation and the same augmentation
        # draws as the HR arm; only the index WITHIN each series differs.
        if self.train:
            rng = np.random.default_rng(self.seed + i + int(torch.randint(0, 1 << 30, (1,)).item()))
        else:
            rng = np.random.default_rng(self.seed + i)
        paths = sample_study_slices_ordered(self.index[sid], self.k, train=self.train, rng=rng,
                                            ranks_fn=self._ranks_fn(), coverage=self.coverage,
                                            center_frac=self.center_frac)

        size = self.size
        imgs: List[Optional[np.ndarray]] = [read_slice_hr(p, size) for p in paths]
        valid = [im for im in imgs if im is not None]
        if valid:  # backfill unreadable slots by repeating good slices (never mix in zeros)
            fill = valid * ((len(imgs) // len(valid)) + 1)
            fi = 0
            for j in range(len(imgs)):
                if imgs[j] is None:
                    imgs[j] = fill[fi]
                    fi += 1
        else:      # whole study undecodable -> zeros (still a valid, in-range bag)
            imgs = [np.zeros((size, size), dtype=np.float32) for _ in imgs]

        bag = np.empty((self.k, 3, size, size), dtype=np.float32)
        for j, img in enumerate(imgs):
            if self.train:
                img = self._augment(img, rng)   # inherited: affine + intensity, NO flips
            chw = np.stack([img, img, img], axis=0)
            chw = (chw - self.mean) / self.std
            bag[j] = chw
        x = torch.from_numpy(bag)
        y = torch.from_numpy(self.label_map.get(sid, np.zeros(len(TARGETS), dtype=np.float32)))
        return x, y, sid


__all__ = [
    "StudyBagDatasetOrdered",
    "sample_study_slices_ordered",
    "build_order_resolver",
    "allocation_of",
    "assert_same_allocation",
    "SlotInfo",
    "MODE_TAG",
    "COVERAGE_MODES",
    "DEFAULT_COVERAGE",
    "DEFAULT_CENTER_FRAC",
    "build_hr_index",
    "default_cache_root",
    "resolve_series_csv",
    "read_slice_hr",
    "slot_instance_uids",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
