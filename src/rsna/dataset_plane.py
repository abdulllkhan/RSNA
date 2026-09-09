"""Per-STUDY, PER-PLANE bag dataset — the Wave-3 "A3" plane-specialised input path.

The baseline (`dataset.StudyBagDataset`) pools ONE bag of K=24 slices whose slots are
*mixed across planes*: the attention pool sees sagittal, coronal and axial slices in the
same softmax and has to discover, implicitly and per finding, which plane carries the
evidence. Radiology reads are explicitly plane-specific:

    Sagittal -> ACL, Medial/Lateral Meniscus, Baker's, Effusion
    Coronal  -> MCL, Medial/Lateral OA
    Axial    -> PF OA, Synovitis/Effusion, Contusion/Fracture

So A3 hands the model THREE bags (one per plane) and lets it weight the planes per finding
(`model_plane.StudyPlaneModel`). This module builds those bags.

What is REUSED verbatim (imported, never re-implemented):
  * `dataset.build_study_index` / `dataset_hr.build_hr_index` — the study manifest
    `{study: [(series, plane, [paths...]), ...]}`; planes come from `<split>_series.csv`
    (`Anatomical_Plane` in {Sagittal, Coronal, Axial});
  * `dataset.sample_study_slices` — the SAME slot sampler (plane -> series -> evenly-spaced
    middle-biased slices, middle-jittered in train mode), just called once per plane on the
    plane-restricted sub-entry. Because a plane-restricted entry has a single plane key, the
    sampler's plane-stratification step collapses and its per-series/per-slice behaviour is
    bit-identical to the baseline's — nothing is forked;
  * `dataset_hr.read_slice_hr` — the size-parametric, extension-dispatched reader
    (`.jpeg/.jpg/.png` -> cv2, `.dcm` -> `decode.read_dicom_slice_float`), so the 224 JPEG
    tree (`data/processed_jpeg224`), the decoded caches (`data/cache384`, `data/cache512`)
    and a raw DICOM mount are interchangeable inputs;
  * `StudyBagDatasetHR` (hence `StudyBagDataset`) is the BASE CLASS -> the label lookup, the
    RNG seeding and the augmentation policy (small affine + intensity, **NO flips** — flips
    swap medial<->lateral / anterior<->posterior and would corrupt the laterality-specific
    targets) are literally the baseline's code.

What is NEW:
  * `k_per_plane=(8, 8, 8)` slots, allocated PER PLANE instead of `k=24` mixed slots. The
    default keeps the per-study slice budget at 24 — identical compute to the 224 baseline;
  * a **plane mask**: a study missing a plane (see `plane_coverage()`; ~13% of studies lack
    Axial, ~2% lack Coronal) still yields a fixed-shape bag — the absent plane's slots are
    filled by re-sampling the SAME sampler over the study's available planes — and
    `plane_mask[p] = False` tells the model to drop that plane from both the per-plane
    attention pool and the per-finding plane softmax. The fill is real anatomy rather than
    zeros so the shared backbone never sees a degenerate batch.

Item layout (deterministic, so the model can `split` without extra bookkeeping):

    x           float32 [sum(k_per_plane), 3, size, size]   Sagittal block, then Coronal,
                                                            then Axial (canonical PLANE_ORDER)
    y           float32 [12]                                soft targets (0 at inference)
    plane_mask  bool    [3]                                 True where the plane really exists
    sid         str                                         StudyInstanceUID

Typical use:

    from src.rsna.dataset_plane import StudyPlaneBagDataset, build_plane_index
    index = build_plane_index("data/processed_jpeg224", split="train")
    ds = StudyPlaneBagDataset(ids, index, labels, k_per_plane=(8, 8, 8), size=224,
                              train=True, mean=m, std=s)
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from . import TARGETS, ID_COL  # noqa: F401  (re-exported for callers)
from .dataset import (  # reused verbatim -- do NOT re-implement any of these
    DICOM_EXTS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_EXTS,
    PLANE_ORDER,
    SLICE_EXTS,
    build_study_index,
    sample_study_slices,
)
from .dataset_hr import (
    ROOT,
    StudyBagDatasetHR,
    build_hr_index,
    default_cache_root,
    read_slice_hr,
    resolve_series_csv,
)

cv2.setNumThreads(0)  # avoid oversubscription inside DataLoader workers

N_PLANES = len(PLANE_ORDER)  # 3: Sagittal, Coronal, Axial
DEFAULT_K_PER_PLANE: Tuple[int, int, int] = (8, 8, 8)  # 24 slices/study = baseline budget

StudyEntry = List[Tuple[str, str, List[str]]]


# --------------------------------------------------------------------------------------
# Index plumbing (thin aliases over the shared builders so callers import ONE module)
# --------------------------------------------------------------------------------------
def build_plane_index(
    cache_root: str,
    split: str = "train",
    series_csv: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> Dict[str, StudyEntry]:
    """`dataset_hr.build_hr_index` — unchanged logic; the planes it records drive everything."""
    return build_hr_index(cache_root, split=split, series_csv=series_csv, cache_path=cache_path)


# --------------------------------------------------------------------------------------
# Plane splitting + the per-plane slot sampler
# --------------------------------------------------------------------------------------
def split_entry_by_plane(entry: StudyEntry) -> Dict[str, StudyEntry]:
    """{plane: [(series, plane, paths), ...]} for the 3 canonical planes (missing -> [])."""
    out: Dict[str, StudyEntry] = {p: [] for p in PLANE_ORDER}
    for series, plane, paths in entry:
        p = plane if plane in PLANE_ORDER else PLANE_ORDER[0]  # same fallback as build_study_index
        out[p].append((series, p, paths))
    return out


def path_plane_map(entry: StudyEntry) -> Dict[str, str]:
    """{slice path -> plane} for every slice of a study (used to PROVE per-plane sampling)."""
    m: Dict[str, str] = {}
    for _series, plane, paths in entry:
        p = plane if plane in PLANE_ORDER else PLANE_ORDER[0]
        for path in paths:
            m[path] = p
    return m


def sample_plane_slots(
    entry: StudyEntry,
    k_per_plane: Sequence[int] = DEFAULT_K_PER_PLANE,
    train: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[str], List[bool]]:
    """Slots for one study, plane block by plane block, in canonical `PLANE_ORDER`.

    For a plane the study HAS: `sample_study_slices` on that plane's series only -> the slots
    are guaranteed to be slices of that plane (asserted in `reports/plane_arm.md`).
    For a plane the study LACKS: `sample_study_slices` over the WHOLE entry (i.e. repeated
    material from the planes that do exist) so the bag keeps its fixed shape; the returned
    `present` flag is False and the model masks that plane out entirely.

    Returns (paths [sum(k_per_plane)], present [3 bools]).
    """
    by_plane = split_entry_by_plane(entry)
    paths: List[str] = []
    present: List[bool] = []
    for pi, plane in enumerate(PLANE_ORDER):
        k = int(k_per_plane[pi])
        sub = by_plane[plane]
        has = bool(sub) and any(p for _s, _pl, p in sub)
        present.append(has)
        if k <= 0:
            continue
        source = sub if has else entry          # fill an absent plane from what the study has
        chosen = sample_study_slices(source, k, train=train, rng=rng)
        if len(chosen) < k and chosen:          # defensive; sample_study_slices already pads
            chosen = (chosen * ((k // len(chosen)) + 1))[:k]
        paths.extend(chosen[:k])
    return paths, present


def plane_ids_for(k_per_plane: Sequence[int] = DEFAULT_K_PER_PLANE) -> np.ndarray:
    """int64 [sum(k_per_plane)] mapping each slot to its plane index (0=Sag, 1=Cor, 2=Ax)."""
    return np.concatenate(
        [np.full(int(k), i, dtype=np.int64) for i, k in enumerate(k_per_plane) if int(k) > 0]
    ) if sum(int(k) for k in k_per_plane) else np.zeros(0, dtype=np.int64)


def plane_slice_counts(entry: StudyEntry) -> Dict[str, int]:
    """{plane: n_slices} for one study (0 for an absent plane)."""
    out = {p: 0 for p in PLANE_ORDER}
    for _series, plane, paths in entry:
        p = plane if plane in PLANE_ORDER else PLANE_ORDER[0]
        out[p] += len(paths)
    return out


def plane_coverage(index: Dict[str, StudyEntry]) -> Dict[str, object]:
    """Corpus-level per-plane statistics — how many studies lack each plane, slices/plane.

    Reported in `reports/plane_arm.md`: it sizes the cost of the missing-plane fill (wasted
    backbone slots) and tells us whether a per-plane MODEL (which would simply have no input
    for those studies) is viable versus the single plane-aware model we build here.
    """
    n = len(index)
    missing = {p: 0 for p in PLANE_ORDER}
    slices: Dict[str, List[int]] = {p: [] for p in PLANE_ORDER}
    combo = Counter()
    for _sid, entry in index.items():
        c = plane_slice_counts(entry)
        for p in PLANE_ORDER:
            if c[p] == 0:
                missing[p] += 1
            else:
                slices[p].append(c[p])
        combo["+".join(p[:3] for p in PLANE_ORDER if c[p] > 0) or "none"] += 1
    def _stats(v: List[int]) -> Dict[str, float]:
        if not v:
            return dict(n=0, mean=0.0, median=0.0, p10=0.0, p90=0.0, min=0.0, max=0.0)
        a = np.asarray(v, dtype=np.float64)
        return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                    p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
                    min=float(a.min()), max=float(a.max()))
    return dict(
        n_studies=n,
        missing_counts=missing,
        missing_frac={p: (missing[p] / n if n else 0.0) for p in PLANE_ORDER},
        slices_per_plane={p: _stats(slices[p]) for p in PLANE_ORDER},
        plane_combos=dict(combo.most_common()),
        n_all_three=int(combo["+".join(p[:3] for p in PLANE_ORDER)]),
    )


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class StudyPlaneBagDataset(StudyBagDatasetHR):
    """Three plane bags per study -> [sum(k_per_plane), 3, size, size] + a [3] plane mask.

    Subclass of the HR dataset (hence of the 224 baseline dataset): identical label map,
    identical RNG seeding, identical augmentation (affine + intensity, NO flips), identical
    per-slice reader. Only the SLOT ALLOCATION (per plane, not mixed) and the extra mask are
    new. With `k_per_plane=(k1,k2,k3)` the bag holds `k1+k2+k3` slices — the default
    `(8,8,8)` matches the baseline's 24-slice budget exactly, so a fold costs the same.
    """

    def __init__(
        self,
        study_ids: Sequence[str],
        index: Dict[str, StudyEntry],
        labels=None,
        k_per_plane: Sequence[int] = DEFAULT_K_PER_PLANE,
        train: bool = False,
        size: int = 224,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        seed: int = 42,
    ):
        kpp = tuple(int(v) for v in k_per_plane)
        if len(kpp) != N_PLANES:
            raise ValueError(f"k_per_plane must have {N_PLANES} entries, got {kpp}")
        if sum(kpp) <= 0:
            raise ValueError(f"k_per_plane must allocate at least one slot, got {kpp}")
        super().__init__(study_ids, index, labels=labels, k=sum(kpp), train=train, size=size,
                         mean=mean, std=std, seed=seed)
        self.k_per_plane = kpp
        self.k_total = sum(kpp)
        self.plane_ids = plane_ids_for(kpp)

    # ---- introspection (used by the smoke tests / the report) -------------------------
    def _rng_for(self, i: int) -> np.random.Generator:
        """RNG seeding copied EXACTLY from `StudyBagDataset.__getitem__`."""
        if self.train:
            return np.random.default_rng(self.seed + i + int(torch.randint(0, 1 << 30, (1,)).item()))
        return np.random.default_rng(self.seed + i)

    def slot_paths_plane(self, i: int, train: Optional[bool] = None) -> Tuple[List[str], List[bool]]:
        """The exact (paths, present) this dataset would use for item `i` (eval-mode by default)."""
        tr = self.train if train is None else bool(train)
        rng = np.random.default_rng(self.seed + i)
        return sample_plane_slots(self.index[self.study_ids[i]], self.k_per_plane, train=tr, rng=rng)

    def slot_planes(self, i: int) -> List[str]:
        """The TRUE plane of each chosen slot (from the index) — compare against `plane_ids`."""
        entry = self.index[self.study_ids[i]]
        pmap = path_plane_map(entry)
        paths, _present = self.slot_paths_plane(i)
        return [pmap.get(p, "?") for p in paths]

    def slot_instance_uids_plane(self, i: int) -> List[str]:
        """Slot paths reduced to bare InstanceUIDs -> comparable across caches/resolutions."""
        paths, _present = self.slot_paths_plane(i)
        return [os.path.splitext(os.path.basename(p))[0] for p in paths]

    # ---- the item ---------------------------------------------------------------------
    def __getitem__(self, i: int):
        sid = self.study_ids[i]
        rng = self._rng_for(i)
        paths, present = sample_plane_slots(self.index[sid], self.k_per_plane,
                                            train=self.train, rng=rng)

        size = self.size
        imgs: List[Optional[np.ndarray]] = [read_slice_hr(p, size) for p in paths]
        if len(imgs) < self.k_total:  # degenerate study (should not happen: index entries are non-empty)
            imgs += [None] * (self.k_total - len(imgs))
        valid = [im for im in imgs if im is not None]
        if valid:  # backfill unreadable slots by repeating good slices (never mix in zeros)
            fill = valid * ((len(imgs) // len(valid)) + 1)
            fi = 0
            for j in range(len(imgs)):
                if imgs[j] is None:
                    imgs[j] = fill[fi]
                    fi += 1
        else:  # whole study undecodable -> zeros (still a valid, in-range bag)
            imgs = [np.zeros((size, size), dtype=np.float32) for _ in range(self.k_total)]

        bag = np.empty((self.k_total, 3, size, size), dtype=np.float32)
        for j, img in enumerate(imgs):
            if self.train:
                img = self._augment(img, rng)  # inherited: affine + intensity, NO flips
            chw = np.stack([img, img, img], axis=0)  # replicate grayscale to 3ch
            chw = (chw - self.mean) / self.std
            bag[j] = chw

        x = torch.from_numpy(bag)
        y = torch.from_numpy(self.label_map.get(sid, np.zeros(len(TARGETS), dtype=np.float32)))
        mask = torch.tensor([bool(present[p]) and self.k_per_plane[p] > 0 for p in range(N_PLANES)],
                            dtype=torch.bool)
        return x, y, mask, sid


__all__ = [
    "StudyPlaneBagDataset",
    "build_plane_index",
    "split_entry_by_plane",
    "sample_plane_slots",
    "path_plane_map",
    "plane_ids_for",
    "plane_slice_counts",
    "plane_coverage",
    "default_cache_root",
    "resolve_series_csv",
    "read_slice_hr",
    "build_study_index",
    "sample_study_slices",
    "DEFAULT_K_PER_PLANE",
    "N_PLANES",
    "PLANE_ORDER",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMG_EXTS",
    "DICOM_EXTS",
    "SLICE_EXTS",
    "ROOT",
]
