"""Size-parametric per-STUDY bag dataset (384 / 512 / ...) — the Wave-3 "A2-hi" input path.

`dataset.StudyBagDataset` is hard-wired to 224 (`_read_slice` resizes to (224,224) and the
bag is allocated `[K,3,224,224]`). Wave-3 trains the same recipe at 384px, so this module
adds a `size` parameter WITHOUT touching `dataset.py` (which running trainings import).

What is REUSED verbatim from `dataset.py` (imported, not re-implemented):
  * `build_study_index`  — the study -> [(series, plane, [paths...])] manifest;
  * `sample_study_slices` (+ `_split_evenly`, `_pick_positions`) — the plane -> series ->
    evenly-spaced slot sampler. Slot selection therefore depends ONLY on the per-series
    file *lists*, and the cache tree built by `dev/decode_raw.py` mirrors the 224 tree
    (same `<split>_series/<Study>/<Series>/<Instance>.<ext>` layout, same UIDs, filenames
    still keyed on InstanceUID) -> the SAME InstanceUIDs are picked at 224 and at 384.
    `slot_instance_uids()` below makes that checkable; `reports/hr_arm.md` records the proof.
  * `StudyBagDatasetHR` SUBCLASSES `StudyBagDataset`, so the label lookup, the RNG seeding
    and the augmentation policy (small affine + intensity, **NO flips** — flips would swap
    medial<->lateral / anterior<->posterior and corrupt the laterality-specific targets)
    are literally the baseline's code. Only the per-slice read and the bag shape change.

What is NEW: the per-slice reader is size-parametric and dispatches on file extension —
  * `.jpg/.jpeg/.png` -> `cv2.imread` grayscale, resized to `size` if it is not already
    (so a 384 model can also be fed from the 512 cache);
  * `.dcm`            -> `decode.read_dicom_slice_float(path, size)`, i.e. the ONE validated
    rendering recipe used to build the caches (bit-identical at 224 to the trained path).

Normalisation is unchanged: grayscale -> replicate to 3ch -> (x - mean) / std with the timm
data-config mean/std passed in by the trainer.

Typical use (after `dev/decode_raw.py --sizes 384` has produced `data/cache384`):

    from src.rsna.dataset_hr import StudyBagDatasetHR, build_hr_index, resolve_series_csv
    root = "data/cache384"
    index = build_hr_index(root, split="train", cache_path="data/cache384/study_index_train.pkl")
    ds = StudyBagDatasetHR(ids, index, labels, k=24, train=True, size=384, mean=m, std=s)
"""
from __future__ import annotations

import os
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
    StudyBagDataset,
    build_study_index,
    sample_study_slices,
)
from .decode import read_dicom_slice_float

cv2.setNumThreads(0)  # avoid oversubscription inside DataLoader workers

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# The community 224px JPEG set lives here; every other size comes from dev/decode_raw.py.
JPEG224_ROOT = os.path.join(ROOT, "data", "processed_jpeg224")


# --------------------------------------------------------------------------------------
# Cache-root plumbing
# --------------------------------------------------------------------------------------
def default_cache_root(size: int, root: str = ROOT) -> str:
    """`data/cache<size>` for the decoded caches; the 224 JPEG set keeps its own name."""
    if int(size) == 224 and os.path.isdir(JPEG224_ROOT):
        return JPEG224_ROOT
    return os.path.join(root, "data", f"cache{int(size)}")


def resolve_series_csv(cache_root: str, split: str = "train", root: str = ROOT) -> str:
    """`<cache_root>/<split>_series.csv` (copied there by decode_raw) else `data/comp_csv/`.

    The fallback matters for hand-built / partial caches, where the CSVs were never copied.
    """
    local = os.path.join(cache_root, f"{split}_series.csv")
    if os.path.exists(local):
        return local
    return os.path.join(root, "data", "comp_csv", f"{split}_series.csv")


def build_hr_index(
    cache_root: str,
    split: str = "train",
    series_csv: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> Dict[str, List[Tuple[str, str, List[str]]]]:
    """`dataset.build_study_index` on a cache root (or the raw DICOM mount) — unchanged logic."""
    if series_csv is None:
        series_csv = resolve_series_csv(cache_root, split)
    return build_study_index(cache_root, series_csv, split=split, cache_path=cache_path)


# --------------------------------------------------------------------------------------
# Size-parametric per-slice reader
# --------------------------------------------------------------------------------------
def _read_slice_img(path: str, size: int) -> Optional[np.ndarray]:
    """Read a cached grayscale JPEG as float32 [size,size] in [0,1] (None if unreadable).

    Identical arithmetic to `dataset._read_slice` when `size` matches the file's own edge
    (the normal case: a 384 model reading `data/cache384`); a mismatched cache is resized
    with INTER_AREA, the same interpolation the cache itself was written with.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def read_slice_hr(path: str, size: int) -> Optional[np.ndarray]:
    """Per-slice loader dispatched by extension: DICOM -> validated decoder, else JPEG/PNG.

    Returns float32 [size,size] in [0,1], or None if the slice cannot be read (the dataset
    backfills a None by repeating a good slice, exactly as the 224 DICOM path does).
    """
    if path.lower().endswith(DICOM_EXTS):
        return read_dicom_slice_float(path, size=size, verbose=False)
    return _read_slice_img(path, size)


# --------------------------------------------------------------------------------------
# Slot inspection (used to PROVE 224 vs 384 sampler identity)
# --------------------------------------------------------------------------------------
def slot_paths(
    entry: List[Tuple[str, str, List[str]]],
    k: int,
    i: int = 0,
    seed: int = 42,
    train: bool = False,
) -> List[str]:
    """The exact paths `StudyBagDataset`/`StudyBagDatasetHR` would pick for eval item `i`.

    Mirrors the deterministic (train=False) branch of `StudyBagDataset.__getitem__`:
    `rng = default_rng(seed + i)` then `sample_study_slices(entry, k, train, rng)`.
    """
    rng = np.random.default_rng(seed + i)
    return sample_study_slices(entry, k, train=train, rng=rng)


def slot_instance_uids(
    entry: List[Tuple[str, str, List[str]]],
    k: int,
    i: int = 0,
    seed: int = 42,
    train: bool = False,
) -> List[str]:
    """`slot_paths` reduced to bare InstanceUIDs -> comparable across caches/resolutions."""
    return [os.path.splitext(os.path.basename(p))[0] for p in slot_paths(entry, k, i, seed, train)]


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class StudyBagDatasetHR(StudyBagDataset):
    """Per-study bag of K slices -> [K, 3, size, size] float tensor + 12 soft targets + id.

    Subclass of the 224 dataset: same label map, same RNG seeding, same augmentation
    (`_augment` is resolution-agnostic), same slot sampler. Only the read + bag shape are
    size-aware. With `size=224` and a JPEG tree it is numerically identical to the parent.
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
    ):
        super().__init__(study_ids, index, labels=labels, k=k, train=train,
                         mean=mean, std=std, seed=seed)
        self.size = int(size)

    def slot_instance_uids(self, i: int) -> List[str]:
        """InstanceUIDs this dataset picks for eval item `i` (train-mode jitter excluded)."""
        return slot_instance_uids(self.index[self.study_ids[i]], self.k, i, self.seed, False)

    def __getitem__(self, i: int):
        sid = self.study_ids[i]
        # RNG seeding copied EXACTLY from StudyBagDataset.__getitem__ so that, for the same
        # study and the same item index, the slot sampler makes the same choices at any size.
        if self.train:
            rng = np.random.default_rng(self.seed + i + int(torch.randint(0, 1 << 30, (1,)).item()))
        else:
            rng = np.random.default_rng(self.seed + i)
        paths = sample_study_slices(self.index[sid], self.k, train=self.train, rng=rng)

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
        else:  # whole study undecodable -> zeros (still a valid, in-range bag)
            imgs = [np.zeros((size, size), dtype=np.float32) for _ in imgs]

        bag = np.empty((self.k, 3, size, size), dtype=np.float32)
        for j, img in enumerate(imgs):
            if self.train:
                img = self._augment(img, rng)  # inherited: affine + intensity, NO flips
            chw = np.stack([img, img, img], axis=0)
            chw = (chw - self.mean) / self.std
            bag[j] = chw
        x = torch.from_numpy(bag)
        y = torch.from_numpy(self.label_map.get(sid, np.zeros(len(TARGETS), dtype=np.float32)))
        return x, y, sid


__all__ = [
    "StudyBagDatasetHR",
    "build_hr_index",
    "default_cache_root",
    "resolve_series_csv",
    "read_slice_hr",
    "slot_paths",
    "slot_instance_uids",
    "build_study_index",
    "sample_study_slices",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMG_EXTS",
    "DICOM_EXTS",
    "SLICE_EXTS",
    "PLANE_ORDER",
]
