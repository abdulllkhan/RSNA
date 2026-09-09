"""Per-STUDY bag dataset over the 224px JPEGs.

For each study we build a fixed-size "slot" bag of K slices sampled across its
series/planes (stratified by anatomical plane, evenly-spaced/middle slices per series;
pad/repeat if few). Returns a [K, 3, 224, 224] tensor + the 12 soft targets + study id.

Directory layout (both train and the Kaggle mount mirror this):
    <root>/<train|test>_series/<StudyUID>/<SeriesUID>/*.jpeg
Series planes come from <train|test>_series.csv (Anatomical_Plane).

NOTE on augmentation: we deliberately DO NOT use horizontal/vertical flips. Flips swap
medial<->lateral / anterior<->posterior, which would corrupt the laterality-specific
targets (Medial vs Lateral Meniscus/OA). Only small affine + intensity jitter is used.
"""
from __future__ import annotations

import math
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import TARGETS, ID_COL

cv2.setNumThreads(0)  # avoid oversubscription inside DataLoader workers

# Canonical plane ordering for deterministic stratification.
PLANE_ORDER = ["Sagittal", "Coronal", "Axial"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Slice file extensions we understand. JPEG = the alenic 224px training set (local train);
# DICOM = the real Kaggle test mount. The per-image loader is dispatched by extension
# (see _read_image), so the slot sampler below is shared across both formats.
IMG_EXTS = (".jpeg", ".jpg", ".png")
DICOM_EXTS = (".dcm",)
SLICE_EXTS = IMG_EXTS + DICOM_EXTS


# --------------------------------------------------------------------------------------
# Study index (manifest): study -> list of (series_uid, plane, [sorted jpeg paths])
# --------------------------------------------------------------------------------------
def build_study_index(
    root: str,
    series_csv: str,
    split: str = "train",
    cache_path: Optional[str] = None,
) -> Dict[str, List[Tuple[str, str, List[str]]]]:
    """Scan the JPEG tree and return {study: [(series, plane, [paths...]), ...]}.

    Uses <split>_series.csv for the (study, series, plane) triples, then globs each series
    directory for its .jpeg files (sorted for determinism). Cached to `cache_path`.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    ser = pd.read_csv(series_csv)
    base = os.path.join(root, f"{split}_series")
    index: Dict[str, List[Tuple[str, str, List[str]]]] = {}
    for study, plane, series in zip(ser[ID_COL], ser["Anatomical_Plane"], ser["SeriesInstanceUID"]):
        sdir = os.path.join(base, str(study), str(series))
        if not os.path.isdir(sdir):
            continue
        # Accept either the alenic 224px JPEGs (local train/val) OR raw DICOM (Kaggle test
        # mount). Filenames are "<InstanceUID>.<ext>" in both, so a lexicographic sort by
        # filename yields the SAME slice ordering for the .dcm mount as for the .jpeg set the
        # model was trained on (both keyed on InstanceUID) -> the shared slot sampler picks
        # matching positions. Within one series dir there is only one extension.
        files = sorted(fn for fn in os.listdir(sdir) if fn.lower().endswith(SLICE_EXTS))
        if not files:
            continue
        paths = [os.path.join(sdir, fn) for fn in files]
        plane = plane if plane in PLANE_ORDER else "Sagittal"
        index.setdefault(str(study), []).append((str(series), plane, paths))

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(index, f)
    return index


def _split_evenly(total_slots: int, n_bins: int) -> List[int]:
    """Split total_slots into n_bins as evenly as possible (extras to the first bins)."""
    if n_bins <= 0:
        return []
    base, rem = divmod(total_slots, n_bins)
    return [base + (1 if i < rem else 0) for i in range(n_bins)]


def _pick_positions(length: int, n: int, jitter: float = 0.0, rng: Optional[np.random.Generator] = None) -> List[int]:
    """Pick n evenly-spaced, middle-biased indices from range(length). Repeats if n>length."""
    if length <= 0 or n <= 0:
        return []
    out = []
    for i in range(n):
        frac = (i + 0.5) / n
        if jitter and rng is not None:
            frac += (rng.random() - 0.5) * jitter / n
        idx = int(math.floor(frac * length))
        out.append(min(length - 1, max(0, idx)))
    return out


def sample_study_slices(
    entry: List[Tuple[str, str, List[str]]],
    k: int,
    train: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> List[str]:
    """Return exactly K jpeg paths for a study, stratified by plane then by series.

    Slots are split evenly across the planes present, then across each plane's series,
    then evenly-spaced (middle-biased) slices are taken from each series. Pads by
    repeating the collected paths if the study yields fewer than K.
    """
    if not entry:
        return []
    # group series paths by plane, in canonical order
    by_plane: Dict[str, List[List[str]]] = {}
    for _series, plane, paths in entry:
        by_plane.setdefault(plane, []).append(paths)
    planes = [p for p in PLANE_ORDER if p in by_plane]
    for p in by_plane:  # any unexpected plane label
        if p not in planes:
            planes.append(p)

    jitter = 1.0 if train else 0.0
    plane_slots = _split_evenly(k, len(planes))
    chosen: List[str] = []
    for plane, slots in zip(planes, plane_slots):
        series_list = by_plane[plane]
        if train and rng is not None and len(series_list) > 1:
            order = list(rng.permutation(len(series_list)))
            series_list = [series_list[j] for j in order]
        series_slots = _split_evenly(slots, len(series_list))
        for paths, s in zip(series_list, series_slots):
            for pos in _pick_positions(len(paths), s, jitter=jitter, rng=rng):
                chosen.append(paths[pos])

    if not chosen:  # degenerate; fall back to all paths flat
        chosen = [p for _s, _pl, paths in entry for p in paths]
    if len(chosen) < k:
        reps = (k + len(chosen) - 1) // len(chosen)
        chosen = (chosen * reps)[:k]
    elif len(chosen) > k:
        chosen = chosen[:k]
    return chosen


def _read_slice(path: str) -> np.ndarray:
    """Read a 224px grayscale JPEG as a float32 HxW array in [0,1] (the TRAINING loader)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((224, 224), dtype=np.uint8)
    if img.shape != (224, 224):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def _read_slice_dicom(path: str) -> Optional[np.ndarray]:
    """Decode a DICOM slice into the SAME 224px [0,1] float array the model trained on.

    Reverse-engineered from the alenic 224px JPEG set (validated by pixel comparison against
    the real competition DICOM: mean-abs-error ~1.1/255, correlation >0.998 on 9 instances
    across all planes/scanners — see reports/dicom_inference_fix.md). alenic's recipe is:

        pixel_array (apply RescaleSlope/Intercept; invert if MONOCHROME1)
        -> PER-IMAGE min-max normalise to [0,1]      (NOT DICOM window/level, NOT percentile)
        -> cv2.resize to 224x224 with INTER_AREA      (squishes non-square; no aspect crop)
        -> uint8 [0,255]  (we mirror the 8-bit quantisation, then /255)

    Returns float32 HxW in [0,1], or None if the slice cannot be decoded (caller pads).
    """
    try:
        import pydicom  # lazy: keep the JPEG training path free of a hard pydicom dependency
        ds = pydicom.dcmread(path)
        px = ds.pixel_array.astype(np.float32)
    except Exception as e:  # unreadable / missing pixel data -> skip this slice
        print(f"[dicom] skip {os.path.basename(path)}: {type(e).__name__}: {e}")
        return None

    # Multi-frame (rare for MR knee) -> middle frame; RGB -> luminance.
    if px.ndim == 3:
        if px.shape[-1] == 3:  # (H, W, 3) colour
            px = px[..., 0] * 0.299 + px[..., 1] * 0.587 + px[..., 2] * 0.114
        else:                  # (frames, H, W)
            px = px[px.shape[0] // 2]
    if px.ndim != 2:
        return None

    slope = getattr(ds, "RescaleSlope", 1)
    inter = getattr(ds, "RescaleIntercept", 0)
    try:
        px = px * float(slope if slope is not None else 1) + float(inter if inter is not None else 0)
    except Exception:
        pass
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        px = px.max() - px  # MONOCHROME1: high value = black -> invert to match MONOCHROME2

    lo = float(px.min())
    hi = float(px.max())
    if hi <= lo:
        norm = np.zeros_like(px, dtype=np.float32)
    else:
        norm = (px - lo) / (hi - lo)
    resized = cv2.resize((norm * 255.0).astype(np.float32), (224, 224), interpolation=cv2.INTER_AREA)
    u8 = np.clip(np.round(resized), 0, 255).astype(np.float32)  # match stored 8-bit domain
    return u8 / 255.0


def _read_image(path: str) -> Optional[np.ndarray]:
    """Pluggable per-slice loader: DICOM for the Kaggle mount, JPEG for local train/val.

    Auto-detected by file extension so one dataset works on both formats. JPEG reads never
    return None (zeros on a missing file, as before); DICOM reads may return None on a bad
    slice and the dataset backfills by repeating a good slice."""
    if path.lower().endswith(DICOM_EXTS):
        return _read_slice_dicom(path)
    return _read_slice(path)


class StudyBagDataset(Dataset):
    """Per-study bag of K slices -> [K,3,224,224] float tensor + 12 soft targets + id."""

    def __init__(
        self,
        study_ids: Sequence[str],
        index: Dict[str, List[Tuple[str, str, List[str]]]],
        labels: Optional[pd.DataFrame] = None,
        k: int = 24,
        train: bool = False,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        seed: int = 42,
    ):
        self.study_ids = [s for s in study_ids if s in index]
        self.index = index
        self.k = k
        self.train = train
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.seed = seed
        # label lookup: study -> np.float32[12]  (targets = 0 if unknown / inference)
        self.label_map: Dict[str, np.ndarray] = {}
        if labels is not None:
            lab = labels.set_index(ID_COL)[TARGETS]
            for sid in self.study_ids:
                if sid in lab.index:
                    self.label_map[sid] = lab.loc[sid].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.study_ids)

    def _augment(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Light aug: small affine (rotate/scale/translate) + intensity. No flips."""
        h, w = img.shape
        angle = (rng.random() - 0.5) * 2 * 10.0            # +/- 10 deg
        scale = 1.0 + (rng.random() - 0.5) * 2 * 0.10       # +/- 10%
        tx = (rng.random() - 0.5) * 2 * 0.06 * w            # +/- 6% translate
        ty = (rng.random() - 0.5) * 2 * 0.06 * h
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        # intensity: brightness/contrast + gamma
        alpha = 1.0 + (rng.random() - 0.5) * 2 * 0.15       # contrast
        beta = (rng.random() - 0.5) * 2 * 0.10              # brightness
        img = np.clip(img * alpha + beta, 0.0, 1.0)
        gamma = 1.0 + (rng.random() - 0.5) * 2 * 0.15
        img = np.power(img, gamma, dtype=np.float32)
        return img

    def __getitem__(self, i: int):
        sid = self.study_ids[i]
        if self.train:
            rng = np.random.default_rng(self.seed + i + int(torch.randint(0, 1 << 30, (1,)).item()))
        else:
            rng = np.random.default_rng(self.seed + i)
        paths = sample_study_slices(self.index[sid], self.k, train=self.train, rng=rng)
        # Decode all K slots first (DICOM may fail on a bad slice -> None); then backfill any
        # None by repeating a good slice so the bag is never empty / never mixes in zeros when
        # any real slice exists. JPEG reads never return None, so training is unchanged.
        imgs: List[Optional[np.ndarray]] = [_read_image(p) for p in paths]
        valid = [im for im in imgs if im is not None]
        if valid:
            fill = (valid * ((len(imgs) // len(valid)) + 1))
            fi = 0
            for j in range(len(imgs)):
                if imgs[j] is None:
                    imgs[j] = fill[fi]
                    fi += 1
        else:  # extreme fallback: whole study undecodable -> zeros (still a valid, in-range bag)
            imgs = [np.zeros((224, 224), dtype=np.float32) for _ in imgs]
        bag = np.empty((self.k, 3, 224, 224), dtype=np.float32)
        for j, img in enumerate(imgs):
            if self.train:
                img = self._augment(img, rng)
            chw = np.stack([img, img, img], axis=0)  # replicate grayscale to 3ch
            chw = (chw - self.mean) / self.std
            bag[j] = chw
        x = torch.from_numpy(bag)
        y = torch.from_numpy(self.label_map.get(sid, np.zeros(len(TARGETS), dtype=np.float32)))
        return x, y, sid
