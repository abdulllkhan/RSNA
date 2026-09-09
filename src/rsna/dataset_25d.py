"""2.5D per-STUDY bag dataset: each sampled slice becomes [prev, cur, next] RGB channels.

THE IDEA
--------
The 224/384 arms feed every sampled slice as `gray x 3` — the three input channels carry
identical pixels, so an ImageNet-pretrained first conv sees no more than a grayscale image.
The classic knee-MRI trick (MRNet et al.) is to spend those channels on **through-plane
context**: channel 0 = the previous slice, channel 1 = the slice itself, channel 2 = the next
slice, all from the same series in ANATOMICAL order. Local 3-slice context is what a
radiologist uses to tell a torn ligament from a partial-volume artefact, and it is expected
to help exactly the columns the 384 arm is weakest on (MCL ~0.80, Contusion ~0.82,
PF-OA/Lat-OA ~0.84 — EXPERIMENTS.md A2-hi).

Cost is UNCHANGED for the network: still K slots, still 3 channels, still one 2D backbone
forward per slot, still the ImageNet first conv. Only the *decode* triples (3 slices read per
slot instead of 1).

WHAT IS REUSED VERBATIM (imported, never re-implemented)
--------------------------------------------------------
* `dataset.sample_study_slices` — the plane -> series -> evenly-spaced SLOT SAMPLER. The
  centre channel of slot j is *bit-identically the slice the 224/384 arms put in slot j*
  (`slot_instance_uids()` is inherited from `StudyBagDatasetHR` and proves it; see
  `reports/arm_25d.md` §3).
* `dataset_hr.read_slice_hr` — the one validated per-slice reader (DICOM -> `decode`, else
  cached JPEG), size-parametric.
* `StudyBagDatasetHR` — subclassed, so the label map, the RNG seeding and the
  no-flip augmentation POLICY are the HR arm's own code.

WHAT IS NEW
-----------
1. **Neighbour lookup through `order.SliceOrderResolver`.** This is the correctness crux:
   the study index sorts a series' files lexicographically by InstanceUID, and that order is
   NOT anatomical — measured 0/400 train and 0/15 test series match it, mean |Spearman| 0.16
   (see `order.py`). Taking `paths[i±1]` would pair each slice with two random other slices.
   Order therefore comes from DICOM headers (cached in `data/series_order.parquet` for the
   JPEG training caches, read live from the `.dcm` mount at inference).
2. **One shared augmentation per slot.** The parent augments a single 2-D image; here the
   SAME affine + intensity transform must hit all three channels or the through-plane
   context is destroyed. `_augment_stack` replicates the parent's arithmetic on an (H,W,3)
   array and draws the same six random numbers in the same order, so with a replicated
   grayscale stack it is bit-identical to `StudyBagDataset._augment` (asserted in
   `dev`-free form by the smoke in `reports/arm_25d.md` §5).
3. **Edge replication**: the first slice of a stack has `prev = itself`, the last has
   `next = itself` — never a wrap-around, never a zero channel.

USE
---
    from src.rsna.dataset_25d import StudyBagDataset25D, build_order_resolver
    from src.rsna.dataset_hr import build_hr_index
    index = build_hr_index("data/cache384", split="train")
    order = build_order_resolver(parquet="data/series_order.parquet")   # or dicom_root=...
    ds = StudyBagDataset25D(ids, index, labels, k=24, train=True, size=384,
                            mean=m, std=s, order=order)
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from . import TARGETS, ID_COL  # noqa: F401  (re-exported for callers)
from .dataset import IMAGENET_MEAN, IMAGENET_STD, sample_study_slices
from .dataset_hr import StudyBagDatasetHR, build_hr_index, default_cache_root, \
    read_slice_hr, resolve_series_csv, slot_instance_uids  # noqa: F401
from .order import (
    DEFAULT_ORDER_MODE,
    DEFAULT_ORDER_PARQUET,
    SliceOrderResolver,
    stem_of,
)

cv2.setNumThreads(0)  # avoid oversubscription inside DataLoader workers

MODE_TAG = "2.5d"  # written into the checkpoint's args by train_25d; read back by infer_25d


def build_order_resolver(
    parquet: Optional[str] = DEFAULT_ORDER_PARQUET,
    dicom_root: Optional[str] = None,
    split: str = "train",
    mode: str = DEFAULT_ORDER_MODE,
    allow_lex: bool = True,
) -> SliceOrderResolver:
    """Convenience constructor — parquet first, DICOM headers second, lex last (warned).

    Build it in the MAIN process, before the DataLoader forks: the compact rank vectors are
    then shared copy-on-write by every worker instead of being loaded once per worker.
    """
    return SliceOrderResolver.from_parquet(parquet, dicom_root=dicom_root, split=split,
                                           mode=mode, allow_lex=allow_lex)


class StudyBagDataset25D(StudyBagDatasetHR):
    """Per-study bag of K slots -> `[K, 3, size, size]`, channels = (prev, cur, next).

    Arguments beyond `StudyBagDatasetHR`:
        order:           a `SliceOrderResolver` (defaults to a header/lex-only one, which
                         WARNS on a JPEG tree — always pass one built with the parquet or a
                         `dicom_root` for training).
        neighbour_step:  channels are (cur-step, cur, cur+step) in anatomical order. 1 keeps
                         the tightest context; 2 widens it for thick-slice series.
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
        order: Optional[SliceOrderResolver] = None,
        neighbour_step: int = 1,
    ):
        super().__init__(study_ids, index, labels=labels, k=k, train=train, size=size,
                         mean=mean, std=std, seed=seed)
        self.order = order if order is not None else SliceOrderResolver()
        self.neighbour_step = max(1, int(neighbour_step))

    # ---------------------------------------------------------------------------------
    # neighbour resolution
    # ---------------------------------------------------------------------------------
    def _positions(self, sid: str) -> Dict[str, Tuple[int, int]]:
        """`{path: (series_slot, index_within_series)}` for one study (built per item; the
        study has ~150 slices, so this is negligible next to decoding 72 of them)."""
        pos: Dict[str, Tuple[int, int]] = {}
        for si, (_series, _plane, paths) in enumerate(self.index[sid]):
            for j, p in enumerate(paths):
                pos[p] = (si, j)
        return pos

    def triples_for(self, sid: str, paths: Sequence[str]) -> List[Tuple[str, str, str]]:
        """Map each sampled path to `(prev_path, path, next_path)` in ANATOMICAL order.

        A path that is somehow not in the study index (cannot happen for paths produced by
        the sampler, but a caller may pass its own) degrades to `(p, p, p)`.
        """
        entry = self.index[sid]
        pos = self._positions(sid)
        out: List[Tuple[str, str, str]] = []
        nb_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for p in paths:
            hit = pos.get(p)
            if hit is None:
                out.append((p, p, p))
                continue
            si, j = hit
            if si not in nb_cache:
                series, _plane, sp = entry[si]
                nb_cache[si] = self.order.neighbours(series, sp, step=self.neighbour_step)
            prev_idx, next_idx = nb_cache[si]
            sp = entry[si][2]
            out.append((sp[int(prev_idx[j])], p, sp[int(next_idx[j])]))
        return out

    def slot_triples(self, i: int) -> List[Tuple[str, str, str]]:
        """The (prev, cur, next) InstanceUIDs for eval item `i` — inspection/proof helper.

        `[t[1] for t in slot_triples(i)]` is exactly `slot_instance_uids(i)`, i.e. the HR
        arm's slot list.
        """
        sid = self.study_ids[i]
        rng = np.random.default_rng(self.seed + i)
        paths = sample_study_slices(self.index[sid], self.k, train=False, rng=rng)
        return [tuple(stem_of(q) for q in t) for t in self.triples_for(sid, paths)]

    # ---------------------------------------------------------------------------------
    # augmentation (parent's arithmetic, applied ONCE to the 3-channel stack)
    # ---------------------------------------------------------------------------------
    def _augment_stack(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """`StudyBagDataset._augment` for an (H, W, 3) stack: ONE transform for all channels.

        The six `rng.random()` draws happen in the same order and are used identically, so
        for a replicated grayscale stack the result equals the parent's per-slice output
        channel-for-channel. Still NO flips: they would swap medial<->lateral /
        anterior<->posterior and corrupt the laterality-specific targets.
        """
        h, w = img.shape[:2]
        angle = (rng.random() - 0.5) * 2 * 10.0            # +/- 10 deg
        scale = 1.0 + (rng.random() - 0.5) * 2 * 0.10       # +/- 10%
        tx = (rng.random() - 0.5) * 2 * 0.06 * w            # +/- 6% translate
        ty = (rng.random() - 0.5) * 2 * 0.06 * h
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        alpha = 1.0 + (rng.random() - 0.5) * 2 * 0.15       # contrast
        beta = (rng.random() - 0.5) * 2 * 0.10              # brightness
        img = np.clip(img * alpha + beta, 0.0, 1.0)
        gamma = 1.0 + (rng.random() - 0.5) * 2 * 0.15
        return np.power(img, gamma, dtype=np.float32)

    # ---------------------------------------------------------------------------------
    def __getitem__(self, i: int):
        sid = self.study_ids[i]
        # RNG seeding copied EXACTLY from StudyBagDataset.__getitem__ -> the same study at the
        # same item index picks the same slots as the 224/384 arms (and, in train mode, the
        # same six augmentation draws per slot).
        if self.train:
            rng = np.random.default_rng(self.seed + i + int(torch.randint(0, 1 << 30, (1,)).item()))
        else:
            rng = np.random.default_rng(self.seed + i)
        paths = sample_study_slices(self.index[sid], self.k, train=self.train, rng=rng)
        triples = self.triples_for(sid, paths)

        size = self.size
        # Decode slot by slot into (H, W, 3) stacks. A slot is invalid only if its CENTRE
        # slice is undecodable (the centre is the slice the other arms would have used); an
        # undecodable NEIGHBOUR just replicates the centre, exactly like an edge slice.
        stacks: List[Optional[np.ndarray]] = []
        for prev_p, cur_p, next_p in triples:
            cur = read_slice_hr(cur_p, size)
            if cur is None:
                stacks.append(None)
                continue
            prv = cur if prev_p == cur_p else read_slice_hr(prev_p, size)
            nxt = cur if next_p == cur_p else read_slice_hr(next_p, size)
            stacks.append(np.stack([cur if prv is None else prv,
                                    cur,
                                    cur if nxt is None else nxt], axis=-1))

        valid = [s for s in stacks if s is not None]
        if valid:  # backfill unreadable slots by repeating good ones (never mix in zeros)
            fill = valid * ((len(stacks) // len(valid)) + 1)
            fi = 0
            for j in range(len(stacks)):
                if stacks[j] is None:
                    stacks[j] = fill[fi]
                    fi += 1
        else:  # whole study undecodable -> zeros (still a valid, in-range bag)
            stacks = [np.zeros((size, size, 3), dtype=np.float32) for _ in stacks]

        bag = np.empty((self.k, 3, size, size), dtype=np.float32)
        for j, st in enumerate(stacks):
            if self.train:
                st = self._augment_stack(st, rng)   # ONE transform across (prev, cur, next)
            chw = np.ascontiguousarray(st.transpose(2, 0, 1))
            chw = (chw - self.mean) / self.std
            bag[j] = chw
        x = torch.from_numpy(bag)
        y = torch.from_numpy(self.label_map.get(sid, np.zeros(len(TARGETS), dtype=np.float32)))
        return x, y, sid


__all__ = [
    "StudyBagDataset25D",
    "build_order_resolver",
    "build_hr_index",
    "default_cache_root",
    "resolve_series_csv",
    "read_slice_hr",
    "slot_instance_uids",
    "MODE_TAG",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
