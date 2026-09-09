"""Slot-sampling TTA: S deterministic re-samplings of the same study bag.

PLAN §9 lists "TTA over 2-3 slot samplings" among the cheap wins. This module implements it
WITHOUT touching `dataset.py` / `dataset_hr.py` (a trainer imports them) and without touching
`infer*.py` (the notebook path inlines them).

WHAT IS BEING VARIED
--------------------
At eval the slot sampler is fully deterministic: `dataset.sample_study_slices(entry, k,
train=False, rng=...)` never reads `rng` — it splits K slots across planes, then across each
plane's series, then calls `dataset._pick_positions(length, n, jitter=0.0)`, which returns the
*midpoints* of n equal bins over the series:

        frac_i = (i + 0.5) / n            idx_i = floor(frac_i * length)          (i = 0..n-1)

So a study always contributes the SAME K slices. The model therefore sees one fixed sample of
each series; a different (equally valid) sample would give a slightly different logit. TTA
averages over several such samples to cut that sampling variance.

A "variant" here is a phase shift of that sampling grid, expressed in units of the inter-slice
spacing (`length / n` slices):

        mid  (default):  frac_i = (i + 0.5 + phase) / n            phase = 0 -> the baseline
        span          :  frac_i = (i + phase) / (n - 1)            endpoints included

`phase = +1/3` moves every picked index a third of a bin deeper into the series; `-1/3` a third
earlier. `span` is a different *shape* of grid (uniform across the whole series including its
first and last slice) rather than a shift of the midpoint grid, which decorrelates a little more
than a pure phase shift. Nothing else changes: the plane -> series slot allocation, the pad /
truncate rule, the decode, the normalisation and the model are untouched.

Deliberately NOT varied: horizontal/vertical flips (they swap medial<->lateral and
anterior<->posterior and would corrupt the laterality-specific targets -- see `dataset.py`), and
the train-mode affine/intensity jitter (TTA here is eval-only, so the bags stay reproducible).

VARIANT 0 IS THE BASELINE, BY CONSTRUCTION
------------------------------------------
`sampling_variants(S)[0]` is always `phase=0, mode="mid"`, and the identity variant is
short-circuited: the wrappers call the *unmodified* parent `__getitem__` with no patch at all.
So `S=1` reproduces the existing predictions bit-for-bit rather than "to within rounding".
`python -m src.rsna.tta` runs the proof (identity == baseline paths; variants differ).

HOW A VARIANT IS APPLIED
------------------------
`sample_study_slices` looks `_pick_positions` up as a module global at call time, so a scoped
rebind of that ONE name in `sample_study_slices.__globals__` re-points exactly one primitive and
reuses every other line of the real sampler -- no copy of the plane/series slot logic that could
drift out of sync. Going through the function's own `__globals__` (rather than naming the
`dataset` module) is what keeps this working inside the Kaggle notebook, where
`build_notebook_multi.py` inlines every module into ONE flat namespace and strips the relative
imports. The patch is applied inside `__getitem__` (one DataLoader worker = one process = one
thread, and `cv2.setNumThreads(0)` keeps it that way), and is always undone in a `finally`.

    Thread-safety: do NOT run two different variants concurrently *inside one process*
    (multi-process DataLoader workers are fine, and are how this is actually used).

USE
---
    from src.rsna.tta import sampling_variants, TTAStudyBagDatasetHR
    for v in sampling_variants(3):                 # v0 == the current eval sampling
        ds = TTAStudyBagDatasetHR(ids, index, k=24, size=384, mean=m, std=s, variant=v)
        ...                                        # predict, then rank-average the S frames

`infer_tta.predict_studies_tta` does the loop + the rank-average; `dev/tta_eval.py` measures
whether it is worth the S x inference cost on OOF before it is paid for on Kaggle.
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dataset import StudyBagDataset, sample_study_slices
from .dataset_hr import StudyBagDatasetHR

# The namespace `sample_study_slices` resolves `_pick_positions` in. In the package this is
# `src.rsna.dataset.__dict__`; in the inlined Kaggle bundle it is the one flat module namespace.
# Either way it is the exact dict the sampler reads, which is what makes the swap below correct.
SAMPLER_GLOBALS = sample_study_slices.__globals__

__all__ = [
    "SAMPLER_GLOBALS",
    "SamplingVariant",
    "IDENTITY_VARIANT",
    "VARIANT_LADDER",
    "sampling_variants",
    "pick_positions_variant",
    "variant_sampling",
    "sample_study_slices_variant",
    "slot_instance_uids_variant",
    "TTAStudyBagDataset",
    "TTAStudyBagDatasetHR",
    "dataset_factory",
    "self_check",
]


# --------------------------------------------------------------------------------------
# Variant definition
# --------------------------------------------------------------------------------------
class SamplingVariant:
    """One deterministic slot-sampling configuration (immutable, hashable, comparable).

    idx   position in the ladder (0 == the baseline eval sampling; used for file naming)
    phase shift of the sampling grid, in units of the inter-slice spacing (length / n)
    mode  "mid"  -> midpoint grid  frac_i = (i + 0.5 + phase) / n      (the baseline shape)
          "span" -> endpoint grid  frac_i = (i + phase) / (n - 1)      (includes first+last)

    Hand-written rather than a `@dataclass` on purpose. Under `from __future__ import annotations`
    (which `build_notebook_multi.py` prepends to the inlined bundle) the dataclass machinery
    resolves string annotations through `sys.modules[cls.__module__]`, so it needs the executing
    namespace to be a *registered module*. A Kaggle cell is `__main__` and satisfies that -- the
    bundle already carries `@dataclass ModelConfig` -- but any harness that exec's the bundle into
    a bare dict does not, and a 0.5-file fallback is too expensive to spend on that dependency.
    """

    __slots__ = ("idx", "phase", "mode")

    def __init__(self, idx: int, phase: float, mode: str = "mid"):
        object.__setattr__(self, "idx", int(idx))
        object.__setattr__(self, "phase", float(phase))
        object.__setattr__(self, "mode", str(mode))

    def __setattr__(self, *_a):  # immutable
        raise AttributeError("SamplingVariant is immutable")

    def _key(self):
        return (self.idx, self.phase, self.mode)

    def __eq__(self, other):
        return isinstance(other, SamplingVariant) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return f"SamplingVariant(idx={self.idx}, phase={self.phase!r}, mode={self.mode!r})"

    def __reduce__(self):
        # __slots__ + a raising __setattr__ would break the default pickle path (it setattr's the
        # slots back). DataLoader pickles the dataset -- and with it self.variant -- under a
        # `spawn` start method, so spell the reconstruction out.
        return (SamplingVariant, (self.idx, self.phase, self.mode))

    @property
    def is_identity(self) -> bool:
        """True iff this variant reproduces `dataset._pick_positions` exactly."""
        return self.mode == "mid" and self.phase == 0.0

    @property
    def name(self) -> str:
        return f"v{self.idx}"

    @property
    def label(self) -> str:
        return f"v{self.idx}[{self.mode} phase={self.phase:+.4f}]" + \
               (" == baseline eval sampling" if self.is_identity else "")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.label


IDENTITY_VARIANT = SamplingVariant(idx=0, phase=0.0, mode="mid")

# Fixed ladder: a run with S variants uses the first S entries, so S=2's variants are a prefix
# of S=3's and the per-(fold, variant) prediction cache is shared across S.
#   v0  baseline (midpoint grid)
#   v1  +1/3 spacing  (a third of a slice-gap deeper into each series)
#   v2  -1/3 spacing
#   v3  endpoint-inclusive uniform grid (different grid *shape*, not just a shift)
#   v4  +1/6 spacing
#   v5  -1/6 spacing
VARIANT_LADDER: Tuple[Tuple[float, str], ...] = (
    (0.0, "mid"),
    (+1.0 / 3.0, "mid"),
    (-1.0 / 3.0, "mid"),
    (0.0, "span"),
    (+1.0 / 6.0, "mid"),
    (-1.0 / 6.0, "mid"),
)


def sampling_variants(n: int, seed: int = 42) -> List[SamplingVariant]:
    """The first `n` deterministic sampling variants; `[0]` is always the baseline.

    `seed` only affects variants BEYOND the fixed ladder (n > len(VARIANT_LADDER)), which are
    drawn as uniform phases in (-0.5, +0.5) from `np.random.default_rng(seed)`. Keeping the
    first six fixed means a cached v1 frame from one run is reusable by any other run.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    out = [SamplingVariant(i, ph, md) for i, (ph, md) in enumerate(VARIANT_LADDER[:n])]
    if n > len(VARIANT_LADDER):
        rng = np.random.default_rng(seed)
        for i in range(len(VARIANT_LADDER), n):
            out.append(SamplingVariant(i, float(rng.uniform(-0.5, 0.5)), "mid"))
    assert out[0].is_identity, "variant 0 must be the baseline eval sampling"
    return out


# --------------------------------------------------------------------------------------
# The one primitive that changes
# --------------------------------------------------------------------------------------
def pick_positions_variant(
    length: int,
    n: int,
    phase: float = 0.0,
    mode: str = "mid",
) -> List[int]:
    """Variant of `dataset._pick_positions` (jitter/rng dropped -- TTA is eval-only).

    With `phase=0.0, mode="mid"` this is arithmetically identical to
    `dataset._pick_positions(length, n, jitter=0.0, rng=None)`: same `frac`, same `floor`,
    same clamp. `self_check()` asserts that over a grid of (length, n).
    """
    if length <= 0 or n <= 0:
        return []
    out: List[int] = []
    for i in range(n):
        if mode == "span" and n > 1:
            # endpoint-inclusive grid; clamp just under 1.0 so floor() cannot reach `length`
            frac = (i + phase) / (n - 1)
            frac = min(max(frac, 0.0), 1.0 - 1e-12)
        elif mode in ("mid", "span"):
            frac = (i + 0.5 + phase) / n
        else:
            raise ValueError(f"unknown variant mode {mode!r} (expected 'mid' or 'span')")
        idx = int(math.floor(frac * length))
        out.append(min(length - 1, max(0, idx)))
    return out


@contextmanager
def variant_sampling(variant: Optional[SamplingVariant]):
    """Scope in which `dataset._pick_positions` implements `variant`.

    The identity variant patches nothing at all, so code inside the block runs the literal
    baseline sampler. Always restores the original in `finally`.
    """
    if variant is None or variant.is_identity:
        yield
        return
    orig = SAMPLER_GLOBALS["_pick_positions"]

    def _pick(length, n, jitter=0.0, rng=None):  # signature-compatible with the original
        return pick_positions_variant(length, n, phase=variant.phase, mode=variant.mode)

    SAMPLER_GLOBALS["_pick_positions"] = _pick
    try:
        yield
    finally:
        SAMPLER_GLOBALS["_pick_positions"] = orig


def sample_study_slices_variant(
    entry: Sequence[Tuple[str, str, List[str]]],
    k: int,
    variant: Optional[SamplingVariant] = None,
) -> List[str]:
    """The K paths this variant picks for a study (eval mode; the real sampler, one primitive swapped)."""
    with variant_sampling(variant):
        return sample_study_slices(list(entry), k, train=False, rng=None)


def slot_instance_uids_variant(
    entry: Sequence[Tuple[str, str, List[str]]],
    k: int,
    variant: Optional[SamplingVariant] = None,
) -> List[str]:
    """`sample_study_slices_variant` reduced to bare InstanceUIDs (comparable across caches)."""
    return [os.path.splitext(os.path.basename(p))[0]
            for p in sample_study_slices_variant(entry, k, variant)]


# --------------------------------------------------------------------------------------
# Dataset wrappers (subclasses -- dataset.py / dataset_hr.py are NOT edited)
# --------------------------------------------------------------------------------------
class _VariantMixin:
    """Applies `self.variant` around the parent's `__getitem__` (eval only)."""

    variant: SamplingVariant

    def _set_variant(self, variant: Optional[SamplingVariant], train: bool) -> None:
        if train:
            raise ValueError("TTA datasets are eval-only (train=True would add its own jitter)")
        self.variant = variant or IDENTITY_VARIANT

    def slot_paths(self, i: int) -> List[str]:
        """The exact paths this dataset feeds for item `i` (for the identity proof)."""
        return sample_study_slices_variant(self.index[self.study_ids[i]], self.k, self.variant)

    def slot_instance_uids(self, i: int) -> List[str]:
        return slot_instance_uids_variant(self.index[self.study_ids[i]], self.k, self.variant)

    def __getitem__(self, i: int):
        if self.variant.is_identity:
            return super().__getitem__(i)  # literal baseline path, no patch in scope
        with variant_sampling(self.variant):
            return super().__getitem__(i)


class TTAStudyBagDataset(_VariantMixin, StudyBagDataset):
    """`dataset.StudyBagDataset` with a TTA sampling variant. `variant=None` -> identical."""

    def __init__(self, *args, variant: Optional[SamplingVariant] = None, **kwargs):
        super().__init__(*args, **kwargs)          # parent default is train=False
        self._set_variant(variant, bool(self.train))


class TTAStudyBagDatasetHR(_VariantMixin, StudyBagDatasetHR):
    """`dataset_hr.StudyBagDatasetHR` with a TTA sampling variant. `variant=None` -> identical."""

    def __init__(self, *args, variant: Optional[SamplingVariant] = None, **kwargs):
        super().__init__(*args, **kwargs)          # parent default is train=False
        self._set_variant(variant, bool(self.train))


def dataset_factory(base_cls, variant: Optional[SamplingVariant]):
    """A drop-in replacement for `base_cls` that binds `variant`.

    `infer.predict_studies` / `infer_hr.predict_studies_hr` construct their dataset by a module
    global lookup, so `infer_tta` swaps that global for `dataset_factory(StudyBagDataset, v)`
    and reuses the whole decode-once / member-loop / autocast body unchanged.
    """
    wrapper = {StudyBagDataset: TTAStudyBagDataset, StudyBagDatasetHR: TTAStudyBagDatasetHR}.get(base_cls)
    if wrapper is None:
        raise TypeError(f"no TTA wrapper for {base_cls!r}")

    def _make(*args, **kwargs):
        return wrapper(*args, variant=variant, **kwargs)

    return _make


# --------------------------------------------------------------------------------------
# Self-check (identity proof, no data needed)
# --------------------------------------------------------------------------------------
def _synthetic_entry(n_series: int = 4, seed: int = 0) -> List[Tuple[str, str, List[str]]]:
    rng = np.random.default_rng(seed)
    planes = ["Sagittal", "Coronal", "Axial"]
    entry = []
    for s in range(n_series):
        length = int(rng.integers(8, 60))
        plane = planes[s % len(planes)]
        entry.append((f"ser{s}", plane, [f"/x/ser{s}/inst{j:04d}.jpeg" for j in range(length)]))
    return entry


def self_check(verbose: bool = True) -> Dict[str, object]:
    """Prove (a) v0 == the baseline sampler, (b) v1.. differ, (c) the patch is always undone."""
    ok_prim = True
    for length in range(1, 80):
        for n in range(1, 33):
            a = SAMPLER_GLOBALS["_pick_positions"](length, n, jitter=0.0, rng=None)
            b = pick_positions_variant(length, n, phase=0.0, mode="mid")
            if a != b:
                ok_prim = False
                if verbose:
                    print(f"  MISMATCH length={length} n={n}: {a} != {b}")
    variants = sampling_variants(6)
    changed = {}
    ident_ok = True
    for seed in range(12):
        for k in (8, 24, 32):
            entry = _synthetic_entry(seed=seed)
            base = sample_study_slices(entry, k, train=False, rng=None)
            for v in variants:
                got = sample_study_slices_variant(entry, k, v)
                if v.is_identity:
                    ident_ok &= (got == base)
                else:
                    changed.setdefault(v.name, 0)
                    changed[v.name] += sum(1 for x, y in zip(got, base) if x != y)
    # the patch must never leak out of the context manager
    before = SAMPLER_GLOBALS["_pick_positions"]
    try:
        with variant_sampling(variants[1]):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    restored = SAMPLER_GLOBALS["_pick_positions"] is before

    res = {"primitive_identical": ok_prim, "variant0_identical": bool(ident_ok),
           "slots_changed_per_variant": changed, "patch_restored": restored}
    if verbose:
        print("[tta.self_check] _pick_positions(phase=0,mid) == dataset._pick_positions:", ok_prim)
        print("[tta.self_check] variant 0 bags == baseline eval bags:", bool(ident_ok))
        print("[tta.self_check] slots changed vs baseline (over 36 synthetic study/k combos):",
              changed)
        print("[tta.self_check] patch restored after an exception:", restored)
        for v in variants:
            print("   ", v.label)
    return res


if __name__ == "__main__":  # pragma: no cover
    r = self_check()
    ok = r["primitive_identical"] and r["variant0_identical"] and r["patch_restored"] \
        and all(c > 0 for c in r["slots_changed_per_variant"].values())
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
