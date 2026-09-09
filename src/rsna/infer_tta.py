"""Slot-sampling TTA at inference: run the SAME ensemble over S deterministic samplings.

`infer.predict_studies` (224 baseline) and `infer_hr.predict_studies_hr` (size-aware) each
decode a study ONCE into one fixed bag of K slices and average the member sigmoids. This module
runs that whole pipeline S times, once per `tta.SamplingVariant`, and per-column rank-averages
the S finished frames -- the PLAN §9 "TTA over 2-3 slot samplings" cheap win.

    per variant v:   decode once per (resolution, k, mean, std) group -> every member forward
                     -> `mode` combine members (prob mean by default)      == the existing path
    across variants: per-column percentile-rank average                    == the new part

Rank (not probability) across variants for the same reason PLAN §9 gives for heterogeneous
arms: only within-column order reaches the metric, and a rank average cannot be dragged by one
variant's scale. `variant_combine="prob"` is available for a like-for-like comparison.

NO EDIT to `infer.py` / `infer_hr.py` / `dataset*.py`. The variant is injected by rebinding, for
the duration of one call, the dataset name that `predict_studies*` looks up when it builds its
loader -- in `predict_studies.__globals__`, i.e. the predictor's OWN namespace. That indirection
matters twice over: it reuses the decode-once grouping, the capability-aware autocast, the fp32
retry backstop, the member weighting and the study-id intersection verbatim instead of
re-implementing them; and it works unchanged inside the Kaggle notebook, where
`build_notebook_multi.py` inlines every module into one flat namespace (so a `module.attr`
patch would have nothing to point at). `variants=1` short-circuits to the identity variant,
which patches nothing -- S=1 output is bit-identical to calling `predict_studies` /
`predict_studies_hr` directly.

COST. Inference time is ~S x the single-pass time (decode + forward both repeat; the study
index is built once and reused). Everything else -- weights loaded, RAM, submission writing --
is unchanged. Measure the OOF gain with `dev/tta_eval.py` BEFORE paying S x on Kaggle.

USE (local check against a decoded cache)

    from src.rsna.infer_hr import load_members_hr
    from src.rsna.infer_tta import predict_studies_tta
    ms = load_members_hr(["models/effv2s_k24_384_fold0.pt"], device="cpu")
    preds = predict_studies_tta("data/cache384", "train", ms, variants=3,
                                batch=2, workers=2, device="cpu", study_ids=ids)

Wiring into `notebook/build_notebook_multi.py` behind `RSNA_TTA=S`: see reports/tta_tooling.md
(that file is the working submission path and is NOT edited here -- the report carries the patch).
"""
from __future__ import annotations

import inspect
from contextlib import ExitStack, contextmanager
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import TARGETS, ID_COL
from .dataset import StudyBagDataset
from .dataset_hr import StudyBagDatasetHR
from .infer import predict_studies
from .infer_hr import predict_studies_hr
from .tta import SamplingVariant, dataset_factory, sampling_variants

__all__ = [
    "predict_studies_tta",
    "predict_studies_base_tta",
    "rank_average_variant_frames",
    "resolve_variants",
    "cpu_pin_memory_off",
]


# --------------------------------------------------------------------------------------
# Combining the S frames
# --------------------------------------------------------------------------------------
def rank_average_variant_frames(
    frames: Sequence[pd.DataFrame],
    weights: Optional[Sequence[float]] = None,
    mode: str = "rank",
) -> pd.DataFrame:
    """Per-column percentile-rank (or probability) average of `[ID_COL + 12]` frames.

    Same combiner as `infer_plane.rank_average_frames` (frames inner-joined on ID_COL, order
    from the first, each column ranked WITHIN its own frame, then weighted-averaged). Kept local
    -- and under a DISTINCT name -- so this module needs neither `infer_plane` nor the plane
    dataset/model modules, and so that inlining it into the flat Kaggle bundle cannot shadow
    `infer_plane.rank_average_frames`, which the multi-arm driver calls with `per_column_weights`.
    `dev/tta_eval.py --check-combiner` asserts the two agree on real frames.
    """
    if not frames:
        raise ValueError("no frames")
    if mode not in ("rank", "prob"):
        raise ValueError(f"mode must be 'rank' or 'prob', got {mode!r}")
    w = np.ones(len(frames)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(w) != len(frames):
        raise ValueError("weights must match frames")
    ids = [str(s) for s in frames[0][ID_COL].astype(str)]
    for f in frames[1:]:
        keep = set(f[ID_COL].astype(str))
        ids = [s for s in ids if s in keep]
    acc = np.zeros((len(ids), len(TARGETS)), dtype=np.float64)
    for f, wi in zip(frames, w):
        g = f.copy()
        g[ID_COL] = g[ID_COL].astype(str)
        g = g.set_index(ID_COL).loc[ids][list(TARGETS)]
        vals = g.rank(pct=True).values if mode == "rank" else g.values.astype(np.float64)
        acc += wi * vals
    acc /= max(1e-12, float(w.sum()))
    out = pd.DataFrame(acc, columns=list(TARGETS))
    out.insert(0, ID_COL, ids)
    return out


def resolve_variants(variants: Union[int, Sequence[SamplingVariant]],
                     seed: int = 42) -> List[SamplingVariant]:
    """`3` -> the first 3 ladder variants; an explicit list passes through unchanged."""
    if isinstance(variants, int):
        return sampling_variants(variants, seed=seed)
    out = list(variants)
    if not out:
        raise ValueError("no variants")
    return out


# --------------------------------------------------------------------------------------
# Injection: bind a variant (and, optionally, a prebuilt study index) into the real predictors
# --------------------------------------------------------------------------------------
@contextmanager
def _tta_swap(ns: dict, name: str, value):
    """Rebind ONE name in a namespace dict for the duration of the block, then restore it."""
    missing = object()
    orig = ns.get(name, missing)
    ns[name] = value
    try:
        yield
    finally:
        if orig is missing:
            ns.pop(name, None)
        else:
            ns[name] = orig


def _tta_index_stub(index, builder_name: str):
    """A stand-in for the predictor's index builder that returns a prebuilt manifest.

    `predict_studies*` rebuild the study index on every call (`cache_path=None`), which means
    a full re-scan of ~4.4k study dirs per variant. Passing `index=` reuses one scan for all S.
    """
    def _build(*_a, **_kw):
        return index
    _build.__name__ = builder_name
    return _build


def _tta_no_pin(orig_loader):
    """`infer.predict_studies` hardcodes `pin_memory=True`; on CPU that pins to CUDA and dies
    with `cudaErrorMemoryAllocation` whenever the GPU is busy (e.g. a training run). Forcing it
    off is the ONLY behavioural difference this module makes to a CPU run -- pinning affects
    host->device transfer speed, never the numbers."""
    def _dl(*a, **kw):
        kw["pin_memory"] = False
        return orig_loader(*a, **kw)
    return _dl


def _target_globals(fn) -> dict:
    """The namespace a predictor's body actually resolves names in.

    `predict_studies` / `predict_studies_hr` are decorated with `@torch.no_grad()`, so the
    exported object is `torch.utils._contextlib.decorate_context` and ITS `__globals__` is
    torch's, not the module's. `inspect.unwrap` follows the `functools.wraps` chain back to the
    real function. Getting this wrong is a silent no-op at best (a KeyError on 'DataLoader' at
    worst), so it is done in one place.
    """
    return inspect.unwrap(fn).__globals__


@contextmanager
def cpu_pin_memory_off(predictor, device: str):
    """Public scope of the `_tta_no_pin` fix, for callers that run a raw predictor on CPU.

    `predictor` is the function whose namespace to patch, e.g. `infer.predict_studies`.
    """
    if str(device).startswith("cuda"):
        yield
        return
    ns = _target_globals(predictor)
    with _tta_swap(ns, "DataLoader", _tta_no_pin(ns["DataLoader"])):
        yield


@contextmanager
def _tta_bind(predictor, dataset_name: str, base_cls, variant: Optional[SamplingVariant],
              index_name: Optional[str] = None, index=None, device: str = "cuda"):
    """Bind a variant (+ optional prebuilt index, + the CPU pin fix) into `predictor`'s namespace."""
    ns = _target_globals(predictor)
    with ExitStack() as st:
        if variant is not None and not variant.is_identity:
            st.enter_context(_tta_swap(ns, dataset_name, dataset_factory(base_cls, variant)))
        if index is not None and index_name:
            st.enter_context(_tta_swap(ns, index_name, _tta_index_stub(index, index_name)))
        if not str(device).startswith("cuda"):
            st.enter_context(_tta_swap(ns, "DataLoader", _tta_no_pin(ns["DataLoader"])))
        yield


# --------------------------------------------------------------------------------------
# Public predictors
# --------------------------------------------------------------------------------------
def predict_studies_tta(
    input_root: Union[str, Mapping[int, str]],
    split: str,
    members: Sequence,
    variants: Union[int, Sequence[SamplingVariant]] = 3,
    batch: int = 8,
    workers: int = 8,
    device: str = "cuda",
    study_ids: Optional[Sequence[str]] = None,
    mode: str = "prob",
    weights: Optional[Sequence[float]] = None,
    variant_combine: str = "rank",
    variant_weights: Optional[Sequence[float]] = None,
    seed: int = 42,
    index=None,
    return_frames: bool = False,
    verbose: bool = True,
):
    """Size-aware ensemble prediction, TTA'd over S slot samplings.

    `members` are `infer_hr.MemberHR` (they carry their own `size`, so a 224 checkpoint works
    here too). Every argument except the four `variant*`/`index` ones is forwarded to
    `infer_hr.predict_studies_hr` untouched.

    Returns the combined `[ID_COL + 12]` frame, or `(combined, {variant_idx: frame})` when
    `return_frames=True` (dev/tta_eval.py caches the per-variant frames from that).
    """
    if index is not None and isinstance(input_root, Mapping) and len(set(input_root.values())) > 1:
        raise ValueError("index= reuses ONE manifest for every decode group; drop it (or call "
                         "once per root) when input_root maps sizes to different trees")
    vs = resolve_variants(variants, seed=seed)
    frames: Dict[int, pd.DataFrame] = {}
    for v in vs:
        if verbose:
            print(f"[tta] variant {v.label}")
        with _tta_bind(predict_studies_hr, "StudyBagDatasetHR", StudyBagDatasetHR, v,
                       index_name="build_hr_index", index=index, device=device):
            frames[v.idx] = predict_studies_hr(
                input_root, split, members, batch=batch, workers=workers, device=device,
                study_ids=study_ids, mode=mode, weights=weights)
    combined = _tta_combine(frames, vs, variant_combine, variant_weights, verbose)
    return (combined, frames) if return_frames else combined


def predict_studies_base_tta(
    input_root: str,
    split: str,
    members: Sequence,
    variants: Union[int, Sequence[SamplingVariant]] = 3,
    batch: int = 12,
    workers: int = 8,
    device: str = "cuda",
    study_ids: Optional[Sequence[str]] = None,
    variant_combine: str = "rank",
    variant_weights: Optional[Sequence[float]] = None,
    seed: int = 42,
    index=None,
    return_frames: bool = False,
    verbose: bool = True,
):
    """The 224 base arm (`infer.predict_studies`, i.e. the exact v7 combiner), TTA'd.

    Use this rather than `predict_studies_tta` for the base arm so the per-variant frame is
    produced by the literal submitted code path (`members` are `infer.Member`).
    """
    vs = resolve_variants(variants, seed=seed)
    frames: Dict[int, pd.DataFrame] = {}
    for v in vs:
        if verbose:
            print(f"[tta] variant {v.label}")
        with _tta_bind(predict_studies, "StudyBagDataset", StudyBagDataset, v,
                       index_name="build_study_index", index=index, device=device):
            frames[v.idx] = predict_studies(
                input_root, split, members, batch=batch, workers=workers, device=device,
                study_ids=study_ids)
    combined = _tta_combine(frames, vs, variant_combine, variant_weights, verbose)
    return (combined, frames) if return_frames else combined


def _tta_combine(frames: Dict[int, pd.DataFrame], vs: Sequence[SamplingVariant],
                 variant_combine: str, variant_weights: Optional[Sequence[float]],
                 verbose: bool) -> pd.DataFrame:
    ordered = [frames[v.idx] for v in vs]
    if len(ordered) == 1:
        # S=1 must be byte-identical to the non-TTA call: no rank transform, no reordering.
        return ordered[0]
    if verbose:
        counts = {v.name: len(f) for v, f in zip(vs, ordered)}
        print(f"[tta] combining {len(ordered)} variant frame(s) by {variant_combine}-average; "
              f"rows {counts}")
    return rank_average_variant_frames(ordered, weights=variant_weights, mode=variant_combine)
