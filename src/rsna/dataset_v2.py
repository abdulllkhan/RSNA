"""Wave-4 "v2" input contract — the plane x fluid SLOT sampler of the 0.92+ public solutions.

WHAT THIS ARM CHANGES (vs every arm we already have)
----------------------------------------------------
Our arms build a study bag by splitting K slots evenly across the *planes present*
(`dataset.sample_study_slices`: 8/8/8 at K=24) and then evenly across each plane's series.
The 0.92+ public solutions do something structurally different — a FIXED slot table over
(plane x Fluid_Sensitive), 64 slices, one chosen series per slot:

    SLOTS = [("Sagittal",1,18), ("Sagittal",0,14), ("Coronal",1,12), ("Coronal",0,8),
             ("Axial",None,12)]                       # 64 slices; the Raptor arm, LB 0.924
    CROP_MM = 140.0 ; SPAN = (0.02, 0.98) ; per-SERIES percentile normalisation

Three consequences (`reports/public_solutions.md` §2c, §4 #2):
  1. **64 slices, not 24.**
  2. **A second series per plane, chosen by fluid-sensitivity** — fluid-sensitive sequences
     show effusion/synovitis/contusion, non-fluid ones show anatomy and cartilage, and the
     twelve findings are split across both. Our sampler mixes them into one plane budget.
  3. **A wide span (0.02-0.98), not a central band** — the collateral ligaments and the
     lateral meniscus live in the outer slices.

Plus (§2c, §4 #3) **laterality canonicalisation to a right knee**: horizontal flip of the
Coronal and Axial slices and a reversal of the Sagittal slice ORDER for a left knee, and
**never a flip augmentation** (a flip would reintroduce the nuisance axis the
canonicalisation removes, and would corrupt Medial-vs-Lateral).

WHAT IS REUSED, NOT RE-IMPLEMENTED
----------------------------------
* `dataset_hr.build_hr_index` -> the `{study: [(series, plane, [paths...])]}` manifest, read
  from the cached pickle (this module NEVER scans a tree itself);
* `dataset_hr.read_slice_hr` -> the size-parametric per-slice reader (JPEG cache or `.dcm`),
  which already does the INTER_AREA down-resize when `--size` < the cache's own edge;
* `order.SliceOrderResolver` -> anatomical rank per series (parquet / DICOM headers /
  mirror tree / lex), exactly as `dataset_ordered` uses it;
* `dataset.StudyBagDataset._augment` -> the small affine + intensity policy, NO flips.

WHAT IS NEW
-----------
* the slot table + its series-choice and fallback rules (`select_slot_series`);
* evenly-spaced positions in ANATOMICAL rank over span [0.02, 0.98] (`plan_slots`);
* laterality canonicalisation driven by `data/study_side.parquet` (train) or
  `laterality.side_from_series_meta` (DICOM inference);
* per-cell label weights (`data/labels/consensus_v1.parquet`), with a 1.0-weight stub built
  from `llm_labels_v4_blend.csv` until that file exists.

THE PLANNER IS SHARED WITH INFERENCE. `select_slot_series` + `plan_slots` are pure functions
over `(plane, fluid, n_slices)` records; `infer_v2.py` calls the SAME two functions on the
DICOM mount, so the slot choice, the positions and the canonicalisation cannot drift between
train and test. Only the materialisation differs (JPEG path vs decoded volume).

TENSOR CONTRACT
---------------
    x        float32 [K, 3, S, S]   grayscale replicated x3, (x - mean) / std
    slot_id  int64   [K]            0..4, index into SLOTS
    pos      float32 [K]            within-slot normalised position (j+0.5)/n, canonicalised
    valid    bool    [K]            0 only where the study has no series for that plane
    y        float32 [12]           soft targets
    w        float32 [12]           per-cell loss weights
    sid      str                    StudyInstanceUID

Use:
    from src.rsna.dataset_v2 import build_index_v2, StudyBagDatasetV2, load_labels_v2
    idx = build_index_v2("data/cache_c140_384", "train", order=resolver)
    lab = load_labels_v2("data/labels/consensus_v1.parquet")
    ds  = StudyBagDatasetV2(ids, idx, lab, size=384, train=True, k_scale=0.75,
                            side_map=load_side_map("data/study_side.parquet"))
"""
from __future__ import annotations

import math
import os
import pickle
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import TARGETS, ID_COL  # noqa: F401  (re-exported for callers)
from .dataset import IMAGENET_MEAN, IMAGENET_STD, PLANE_ORDER, StudyBagDataset
from .dataset_hr import build_hr_index, read_slice_hr, resolve_series_csv
from .order import SliceOrderResolver, stem_of  # noqa: F401

cv2.setNumThreads(0)  # avoid oversubscription inside DataLoader workers

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODE_TAG_V2 = "v2"  # written into the ckpt args by train_v2; read by infer_v2.is_v2_meta

# --------------------------------------------------------------------------------------
# The slot table (Raptor arm, LB 0.924 — reports/public_solutions.md §2c)
# --------------------------------------------------------------------------------------
#   (Anatomical_Plane, Fluid_Sensitive, n_slices)   —  None = "any fluid class"
SLOTS: Tuple[Tuple[str, Optional[int], int], ...] = (
    ("Sagittal", 1, 18),
    ("Sagittal", 0, 14),
    ("Coronal", 1, 12),
    ("Coronal", 0, 8),
    ("Axial", None, 12),
)
SLOT_NAMES: Tuple[str, ...] = ("Sag-fluid", "Sag-nonfluid", "Cor-fluid", "Cor-nonfluid", "Axial-any")
N_SLOTS = len(SLOTS)
K_FULL = sum(s[2] for s in SLOTS)          # 64
SPAN_LO, SPAN_HI = 0.02, 0.98              # the public arms' wide span
V2_JITTER = 1.0                       # train-time +/- 0.5 slot spacing
V2_CROP_MM = 140.0                    # recorded in the ckpt; applied by the cache/decoder
V2_PCT = (1.0, 99.0)                  # per-series percentile normalisation, ditto
CANONICAL_SIDE = "R"                         # canonical knee

# Per-process counters for the two documented degradations. Not fatal, never silent.
FALLBACKS: Dict[str, int] = {"fluid_swap": 0, "missing_plane": 0, "padded": 0}
_WARNED_V2: Dict[str, bool] = {}


def _warn_once_v2(key: str, msg: str) -> None:
    if not _WARNED_V2.get(key):
        _WARNED_V2[key] = True
        print(f"[v2] {msg}", flush=True)


def fallback_report() -> str:
    """Slot fallbacks seen IN THIS PROCESS (DataLoader workers count their own)."""
    return (f"[v2] slot fallbacks: fluid_swap={FALLBACKS['fluid_swap']} "
            f"missing_plane={FALLBACKS['missing_plane']} padded_slots={FALLBACKS['padded']}")


def scaled_slot_counts(k_scale: float = 1.0) -> Tuple[int, ...]:
    """Per-slot slice counts scaled by `k_scale` (round-half-UP, floor 1).

    `k_scale=0.75` -> (14, 11, 9, 6, 9) = 49, the memory-fitting variant quoted in the spec.
    Eval always uses `k_scale=1.0` -> (18, 14, 12, 8, 12) = 64.
    """
    s = float(k_scale)
    return tuple(max(1, int(math.floor(n * s + 0.5))) for _p, _f, n in SLOTS)


def slot_total(k_scale: float = 1.0) -> int:
    return int(sum(scaled_slot_counts(k_scale)))


def slot_id_vector(k_scale: float = 1.0) -> np.ndarray:
    """`[K]` int64: which slot each position of the bag belongs to (fixed layout)."""
    counts = scaled_slot_counts(k_scale)
    return np.concatenate([np.full(n, i, dtype=np.int64) for i, n in enumerate(counts)])


# --------------------------------------------------------------------------------------
# Laterality
# --------------------------------------------------------------------------------------
_OPS_CACHE: Dict[str, dict] = {}


def _default_canonical_ops(side: str) -> dict:
    """Local stand-in for `laterality.canonical_ops` until that module exists."""
    if str(side).upper().startswith("L"):
        return {"flip_lr_planes": ("Coronal", "Axial"), "reverse_sagittal": True}
    return {"flip_lr_planes": (), "reverse_sagittal": False}


def canonical_ops_for(side: str) -> dict:
    """`laterality.canonical_ops(side)` if importable, else the identical local default.

    Imported LAZILY and cached per side: training reads a JPEG cache and must not need
    `laterality.py` (or pydicom) to exist at all.
    """
    key = str(side).upper()[:1] or CANONICAL_SIDE
    hit = _OPS_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        from .laterality import canonical_ops  # type: ignore
        ops = dict(canonical_ops(key))
    except Exception:
        _warn_once_v2("no_laterality",
                   "src/rsna/laterality.py not importable -> using the built-in canonical_ops "
                   "(L: flip Coronal+Axial, reverse Sagittal order)")
        ops = _default_canonical_ops(key)
    _OPS_CACHE[key] = ops
    return ops


def load_side_map(parquet: Optional[str]) -> Dict[str, str]:
    """`{StudyInstanceUID: 'L'|'R'}` from `data/study_side.parquet`; `{}` (=> all R) if absent."""
    if not parquet or not os.path.exists(parquet):
        _warn_once_v2("no_side_parquet",
                   f"study_side parquet {parquet!r} not found -> treating EVERY study as "
                   f"'{CANONICAL_SIDE}' (no canonicalisation). Laterality-specific findings "
                   f"(Medial/Lateral Meniscus, Medial/Lateral OA) will be learned as noise.")
        return {}
    import pandas as pd

    df = pd.read_parquet(parquet, columns=[ID_COL, "side"])
    out = {str(s): str(v).upper()[:1] for s, v in zip(df[ID_COL], df["side"])}
    n_l = sum(1 for v in out.values() if v == "L")
    print(f"[v2] side map: {len(out)} studies from {parquet} ({n_l} L / {len(out) - n_l} R)")
    return out


# --------------------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------------------
class SeriesRecV2(NamedTuple):
    """One series, with everything the slot planner needs."""
    series: str
    plane: str
    fluid: Optional[int]        # Fluid_Sensitive (0/1); None if the CSV had no row
    n: int                      # slices
    paths: Tuple[str, ...] = ()  # ANATOMICALLY ordered when the index was built with `order`


IndexV2 = Dict[str, List[SeriesRecV2]]


def default_index_cache_v2(cache_root: str, split: str = "train") -> str:
    return os.path.join(cache_root, f"study_index_v2_{split}.pkl")


def build_index_v2(
    cache_root: str,
    split: str = "train",
    series_csv: Optional[str] = None,
    order: Optional[SliceOrderResolver] = None,
    cache_path: Optional[str] = None,
    hr_index_cache: Optional[str] = None,
    only: Optional[Sequence[str]] = None,
) -> IndexV2:
    """`{study: [SeriesRecV2, ...]}` — the HR study index joined with the series CSV.

    Reuses `dataset_hr.build_hr_index`, which reads `<cache_root>/study_index_<split>.pkl`
    when it exists, so a full cache is NEVER re-scanned here. The join adds
    `Fluid_Sensitive` (the second slot axis) and, when an `order` resolver is passed, ORDERS
    each series' path list anatomically so a slot position is a rank, not a hash position.

    `Fat_Suppression` is byte-identical to `Fluid_Sensitive` in both CSVs (verified,
    `reports/public_solutions.md` header) and is deliberately ignored.

    Cached to `<cache_root>/study_index_v2_<split>.pkl` (override with `cache_path`), with
    the `ordered` flag stored alongside so a lex-ordered pickle is never mistaken for an
    anatomically ordered one. `only=[...]` restricts the build to those studies (smokes);
    such a build is never cached. NOTHING is cached — and `<cache_root>/study_index_<split>.pkl`
    is never written — while that manifest is absent, because its absence means the cache is
    still being built and any index taken from it would be partial.
    """
    want_ordered = order is not None
    keep = None if only is None else {str(s) for s in only}
    # A filtered build is a smoke/debug artefact and is never cached (it would be mistaken
    # for the full index on the next run).
    cp = None if keep is not None else (cache_path or default_index_cache_v2(cache_root, split))
    if cp and os.path.exists(cp):
        try:
            with open(cp, "rb") as f:
                blob = pickle.load(f)
            if isinstance(blob, dict) and blob.get("ordered") == want_ordered:
                idx = blob["index"]
                print(f"[v2] index: {len(idx)} studies from {cp} (ordered={want_ordered})")
                return idx
            print(f"[v2] index cache {cp} has ordered={blob.get('ordered')!r}, want "
                  f"{want_ordered} -> rebuilding")
        except Exception as e:
            print(f"[v2] index cache {cp} unreadable ({type(e).__name__}: {e}) -> rebuilding")

    import pandas as pd

    if series_csv is None:
        series_csv = resolve_series_csv(cache_root, split)
    # NEVER write `<cache_root>/study_index_<split>.pkl` ourselves. That pickle is the
    # cache's own manifest and is written by the cache builder when the cache is COMPLETE;
    # writing it from here while a build is still running would freeze a PARTIAL manifest
    # that every later run would then trust. Use it if it is there, scan in memory if not.
    hr_pkl = hr_index_cache or os.path.join(cache_root, f"study_index_{split}.pkl")
    complete = os.path.exists(hr_pkl)
    if not complete:
        _warn_once_v2("no_hr_index",
                   f"{hr_pkl} does not exist -> scanning {cache_root}/{split}_series in memory. "
                   f"If the cache is still being built this index is PARTIAL, so neither it "
                   f"nor the v2 index will be cached to disk.")
    hr = build_hr_index(cache_root, split=split, series_csv=series_csv,
                        cache_path=hr_pkl if complete else None)
    ser = pd.read_csv(series_csv)
    fluid_of: Dict[str, Optional[int]] = {}
    plane_of: Dict[str, str] = {}
    for s, f, p in zip(ser["SeriesInstanceUID"].astype(str),
                       ser["Fluid_Sensitive"], ser["Anatomical_Plane"].astype(str)):
        try:
            fluid_of[s] = int(f)
        except Exception:
            fluid_of[s] = None
        plane_of[s] = p if p in PLANE_ORDER else "Sagittal"

    index: IndexV2 = {}
    n_series = n_missing_fluid = 0
    for study, entry in hr.items():
        if keep is not None and str(study) not in keep:
            continue
        recs: List[SeriesRecV2] = []
        for series, plane, paths in entry:
            series = str(series)
            fl = fluid_of.get(series)
            if fl is None:
                n_missing_fluid += 1
            pl = plane_of.get(series, plane)
            ppaths = list(paths)
            if order is not None and ppaths:
                try:
                    ranks = np.asarray(order.ranks(series, ppaths))
                    if len(ranks) == len(ppaths):
                        anat = np.argsort(ranks, kind="stable")
                        ppaths = [ppaths[int(i)] for i in anat]
                except Exception as e:   # a bad header must never kill an index build
                    print(f"[v2] rank lookup failed for {series[-12:]}: "
                          f"{type(e).__name__}: {e} -> lexicographic for it")
            recs.append(SeriesRecV2(series, pl, fl, len(ppaths), tuple(ppaths)))
            n_series += 1
        if recs:
            index[str(study)] = recs

    print(f"[v2] index built: {len(index)} studies, {n_series} series "
          f"(ordered={want_ordered}, series with no CSV fluid flag={n_missing_fluid})")
    if order is not None:
        print(" ", order.report())
    if cp and not complete:
        print(f"[v2] NOT caching the v2 index to {cp}: the cache has no "
              f"study_index_{split}.pkl, so this build may be partial")
        cp = None
    if cp:
        os.makedirs(os.path.dirname(os.path.abspath(cp)), exist_ok=True)
        with open(cp, "wb") as f:
            pickle.dump({"ordered": want_ordered, "index": index}, f)
        print(f"[v2] index cached -> {cp}")
    return index


# --------------------------------------------------------------------------------------
# The planner — SHARED between training (JPEG cache) and inference (DICOM mount)
# --------------------------------------------------------------------------------------
class SlotSel(NamedTuple):
    """Which series serves one slot, and how many slices it owes."""
    slot: int
    plane: str
    fluid: Optional[int]
    n_want: int
    rec: Optional[SeriesRecV2]     # None -> the plane is absent from this study
    fallback: str                  # "" | "fluid_swap" | "missing_plane"


class SlotPick(NamedTuple):
    """One materialised bag position."""
    slot: int
    series: Optional[str]
    idx: int          # index into the series' ANATOMICALLY ORDERED slice list (-1 = invalid)
    pos: float        # within-slot normalised position in [0,1], AFTER canonicalisation
    flip: bool        # horizontal flip (left knee, Coronal/Axial)
    valid: bool


def select_slot_series(
    recs: Sequence[SeriesRecV2],
    k_scale: float = 1.0,
    train: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> List[SlotSel]:
    """Choose one series per slot.

    Candidates = every series of that plane with that `Fluid_Sensitive`.
      * eval  -> the candidate with the MOST slices (ties broken by series UID: deterministic);
      * train -> a uniform random candidate (a free, label-preserving augmentation, and the
                 reason a study with two sagittal fluid series is not always read the same way).
    Fallbacks, both counted in `FALLBACKS` and never silent:
      * no series of the required fluid class -> the other class of the SAME plane
        ("fluid_swap"): a wrong-contrast slice of the right anatomy beats a zero tensor;
      * the plane itself is absent -> `rec=None`, the slot is filled with zeros and `valid=0`
        so the attention pool can never see it ("missing_plane").
    A series whose CSV row is missing (`fluid=None`) can only ever serve as a fluid_swap
    candidate, never as an exact match.
    """
    counts = scaled_slot_counts(k_scale)
    out: List[SlotSel] = []
    for i, ((plane, fluid, _n), n_want) in enumerate(zip(SLOTS, counts)):
        same_plane = [r for r in recs if r.plane == plane and r.n > 0]
        if fluid is None:
            cands, fb = same_plane, ""
        else:
            cands = [r for r in same_plane if r.fluid is not None and int(r.fluid) == int(fluid)]
            fb = ""
            if not cands and same_plane:
                cands, fb = same_plane, "fluid_swap"
                FALLBACKS["fluid_swap"] += 1
        if not cands:
            FALLBACKS["missing_plane"] += 1
            out.append(SlotSel(i, plane, fluid, n_want, None, "missing_plane"))
            continue
        if train and rng is not None and len(cands) > 1:
            rec = cands[int(rng.integers(len(cands)))]
        else:
            rec = max(cands, key=lambda r: (r.n, r.series))
        out.append(SlotSel(i, plane, fluid, n_want, rec, fb))
    return out


def _slot_fracs(n_want: int, train: bool, rng: Optional[np.random.Generator],
                jitter: float) -> List[float]:
    """`n_want` evenly-spaced fractions over the span [0.02, 0.98], jittered on train.

    `jitter=1.0` moves each position by up to +/- HALF the slot spacing — the same
    convention as `dataset._pick_positions` (`frac += (rng.random()-0.5)*jitter/n`), mapped
    onto the span instead of the whole [0,1) range.
    """
    span = SPAN_HI - SPAN_LO
    step = span / max(1, n_want)
    fracs: List[float] = []
    for j in range(n_want):
        f = SPAN_LO + span * (j + 0.5) / n_want
        if train and rng is not None and jitter:
            f += (rng.random() - 0.5) * float(jitter) * step
        fracs.append(min(SPAN_HI, max(0.0, f)))
    return fracs


def plan_slots(
    sels: Sequence[SlotSel],
    side: str = CANONICAL_SIDE,
    train: bool = False,
    rng: Optional[np.random.Generator] = None,
    jitter: float = V2_JITTER,
) -> List[SlotPick]:
    """Positions inside each chosen series + the laterality canonicalisation.

    Positions are evenly spaced in ANATOMICAL RANK over [0.02, 0.98] of the stack — the
    public arms' wide span, because the collaterals and the lateral meniscus sit in the
    outer slices. `idx = floor(frac * n)`, so a series with FEWER slices than the slot needs
    simply repeats: n=3 into a 12-slice slot yields 0,0,0,0,1,1,1,1,2,2,2,2 — all of the
    stack, padded by repeats, and still `valid=1` (real pixels, just over-sampled).

    Laterality (left knee -> canonical right knee):
      * Coronal and Axial slices get a horizontal flip (`flip=True`);
      * Sagittal slots get their emitted slice ORDER reversed, so bag position j always
        means the same anatomical station regardless of side. `pos` is assigned AFTER the
        reversal, so the position embedding is canonical too.
    No flip is ever applied as an augmentation (see module docstring).
    """
    ops = canonical_ops_for(side)
    flip_planes = tuple(ops.get("flip_lr_planes", ()) or ())
    rev_sag = bool(ops.get("reverse_sagittal", False))

    picks: List[SlotPick] = []
    for sel in sels:
        if sel.rec is None or sel.rec.n <= 0:
            picks.extend(SlotPick(sel.slot, None, -1, (j + 0.5) / max(1, sel.n_want), False, False)
                         for j in range(sel.n_want))
            continue
        n = int(sel.rec.n)
        idxs = [min(n - 1, max(0, int(math.floor(f * n))))
                for f in _slot_fracs(sel.n_want, train, rng, jitter)]
        if n < sel.n_want:
            FALLBACKS["padded"] += 1
        if rev_sag and sel.plane == "Sagittal":
            idxs = idxs[::-1]
        flip = sel.plane in flip_planes
        for j, ix in enumerate(idxs):
            picks.append(SlotPick(sel.slot, sel.rec.series, int(ix),
                                  (j + 0.5) / max(1, sel.n_want), flip, True))
    return picks


def plan_study(
    recs: Sequence[SeriesRecV2],
    side: str = CANONICAL_SIDE,
    k_scale: float = 1.0,
    train: bool = False,
    rng: Optional[np.random.Generator] = None,
    jitter: float = V2_JITTER,
) -> Tuple[List[SlotSel], List[SlotPick]]:
    """`select_slot_series` then `plan_slots` — the whole plan for one study."""
    sels = select_slot_series(recs, k_scale=k_scale, train=train, rng=rng)
    return sels, plan_slots(sels, side=side, train=train, rng=rng, jitter=jitter)


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------
W_COLS = [f"{t}__w" for t in TARGETS]


def default_labels_paths() -> Dict[str, str]:
    return dict(
        consensus=os.path.join(ROOT, "data", "labels", "consensus_v1.parquet"),
        steven=os.path.join(ROOT, "data", "labels",
                            "stevenleehans_rsna-knee-llm-report-labels",
                            "llm_labels_v4_blend.csv"),
        folds=os.path.join(ROOT, "data", "folds.csv"),
        train_csv=os.path.join(ROOT, "data", "comp_csv", "train.csv"),
    )


def load_labels_v2(
    path: Optional[str] = None,
    steven_csv: Optional[str] = None,
    folds_csv: Optional[str] = None,
    train_csv: Optional[str] = None,
):
    """The v2 label table: index `StudyInstanceUID`, cols `<T>`, `<T>__w`, `is_gold`, `fold`.

    `path` is `data/labels/consensus_v1.parquet` (the three-teacher soft consensus with a
    per-cell confidence weight, gold rows weight 3.0). Until that file exists this builds an
    EQUIVALENT-SHAPED stub from `llm_labels_v4_blend.csv` with **all weights 1.0**, `is_gold`
    from `train.csv` (rows with all 12 labels non-null) and `fold` from `data/folds.csv`, and
    says so loudly — the stub is a plumbing placeholder, not the recipe.
    """
    import pandas as pd

    P = default_labels_paths()
    if path and os.path.exists(path):
        df = pd.read_parquet(path)
        if ID_COL in df.columns:
            df = df.set_index(ID_COL)
        df.index = df.index.astype(str)
        missing = [c for c in TARGETS if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing target columns {missing}")
        for t in TARGETS:                                   # tolerate a weightless table
            if f"{t}__w" not in df.columns:
                df[f"{t}__w"] = 1.0
        if "is_gold" not in df.columns:
            df["is_gold"] = False
        if "fold" not in df.columns:
            raise ValueError(f"{path}: no `fold` column — v2 folds come from the label file")
        n_gold = int(df["is_gold"].astype(bool).sum())
        print(f"[v2] labels: {len(df)} studies from {path} "
              f"(gold={n_gold}, folds={sorted(df['fold'].unique().tolist())}, "
              f"mean weight={float(df[W_COLS].values.mean()):.3f})")
        return df[TARGETS + W_COLS + ["is_gold", "fold"]]

    src = steven_csv or P["steven"]
    fld = folds_csv or P["folds"]
    trn = train_csv or P["train_csv"]
    print(f"[v2] labels: consensus parquet {path!r} not found -> STUB from {os.path.basename(src)} "
          f"with all per-cell weights = 1.0 (folds from {os.path.basename(fld)})")
    df = pd.read_csv(src).set_index(ID_COL)
    df.index = df.index.astype(str)
    df = df[TARGETS].astype(np.float32)
    for t in TARGETS:
        df[f"{t}__w"] = np.float32(1.0)
    tr = pd.read_csv(trn)
    gold_ids = set(tr.loc[tr[TARGETS].notna().all(axis=1), ID_COL].astype(str))
    df["is_gold"] = [s in gold_ids for s in df.index]
    folds = pd.read_csv(fld).set_index(ID_COL)["fold"]
    folds.index = folds.index.astype(str)
    df["fold"] = folds.reindex(df.index).fillna(-1).astype(int)
    n_gold = int(df["is_gold"].sum())
    print(f"[v2] labels(stub): {len(df)} studies, gold={n_gold}, "
          f"folds={sorted(df['fold'].unique().tolist())}")
    return df[TARGETS + W_COLS + ["is_gold", "fold"]]


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class StudyBagDatasetV2(Dataset):
    """Study -> `(x [K,3,S,S], slot_id [K], pos [K], valid [K], y [12], w [12], sid)`.

    `size` may be SMALLER than the cache's own edge (e.g. 336 off the 384 cache): the
    per-slice reader (`dataset_hr.read_slice_hr`) does the INTER_AREA down-resize, the same
    interpolation the cache itself was written with. It is never up-sized in practice.

    Augmentation is `dataset.StudyBagDataset._augment` — small affine + intensity, **no
    flips** — called through the parent's unbound method so the policy has exactly one
    definition in the repo.
    """

    def __init__(
        self,
        study_ids: Sequence[str],
        index: IndexV2,
        labels=None,
        size: int = 384,
        train: bool = False,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        seed: int = 42,
        k_scale: float = 1.0,
        side_map: Optional[Dict[str, str]] = None,
        jitter: float = V2_JITTER,
        augment: bool = True,
    ):
        self.study_ids = [str(s) for s in study_ids if str(s) in index]
        self.index = index
        self.size = int(size)
        self.train = bool(train)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.seed = int(seed)
        self.k_scale = float(k_scale)
        self.k = slot_total(self.k_scale)
        self.jitter = float(jitter)
        self.augment = bool(augment)
        self.side_map = dict(side_map or {})
        self._slot_id = slot_id_vector(self.k_scale)

        self.y_map: Dict[str, np.ndarray] = {}
        self.w_map: Dict[str, np.ndarray] = {}
        if labels is not None:
            lab = labels
            for sid in self.study_ids:
                if sid in lab.index:
                    row = lab.loc[sid]
                    self.y_map[sid] = row[TARGETS].values.astype(np.float32)
                    self.w_map[sid] = row[W_COLS].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.study_ids)

    def side_of(self, sid: str) -> str:
        return self.side_map.get(sid, CANONICAL_SIDE)

    # ---- inspection (used by the smokes / the report) --------------------------------
    def plan(self, i: int, train: Optional[bool] = None):
        """`(sels, picks, side)` for item `i` — the deterministic eval plan by default."""
        sid = self.study_ids[i]
        tr = self.train if train is None else bool(train)
        rng = np.random.default_rng(self.seed + i)
        sels, picks = plan_study(self.index[sid], side=self.side_of(sid), k_scale=self.k_scale,
                                 train=tr, rng=rng, jitter=self.jitter)
        return sels, picks, self.side_of(sid)

    def _blank(self) -> np.ndarray:
        return np.zeros((3, self.size, self.size), dtype=np.float32)

    def __getitem__(self, i: int):
        sid = self.study_ids[i]
        # Same RNG discipline as every other arm (`dataset.StudyBagDataset.__getitem__`):
        # deterministic on eval, torch-seeded on train so each epoch resamples.
        if self.train:
            rng = np.random.default_rng(self.seed + i + int(torch.randint(0, 1 << 30, (1,)).item()))
        else:
            rng = np.random.default_rng(self.seed + i)
        recs = self.index[sid]
        by_uid = {r.series: r for r in recs}
        _sels, picks = plan_study(recs, side=self.side_of(sid), k_scale=self.k_scale,
                                  train=self.train, rng=rng, jitter=self.jitter)

        size = self.size
        bag = np.zeros((self.k, 3, size, size), dtype=np.float32)
        valid = np.zeros(self.k, dtype=bool)
        pos = np.zeros(self.k, dtype=np.float32)
        for j, p in enumerate(picks):
            pos[j] = p.pos
            if not p.valid or p.series is None:
                continue                                   # zeros + valid=0 (masked out)
            rec = by_uid.get(p.series)
            if rec is None or not rec.paths:
                continue
            img = read_slice_hr(rec.paths[min(p.idx, len(rec.paths) - 1)], size)
            if img is None:                                # unreadable slice -> masked out
                continue
            if p.flip:
                img = np.ascontiguousarray(img[:, ::-1])   # laterality canonicalisation
            if self.train and self.augment:
                img = StudyBagDataset._augment(self, img, rng)   # affine+intensity, NO flips
            chw = np.stack([img, img, img], axis=0)
            bag[j] = (chw - self.mean) / self.std
            valid[j] = True

        y = self.y_map.get(sid, np.zeros(len(TARGETS), dtype=np.float32))
        w = self.w_map.get(sid, np.ones(len(TARGETS), dtype=np.float32))
        return (torch.from_numpy(bag), torch.from_numpy(self._slot_id.copy()),
                torch.from_numpy(pos), torch.from_numpy(valid),
                torch.from_numpy(np.asarray(y, dtype=np.float32)),
                torch.from_numpy(np.asarray(w, dtype=np.float32)), sid)


__all__ = [
    "SLOTS", "SLOT_NAMES", "N_SLOTS", "K_FULL", "SPAN_LO", "SPAN_HI", "MODE_TAG_V2",
    "V2_CROP_MM", "V2_PCT", "V2_JITTER", "CANONICAL_SIDE",
    "SeriesRecV2", "SlotSel", "SlotPick", "StudyBagDatasetV2",
    "build_index_v2", "default_index_cache_v2", "load_labels_v2", "load_side_map",
    "canonical_ops_for", "select_slot_series", "plan_slots", "plan_study",
    "scaled_slot_counts", "slot_total", "slot_id_vector", "fallback_report", "FALLBACKS",
    "W_COLS", "IMAGENET_MEAN", "IMAGENET_STD",
]
