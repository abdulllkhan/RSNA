"""Inference for the Wave-4 "v2" arm — slot sampler + laterality, on DICOM or on a cache.

WHY THIS ARM NEEDS ITS OWN PREDICTOR
------------------------------------
A v2 checkpoint was trained on a bag that is not a bag of "K slices from the study": it is
5 fixed plane x fluid SLOTS, one chosen series each, evenly spaced in anatomical rank over
span 0.02-0.98, canonicalised to a right knee, with `slot_id` / `pos` / `valid` side inputs
the model actually consumes. Feeding it `infer_hr`'s lexicographic 24-slice bag would produce
plausible numbers from an input distribution it has never seen — the same silent failure
`reports/arm_25d.md` §6 documents for the 2.5D arm. Hence:

    ROUTING PREDICATE: `is_v2_meta(args)` must be consulted FIRST in the notebook's
    `detect_arm`, BEFORE the `size != 224 -> hr` rule (a v2 checkpoint is 384 too).

THE SLOT PLAN IS NOT RE-IMPLEMENTED HERE. `dataset_v2.select_slot_series` and
`dataset_v2.plan_slots` are the same two functions the training dataset calls, so the slot
choice, the positions and the canonicalisation cannot drift between train and test. Only the
*materialisation* differs: training reads cached JPEGs, this reads a decoded volume.

THE DICOM PATH, PER STUDY (the ordering of these steps is load-bearing)
-----------------------------------------------------------------------
 1. rows for the study from `<root>/<split>_series.csv` -> (plane, Fluid_Sensitive) per series;
    `n_slices` from the series directory listing (no pixels read);
 2. `select_slot_series` -> at most 5 series are actually needed, out of ~5.5 per study;
 3. each chosen series is decoded ONCE, IN FULL, with `decode.read_series_v2` — in full
    because the normalisation is a per-SERIES percentile (§2c), which a per-slice decode
    cannot reproduce; once because repeated decodes are what eats the public kernels' 8 h
    (`reports/public_solutions.md` §5, gotcha iii);
 4. slice order from the DICOM headers (`order.series_ranks_from_dicom`) — never the
    filename, which is a randomised InstanceUID (lex == anatomical in 0/400 series);
 5. side from `laterality.side_from_series_meta` over the metas of everything decoded;
 6. `plan_slots` -> the same picks the trainer would have made;
 7. one uint8 [K,D,D] canonical stack per study (D = the group's `decode_size`) -> resized
    per member `size` -> normalised per member (mean,std) -> forwarded through every member
    of that size group.

DECODE GROUPS (2026-09-20). Steps 3-5 depend on the RENDERING contract `(crop_mm, pct)`, so
members are partitioned by that pair: one decode per (crop_mm, pct) per study, at
`decode_size = max(CACHE_SIZE, max member size)` of the group. Two 110 mm members and five
140 mm members therefore cost two decodes, not one silently-wrong decode — before this the
predictor decoded everything with member 0's crop and only printed a WARNING, which is how
`v2_effv2s384_c110_*` (whose `args['crop_mm']` said 140) got fed 140 mm crops on Kaggle.

Members of different `size` inside one decode group share that group's decode and differ only
by an INTER_AREA resize, which is exactly what `dataset_v2` does when trained at `--size 336`
off the 384 cache. The floor at `CACHE_SIZE` is deliberate: a sub-384 member was trained on a
384 cache, so it must keep seeing 384 -> its size, not a direct decode at its own size.

SLOT-PHASE TTA (`tta=S`, reports/tta_v2.md)
-------------------------------------------
`predict_studies_v2(..., tta=S)` evaluates the same ensemble over S deterministic slot-phase
variants (`dataset_v2.variant_phases(S)`: 0, +1/3, -1/3, ... slot spacings) and per-column
rank-averages the S finished frames with the SAME combiner the other arms use
(`infer_tta.rank_average_variant_frames`, which `dev/tta_eval.py --check-combiner` pins to
`infer_plane.rank_average_frames`). Variant 0 is the baseline phase and `S=1` returns that
frame untransformed, so `tta=1` is byte-identical to the pre-TTA predictor.

    ONE DECODE, S BAGS. A phase changes only WHICH slice indices of an already-decoded series
    are stacked (`dataset_v2.plan_slots` docstring), so step 3 above runs once per study and
    steps 6-7 run S times over the same `vols`. Kaggle cost is therefore S x the v2 FORWARD
    plus S x a memcpy-cheap assemble — the decode, which is the expensive half on a CPU-bound
    kernel, is unchanged. `[time] stage=v2_decode` stays flat while `stage=v2_forward` scales
    with S; that is the check `dev/smoke_v2_tta.sh` makes.

    (On the CACHE path there is no per-study decode — `StudyBagDatasetV2` reads one cached
    JPEG per picked slice — so there the S variants simply re-read the picked slices, exactly
    as `infer_tta` does for the base/hr arms.)

Reused verbatim from the existing arms: `infer_hr._forward_with_guard` (fp32 recompute on a
dead half-precision engine or a non-finite output), `infer.write_benchmark_submission` /
`write_submission` / `find_data_root`, the device probe, and the advisory fingerprint check.

Local checks:
    # DICOM mount
    .venv/bin/python -m src.rsna.infer_v2 --input-root /tmp/dicom_mount --split test \
        --ckpt models/v2_coat384_fold0.pt --device cpu --batch 1 --out /tmp/sub_v2.csv
    # cache path (OOF-style eval)
    .venv/bin/python -m src.rsna.infer_v2 --input-root data/cache_c140_384 --split train \
        --ckpt models/v2_coat384_fold0.pt --device cpu --batch 2
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import TARGETS, ID_COL
from .dataset_v2 import (
    V2_CROP_MM,
    V2_PCT,
    CANONICAL_SIDE,
    K_FULL,
    MODE_TAG_V2,
    SeriesRecV2,
    StudyBagDatasetV2,
    build_index_v2,
    context_ranks,
    fallback_report,
    load_side_map,
    parse_cache_contract,
    plan_slots,
    select_slot_series,
    variant_index_of,
    variant_label,
    variant_phases,
)
from .infer import (  # reused verbatim -- no duplication, no edit to infer.py
    FINGERPRINT_TOL,
    find_data_root,
    write_benchmark_submission,
    write_submission,
)
from .infer_hr import _forward_with_guard  # reused verbatim -- no edit to infer_hr.py
from .model_v2 import ModelV2Config, build_model_v2, compute_fingerprint_v2
from .order import series_ranks_from_dicom

cv2.setNumThreads(0)

CACHE_SIZE = 384          # the v2 cache / the resolution every DICOM decode produces

_T0 = time.time()


def _stage_v2(name: str, t0: float) -> None:
    """`[time] stage=... elapsed=...s`, the same line the multi-arm notebook prints."""
    print(f"[time] stage={name} elapsed={time.time() - t0:.1f}s", flush=True)


# --------------------------------------------------------------------------------------
# Arm detection — the predicate the notebook's `detect_arm` must consult FIRST
# --------------------------------------------------------------------------------------
def is_v2_meta(args: Optional[Mapping] = None, cfg: Optional[Mapping] = None,
               keys: Optional[Sequence[str]] = None) -> bool:
    """True iff this checkpoint is a v2 one — from its OWN metadata, never its filename.

    `train_v2.py` writes `args["mode"] = "v2"`. No other trainer in the repo writes that
    value (`train_25d` writes "2.5d", `train_ordered` writes "ordered"), and `args` survives
    `dev/slim_ckpt.py`, so this is stable across slimming.
    """
    return str(dict(args or {}).get("mode", "")).strip().lower() == MODE_TAG_V2


# --------------------------------------------------------------------------------------
# Members
# --------------------------------------------------------------------------------------
class MemberV2:
    """One loaded v2 checkpoint + the input contract it was trained with."""

    def __init__(self, model, mean, std, size: int, fingerprint, name: str = "",
                 k_eval: int = K_FULL, k_scale: float = 1.0,
                 slot_prior: float = 0.55, crop_mm: float = V2_CROP_MM,
                 pct: Sequence[float] = V2_PCT, backbone: str = "",
                 slice_context: int = 0):
        self.model = model
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.size = int(size)
        self.fingerprint = fingerprint
        self.name = name
        self.k_eval = int(k_eval)          # ALWAYS the full 64; k_scale was train-time only
        self.k_scale = float(k_scale)
        self.slot_prior = float(slot_prior)
        self.crop_mm = float(crop_mm)
        self.pct = tuple(float(v) for v in pct)
        self.backbone = backbone
        # 2.5D slot context (`train_v2 --slice-context`). 0 = replicated grayscale, 1 =
        # [prev, this, next]. A checkpoint trained before the flag existed has no such key, so
        # the default reproduces the original tensor exactly.
        self.slice_context = int(slice_context)

    @property
    def signature(self) -> Tuple:
        """Members sharing this share ONE normalised tensor (inside one decode group).

        `slice_context` is part of it: two members can share a decode and even a `size` and
        still need different CHANNELS, which is a different float tensor (and, on the cache
        path, a different `StudyBagDatasetV2`). `size` stays first — the group banners print
        `sig[0]`.
        """
        return (self.size,
                tuple(np.asarray(self.mean).ravel().round(6).tolist()),
                tuple(np.asarray(self.std).ravel().round(6).tolist()),
                self.slice_context)

    @property
    def decode_key(self) -> Tuple:
        """Members sharing this share ONE per-study DICOM decode: same crop, same percentiles.

        `size` is deliberately NOT part of it — a bigger sibling's decode can be resized down
        by `_stack_to_tensor`, but a different `crop_mm` or `pct` is a different image.
        `slice_context` is NOT part of it either: the neighbours of a slot slice are other
        slices of the SAME decoded volume, so a context member and a context-0 member share one
        decode and one assembled stack (built at the group's max context, `context_for`); the
        context-0 member then reads its centre plane, byte for byte what it saw before.
        """
        return (round(float(self.crop_mm), 6),
                tuple(round(float(p), 6) for p in self.pct))

    def __repr__(self) -> str:
        return f"MemberV2({self.name or 'ckpt'}, size={self.size}, K={self.k_eval})"


def load_members_v2(checkpoint_paths: Sequence[str], device: str = "cuda",
                    strict_mode: bool = False) -> List[MemberV2]:
    """Load v2 checkpoints with `pretrained=False`, reading the contract from each one."""
    members: List[MemberV2] = []
    for cp in checkpoint_paths:
        st = torch.load(cp, map_location=device, weights_only=False)
        a = st.get("args", {}) or {}
        if not is_v2_meta(a):
            msg = (f"{os.path.basename(cp)}: args['mode']={a.get('mode')!r} — this is NOT a v2 "
                   f"checkpoint; its slots/positions/laterality would be wrong here")
            if strict_mode:
                raise ValueError(msg)
            print(f"[member] WARNING {msg}; loading it anyway")
        cfg = ModelV2Config(**{k: v for k, v in (st.get("cfg", {}) or {}).items()
                               if k in ModelV2Config.__dataclass_fields__})
        cfg.pretrained = False       # NO network at inference
        cfg.grad_ckpt = False        # inference never checkpoints
        model = build_model_v2(cfg, verbose=False).to(device)
        if model._channels_last and str(device).startswith("cuda"):
            model = model.to(memory_format=torch.channels_last)
        model.load_state_dict(st["model"])
        model.eval()
        m = MemberV2(model, st["mean"], st["std"], int(a.get("size", CACHE_SIZE)),
                     np.asarray(st.get("fingerprint")), name=os.path.basename(cp),
                     k_eval=int(a.get("k_eval", K_FULL)), k_scale=float(a.get("k_scale", 1.0)),
                     slot_prior=float(a.get("slot_prior", cfg.slot_prior)),
                     crop_mm=float(a.get("crop_mm", V2_CROP_MM)),
                     pct=tuple(a.get("pct", V2_PCT)), backbone=cfg.backbone,
                     slice_context=int(a.get("slice_context", 0) or 0))
        print(f"[member] {m.name}: mode=v2 size={m.size} K_eval={m.k_eval} "
              f"(trained k_scale={m.k_scale}) crop_mm={m.crop_mm} pct={m.pct} "
              f"slot_prior={m.slot_prior} slice_context={m.slice_context} "
              f"backbone={m.backbone}")
        members.append(m)
    return members


def decode_groups_v2(members: Sequence[MemberV2]) -> Dict[Tuple, List[int]]:
    """`{(crop_mm, pct): [member index, ...]}` — one per-study decode per returned key."""
    out: Dict[Tuple, List[int]] = {}
    for i, m in enumerate(members):
        out.setdefault(m.decode_key, []).append(i)
    return out


def decode_size_for(members: Sequence[MemberV2], idxs: Sequence[int]) -> int:
    """Pixels to decode a group at: the largest member size, floored at `CACHE_SIZE`.

    Floored because a member smaller than the cache (e.g. `--size 336` off the 384 cache) was
    TRAINED on `CACHE_SIZE -> its size` INTER_AREA, and decoding straight to 336 would hand it
    an image its training distribution never contained. Above the floor the group's biggest
    member gets a native decode and its smaller siblings get the same INTER_AREA down-resize
    the cache path gives them.
    """
    return max(int(CACHE_SIZE), max(int(members[i].size) for i in idxs))


def context_for(members: Sequence[MemberV2], idxs: Sequence[int]) -> int:
    """Neighbour planes the group's SHARED stack must carry: the largest member context.

    A context-0 member reads the centre plane of that stack and gets exactly the pixels a
    1-plane stack would have handed it, so one assemble serves the whole decode group.
    """
    return max(int(members[i].slice_context) for i in idxs)


def check_fingerprint_v2(member: MemberV2, device: str, tol: float = FINGERPRINT_TOL) -> bool:
    """Advisory self-check (see `infer.py`: a cross-arch mismatch must NOT abort the run)."""
    if member.fingerprint is None or np.asarray(member.fingerprint).size == 0:
        print(f"[fingerprint] {member.name}: none stored; skipping check")
        return True
    got = compute_fingerprint_v2(member.model, device, member.k_eval, member.size)
    ref = np.asarray(member.fingerprint)
    if got.shape != ref.shape:
        print(f"[fingerprint] {member.name}: shape {got.shape} vs stored {ref.shape} -> SKIP")
        return True
    ok = bool(np.allclose(got, ref, atol=tol))
    print(f"[fingerprint] {member.name}: max|diff|={float(np.max(np.abs(got - ref))):.2e} "
          f"tol={tol:.0e} -> {'OK' if ok else 'MISMATCH'}")
    return ok


# --------------------------------------------------------------------------------------
# The DICOM path
# --------------------------------------------------------------------------------------
def has_read_series_v2() -> bool:
    """Is `decode.read_series_v2` (the 140 mm / per-series-percentile decoder) available yet?"""
    try:
        from .decode import read_series_v2  # noqa: F401
        return True
    except Exception:
        return False


def _side_from_metas(metas_per_series: Sequence[Sequence[dict]]) -> Tuple[str, str, float]:
    """`laterality.side_from_series_meta` if importable, else ('R', 'default', nan) + a warning."""
    try:
        from .laterality import side_from_series_meta  # type: ignore
        side, source, x_mm = side_from_series_meta(list(metas_per_series))
        return str(side).upper()[:1], str(source), float(x_mm) if x_mm is not None else float("nan")
    except Exception as e:
        print(f"[v2] laterality.side_from_series_meta unavailable ({type(e).__name__}) -> "
              f"assuming '{CANONICAL_SIDE}' for every study (no canonicalisation)")
        return CANONICAL_SIDE, "default", float("nan")


def group_series_rows(series_df: pd.DataFrame) -> Dict[str, List[Tuple[str, Optional[int], str]]]:
    """`{study: [(series, Fluid_Sensitive, Anatomical_Plane), ...]}` — grouped ONCE.

    (A per-study `series_df[series_df[ID] == study]` scan would be O(studies x rows); at
    1,300 hidden studies that is the kind of quiet quadratic that shows up as an unexplained
    hour in the Kaggle log.)
    """
    out: Dict[str, List[Tuple[str, Optional[int], str]]] = {}
    for study, series, fluid, plane in zip(series_df[ID_COL].astype(str),
                                           series_df["SeriesInstanceUID"].astype(str),
                                           series_df["Fluid_Sensitive"],
                                           series_df["Anatomical_Plane"].astype(str)):
        try:
            fl: Optional[int] = int(fluid)
        except Exception:
            fl = None
        out.setdefault(study, []).append((series, fl, plane))
    return out


def _study_series_records(root: str, split: str, rows_by_study, study: str) -> List[SeriesRecV2]:
    """`SeriesRecV2` per series of one study, with `n` from the directory listing (no pixels)."""
    base = os.path.join(root, f"{split}_series", str(study))
    recs: List[SeriesRecV2] = []
    rows = rows_by_study.get(str(study), []) if isinstance(rows_by_study, dict) \
        else group_series_rows(rows_by_study).get(str(study), [])
    for series, fl, plane in rows:
        sdir = os.path.join(base, series)
        if not os.path.isdir(sdir):
            continue
        files = sorted(fn for fn in os.listdir(sdir) if fn.lower().endswith(".dcm"))
        if not files:
            continue
        recs.append(SeriesRecV2(series, plane, fl, len(files),
                                tuple(os.path.join(sdir, fn) for fn in files)))
    return recs


def decode_study_series(
    root: str,
    split: str,
    study: str,
    series_rows,
    crop_mm: float = V2_CROP_MM,
    pct: Sequence[float] = V2_PCT,
    verbose: bool = False,
    timers: Optional[Dict[str, float]] = None,
    size: int = CACHE_SIZE,
    timer_tag: Optional[str] = None,
    plan_cache: Optional[Dict] = None,
) -> Tuple[List, Dict[str, np.ndarray], Dict[str, np.ndarray], str, str]:
    """Steps 1-5 -> `(sels, vols, anat, side, source)`. THE EXPENSIVE HALF, run ONCE per study.

    `select_slot_series` picks at most 5 series; each is decoded once and IN FULL (the
    percentile normalisation is per SERIES), its anatomical order read from the DICOM headers,
    and the laterality taken from the metas the decode already returned.

    `plan_slots` is deliberately NOT called here. The S slot-phase TTA variants differ only in
    which indices of these same `vols` get stacked, so they all reuse this one decode —
    `assemble_stack` is the cheap half that runs S times.

    `series_rows` is `group_series_rows(<split>_series.csv)` (or the raw DataFrame, which is
    grouped on the spot — pass the dict in a loop).

    `size` is the decode edge (`decode_size_for` of the calling group). `plan_cache` is a
    per-STUDY scratch dict: when several decode groups decode the same study, the slot plan and
    the header-derived slice order do not depend on `(crop_mm, pct, size)`, so the second and
    later groups reuse them and pay only for pixels. Passing `plan_cache=None` (the default)
    reproduces the original single-decode code path exactly.
    """
    from .decode import read_series_v2  # lazy: the cache path must not need pydicom

    t = time.time()
    if plan_cache is not None and "sels" in plan_cache:
        sels, need = plan_cache["sels"], plan_cache["need"]
    else:
        recs = _study_series_records(root, split, series_rows, study)
        sels = select_slot_series(recs, k_scale=1.0, train=False)      # eval = full 64
        need = {s.rec.series: s.rec for s in sels if s.rec is not None}
        if plan_cache is not None:
            plan_cache["sels"], plan_cache["need"] = sels, need
    if timers is not None:
        timers["plan"] = timers.get("plan", 0.0) + (time.time() - t)

    t = time.time()
    vols: Dict[str, np.ndarray] = {}
    anat: Dict[str, np.ndarray] = {}
    metas_all: List[List[dict]] = []
    rank_cache = None if plan_cache is None else plan_cache.setdefault("anat", {})
    for uid, rec in need.items():
        paths = list(rec.paths)
        if rank_cache is not None and uid in rank_cache:
            order = rank_cache[uid]
        else:
            ranks, _method = series_ranks_from_dicom(paths)
            # anat[r] = index into `paths` (and into `vol`, which read_series_v2 aligns to paths)
            order = np.argsort(np.asarray(ranks), kind="stable").astype(np.int64)
            if rank_cache is not None:
                rank_cache[uid] = order
        vol, metas = read_series_v2(paths, size=int(size), crop_mm=float(crop_mm),
                                    pct=tuple(pct), verbose=verbose)
        vols[uid] = np.asarray(vol)
        anat[uid] = order
        if metas:
            metas_all.append(list(metas))
    if timers is not None:
        dt = time.time() - t
        timers["decode"] = timers.get("decode", 0.0) + dt
        if timer_tag:
            key = f"decode[{timer_tag}]"
            timers[key] = timers.get(key, 0.0) + dt

    side, source, _x = _side_from_metas(metas_all) if metas_all else (CANONICAL_SIDE, "no-meta", 0.0)
    return sels, vols, anat, side, source


def assemble_stack(
    sels: Sequence,
    vols: Dict[str, np.ndarray],
    anat: Dict[str, np.ndarray],
    side: str = CANONICAL_SIDE,
    phase: float = 0.0,
    timers: Optional[Dict[str, float]] = None,
    size: Optional[int] = None,
    context: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Steps 6-7 for ONE slot phase -> `(stack uint8 [K,D,D], slot_id, pos, valid)`.

    Pure indexing + an optional horizontal flip over volumes `decode_study_series` already
    produced: no pixels are re-read and nothing is re-decoded, which is what makes S-variant
    TTA cost S x forward rather than S x everything. `phase=0.0` is the baseline plan.

    `size` (D) defaults to the edge of the volumes themselves — `CACHE_SIZE` when `vols` is
    empty — so a group decoded at 512 assembles a [K,512,512] stack without the caller having
    to repeat itself.

    `context=N` (2.5D slot context) widens the stack to `[K, 2N+1, D, D]`: plane `c` of bag
    position `j` is the slice `(c - N) * step` slices away from the picked one in the series'
    ANATOMICAL order, clamped at the ends — `dataset_v2.context_ranks`, the SAME function the
    training dataset calls on its JPEG cache, walking the SAME `SlotPick.step` (so a left
    knee's reversed sagittal neighbourhood runs the same way in both paths). `context=0`
    returns the original 3-D `[K,D,D]` stack, bit for bit: nothing below it is touched.
    """
    t = time.time()
    picks = plan_slots(sels, side=side, train=False, phase=phase)
    k = len(picks)
    if size is None:
        size = int(next(iter(vols.values())).shape[-1]) if vols else int(CACHE_SIZE)
    size = int(size)
    ctx = int(context)
    stack = (np.zeros((k, size, size), dtype=np.uint8) if ctx <= 0
             else np.zeros((k, 2 * ctx + 1, size, size), dtype=np.uint8))
    slot_id = np.zeros(k, dtype=np.int64)
    pos = np.zeros(k, dtype=np.float32)
    valid = np.zeros(k, dtype=bool)
    for j, p in enumerate(picks):
        slot_id[j] = p.slot
        pos[j] = p.pos
        if not p.valid or p.series is None:
            continue
        vol = vols.get(p.series)
        order = anat.get(p.series)
        if vol is None or order is None or len(vol) == 0:
            continue
        if ctx <= 0:
            src = int(order[min(p.idx, len(order) - 1)])
            if src >= len(vol):
                continue
            img = vol[src]
            if p.flip:
                img = img[:, ::-1]                   # laterality canonicalisation
            stack[j] = img
            valid[j] = True
            continue
        # 2.5D: the neighbourhood in ANATOMICAL rank, mapped through `order` into the volume.
        ranks = context_ranks(p.idx, len(order), p.step, ctx)
        src_c = int(order[ranks[ctx]])
        if src_c >= len(vol):
            continue                                 # centre unreadable -> masked out
        for c, r in enumerate(ranks):
            s_ = int(order[r])
            img = vol[s_] if s_ < len(vol) else vol[src_c]
            if p.flip:
                img = img[:, ::-1]                   # laterality canonicalisation
            stack[j, c] = img
        valid[j] = True
    if timers is not None:
        timers["assemble"] = timers.get("assemble", 0.0) + (time.time() - t)
    return stack, slot_id, pos, valid


def build_study_stack_dicom(
    root: str,
    split: str,
    study: str,
    series_rows,
    crop_mm: float = V2_CROP_MM,
    pct: Sequence[float] = V2_PCT,
    verbose: bool = False,
    timers: Optional[Dict[str, float]] = None,
    phase: float = 0.0,
    size: int = CACHE_SIZE,
    context: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    """One study -> `(stack uint8 [K,D,D], slot_id [K], pos [K], valid [K], side, source)`.

    Steps 1-7 of the module docstring = `decode_study_series` + `assemble_stack`, i.e. exactly
    what this function has always done (`phase=0.0` -> the baseline positions). Kept as the
    one-shot entry point for callers that want a single bag; the S-variant predictor calls the
    two halves separately so the decode is paid once.
    """
    sels, vols, anat, side, source = decode_study_series(
        root, split, study, series_rows, crop_mm=crop_mm, pct=pct, verbose=verbose,
        timers=timers, size=size)
    stack, slot_id, pos, valid = assemble_stack(sels, vols, anat, side=side, phase=phase,
                                                timers=timers, size=size, context=context)
    return stack, slot_id, pos, valid, side, source


def _stack_to_tensor(stack: np.ndarray, size: int, mean: np.ndarray, std: np.ndarray,
                     context: int = 0) -> np.ndarray:
    """uint8 [K,D,D] (or [K,C,D,D]) -> float32 [K,3,size,size], INTER_AREA + (x-mean)/std.

    Identical arithmetic to `dataset_hr._read_slice_img` + the dataset's normalisation, which
    is what a member trained at `--size 336` off the 384 cache saw.

    The stack is assembled at the DECODE GROUP's max `slice_context` (`context_for`), so it may
    carry C = 2*ctx+1 planes while THIS member wants fewer. `context=0` therefore takes the
    CENTRE plane and replicates it x3 — the original expression, byte for byte, whether the
    stack came in 3-D or 4-D — and `context=1` uses the three planes as the channels.
    """
    k = stack.shape[0]
    ctx = int(context)
    if ctx > 0 and stack.ndim != 4:
        raise ValueError(f"member wants slice_context={ctx} but the assembled stack has shape "
                         f"{stack.shape} (no neighbour planes) — the decode group's context "
                         f"was computed without it")
    if ctx > 0 and stack.shape[1] != 2 * ctx + 1:
        raise ValueError(f"member wants slice_context={ctx} ({2 * ctx + 1} channels) but the "
                         f"stack carries {stack.shape[1]}")
    out = np.empty((k, 3, size, size), dtype=np.float32)
    for j in range(k):
        if ctx <= 0:
            img = stack[j] if stack.ndim == 3 else stack[j, stack.shape[1] // 2]
            if img.shape != (size, size):
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            f = img.astype(np.float32) / 255.0
            out[j] = (np.stack([f, f, f], axis=0) - mean) / std
        else:
            planes = []
            for c in range(stack.shape[1]):
                img = stack[j, c]
                if img.shape != (size, size):
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                planes.append(img.astype(np.float32) / 255.0)
            out[j] = (np.stack(planes, axis=0) - mean) / std
    return out


# --------------------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------------------
def _combine_v2(per_member: Dict[int, pd.DataFrame], order: Sequence[str], w: np.ndarray,
             mode: str) -> pd.DataFrame:
    common = [s for s in order if all(s in df.index for df in per_member.values())]
    acc = np.zeros((len(common), len(TARGETS)), dtype=np.float64)
    for i, df in per_member.items():
        vals = df.loc[common][TARGETS].values.astype(np.float64)
        if mode == "rank":
            vals = pd.DataFrame(vals).rank(pct=True).values
        acc += w[i] * vals
    acc /= max(1e-12, float(w.sum()))
    out = pd.DataFrame(acc, columns=TARGETS)
    out.insert(0, ID_COL, common)
    return out


def _is_dicom_root(root: str, split: str) -> bool:
    """Does `<root>/<split>_series` hold `.dcm` files? (peeks at ONE series, never scans)."""
    base = os.path.join(root, f"{split}_series")
    if not os.path.isdir(base):
        return False
    for study in sorted(os.listdir(base))[:1]:
        sd = os.path.join(base, study)
        if not os.path.isdir(sd):
            continue
        for series in sorted(os.listdir(sd))[:1]:
            p = os.path.join(sd, series)
            if os.path.isdir(p):
                for fn in sorted(os.listdir(p))[:1]:
                    return fn.lower().endswith(".dcm")
    return False


def _lbl_idx(pos: int, phase: float) -> int:
    """Ladder index of `phase` when it has one, else its position in the requested list."""
    i = variant_index_of(phase)
    return i if i >= 0 else int(pos)


def _combine_variants_v2(frames: Dict[int, pd.DataFrame], phases: Sequence[float],
                         tta_combine: str, verbose: bool = True) -> pd.DataFrame:
    """Per-column rank (or prob) average of the S variant frames — one combiner, repo-wide.

    `infer_tta.rank_average_variant_frames` is the SAME function the base/hr arms combine
    their TTA frames with (and `dev/tta_eval.py --check-combiner` asserts it agrees with
    `infer_plane.rank_average_frames`, the across-arm combiner). Imported lazily so this
    module keeps its current import graph — and so the Kaggle bundle, which inlines `infer_v2`
    BEFORE `infer_tta`, resolves the name from the flat namespace at CALL time (the stripper
    deletes the relative import; `reports/multi_notebook_v2.md` §1).

    S=1 returns the single frame untouched: no rank transform, no reordering, so `tta=1` is
    byte-identical to the pre-TTA predictor.
    """
    ordered = [frames[i] for i in range(len(phases))]
    if len(ordered) == 1:
        return ordered[0]
    from .infer_tta import rank_average_variant_frames

    if verbose:
        counts = {f"v{i}": len(f) for i, f in enumerate(ordered)}
        print(f"[tta] v2: combining {len(ordered)} variant frame(s) by {tta_combine}-average; "
              f"rows {counts}")
    return rank_average_variant_frames(ordered, mode=tta_combine)


@torch.no_grad()
def predict_studies_v2(
    members: Sequence[MemberV2],
    input_root: Union[str, Mapping[int, str]],
    split: str = "test",
    batch: int = 2,
    workers: int = 2,
    device: str = "cuda",
    study_ids: Optional[Sequence[str]] = None,
    mode: str = "prob",
    weights: Optional[Sequence[float]] = None,
    side_parquet: Optional[str] = None,
    order_parquet: Optional[str] = None,
    dicom_root: Optional[str] = None,
    verbose_decode: bool = False,
    tta: Union[int, Sequence[float]] = 1,
    tta_combine: str = "rank",
    return_frames: bool = False,
    index=None,
):
    """Ensemble prediction -> DataFrame `[ID_COL + 12 columns]`, one row per study.

    Two input paths, chosen automatically:
      * a **DICOM mount** (Kaggle, or `/tmp/dicom_mount`): decode once per study PER DECODE
        GROUP `(crop_mm, pct)`, per-series percentile + that group's physical crop through
        `decode.read_series_v2` at `decode_size_for(group)`, side from the headers;
      * a **cache root** (`data/cache_c140_384`, split `train`): the training dataset itself,
        with `k_scale=1.0` and the side parquet — the OOF-style evaluation path.

    `tta=S` evaluates `dataset_v2.variant_phases(S)` slot phases and combines the S finished
    frames by `tta_combine` ("rank" — the same per-column rank average every other arm uses —
    or "prob"). On the DICOM path the S bags share ONE decode per study. `tta=1` is the literal
    pre-TTA path. `tta` also accepts an explicit list of phases (so a resumable evaluator can
    recompute just the variants it is missing, the way `infer_tta` takes a variant list);
    `return_frames=True` then also returns `{position_in_that_list: frame}`. `index=` reuses a
    prebuilt `build_index_v2` manifest on the cache path.

    The returned frame is exactly the shape `infer_plane.rank_average_frames` consumes, so
    this arm joins the multi-arm rank ensemble with no adapter.
    """
    if not members:
        raise ValueError("no members")
    if mode not in ("prob", "rank"):
        raise ValueError(f"mode must be 'prob' or 'rank', got {mode!r}")
    if tta_combine not in ("rank", "prob"):
        raise ValueError(f"tta_combine must be 'rank' or 'prob', got {tta_combine!r}")
    w = np.ones(len(members)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(w) != len(members):
        raise ValueError("weights must match members")
    phases = (variant_phases(max(1, int(tta))) if isinstance(tta, (int, np.integer))
              else [float(p) for p in tta])
    if not phases:
        raise ValueError("tta: no variants")

    root = next(iter(input_root.values())) if isinstance(input_root, Mapping) else str(input_root)
    groups: Dict[Tuple, List[int]] = {}
    for i, m in enumerate(members):
        groups.setdefault(m.signature, []).append(i)
    dgroups = decode_groups_v2(members)
    print(f"[predict] {len(members)} v2 member(s) in {len(dgroups)} decode group(s): "
          f"{[(key[0], decode_size_for(members, ix), len(ix)) for key, ix in dgroups.items()]} "
          f"(crop_mm, decode_size, n_members)")
    print(f"[predict] {len(groups)} size group(s): "
          f"{[(sig[0], len(ix)) for sig, ix in groups.items()]} (size, n)")
    if len(phases) > 1:
        print(f"[tta] v2: S={len(phases)} slot phase(s), {tta_combine}-averaged across variants")
        for i, ph in enumerate(phases):
            print(f"   {variant_label(_lbl_idx(i, ph), ph)}")

    if _is_dicom_root(root, split):
        frames = _predict_dicom(members, root, split, groups, batch, device, study_ids, mode, w,
                                verbose_decode, phases)
    else:
        frames = _predict_cache(members, root, split, groups, batch, workers, device, study_ids,
                                mode, w, side_parquet, order_parquet, dicom_root, phases, index)
    combined = _combine_variants_v2(frames, phases, tta_combine)
    return (combined, frames) if return_frames else combined


def _predict_dicom(members, root, split, groups, batch, device, study_ids, mode, w,
                   verbose_decode, phases=(0.0,), decode_size_override=None
                   ) -> Dict[int, pd.DataFrame]:
    series_csv = os.path.join(root, f"{split}_series.csv")
    series_rows = group_series_rows(pd.read_csv(series_csv))
    base = os.path.join(root, f"{split}_series")
    have = set(os.listdir(base)) if os.path.isdir(base) else set()
    ids = [str(s) for s in (study_ids if study_ids is not None else sorted(series_rows))
           if str(s) in have]

    # --- decode groups ------------------------------------------------------------------
    # One decode per (crop_mm, pct) per study, at that group's `decode_size`; inside a group
    # the old (size, mean, std) signature split still decides how many normalised tensors are
    # built. `groups` (the caller's signature map) is re-derived per decode group so that with
    # ONE decode group the iteration order — and therefore every number — is unchanged.
    specs: List[Dict] = []
    for key, idxs in decode_groups_v2(members).items():
        dsize = int(decode_size_override) if decode_size_override else decode_size_for(members, idxs)
        sub: Dict[Tuple, List[int]] = {}
        for i in idxs:
            sub.setdefault(members[i].signature, []).append(i)
        specs.append({"crop_mm": key[0], "pct": key[1], "size": dsize, "sub": sub,
                      "n": len(idxs), "tag": f"c{key[0]:g}@{dsize}",
                      "context": context_for(members, idxs)})
    G = len(specs)
    print(f"[predict] DICOM path: {len(ids)} studies from {root} (K={K_FULL}); "
          f"decode groups: {[(s['crop_mm'], s['size'], s['n']) for s in specs]} "
          f"(crop_mm, decode_size, n_members)")
    for s in specs:
        print(f"   decode group {s['tag']}: crop_mm={s['crop_mm']} pct={s['pct']} "
              f"decode_size={s['size']} slice_context={s['context']} "
              f"members={[members[i].name for ix in s['sub'].values() for i in ix]}")

    # Buffers are keyed by (variant, decode group). They hold the uint8 [K,D,D] stacks, NOT the
    # normalised float tensors: at S=3 x batch 2 x one 384 group that is ~56 MB instead of
    # ~680 MB, and `_stack_to_tensor` is a pure function of the stack, so keeping the call in
    # flush() changes no number (the S=1, one-group output stays byte-identical). A second
    # decode group adds one more uint8 stack per (variant, study-in-batch) — another 57 MB at
    # 384, 101 MB at 512, for S=3 x batch 2 — and is freed by the same flush().
    S = len(phases)
    timers: Dict[str, float] = {}
    per_member: Dict[Tuple[int, int], List[np.ndarray]] = {(v, i): []
                                                           for v in range(S)
                                                           for i in range(len(members))}
    order_ids: List[str] = []
    buf: Dict[Tuple[int, int], List[np.ndarray]] = {(v, g): []
                                                    for v in range(S) for g in range(G)}
    buf_meta: Dict[Tuple[int, int], List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        (v, g): [] for v in range(S) for g in range(G)}

    def flush():
        if not buf_meta[(0, 0)]:
            return
        t = time.time()
        for v in range(S):
            for g, spec in enumerate(specs):
                meta = buf_meta[(v, g)]
                sid = torch.from_numpy(np.stack([b[0] for b in meta])).to(device)
                pos = torch.from_numpy(np.stack([b[1] for b in meta])).to(device)
                val = torch.from_numpy(np.stack([b[2] for b in meta])).to(device)
                for sig, idxs in spec["sub"].items():
                    mm = members[idxs[0]]
                    x = torch.from_numpy(
                        np.stack([_stack_to_tensor(s, mm.size, mm.mean, mm.std,
                                                   mm.slice_context)
                                  for s in buf[(v, g)]])).to(device, non_blocking=True)
                    for i in idxs:
                        logits = _forward_with_guard(members[i].model, x, device,
                                                     slot_id=sid, pos=pos, valid=val)
                        per_member[(v, i)].append(torch.sigmoid(logits.float()).cpu().numpy())
                    del x
                buf[(v, g)].clear()
                buf_meta[(v, g)].clear()
        timers["forward"] = timers.get("forward", 0.0) + (time.time() - t)

    for n, study in enumerate(ids):
        # ONE decode per (crop_mm, pct) per study, shared by every variant and every size group
        # inside it. `plan_cache` keeps the slot plan and the DICOM-header slice order across
        # the groups of THIS study, so a second crop costs pixels only.
        plan_cache: Dict = {}
        n_valid, side, source = 0, CANONICAL_SIDE, "no-meta"
        for g, spec in enumerate(specs):
            sels, vols, anat, side, source = decode_study_series(
                root, split, study, series_rows, crop_mm=spec["crop_mm"], pct=spec["pct"],
                verbose=verbose_decode, timers=timers, size=spec["size"],
                timer_tag=(spec["tag"] if G > 1 else None),
                plan_cache=(plan_cache if G > 1 else None))
            for v, ph in enumerate(phases):
                stack, slot_id, pos, valid = assemble_stack(sels, vols, anat, side=side,
                                                            phase=ph, timers=timers,
                                                            size=spec["size"],
                                                            context=spec["context"])
                buf[(v, g)].append(stack)
                buf_meta[(v, g)].append((slot_id, pos, valid))
                if g == 0 and v == 0:
                    n_valid = int(valid.sum())
            del vols, anat
        print(f"  [study {n + 1}/{len(ids)}] {study[-12:]} side={side} ({source}) "
              f"valid={n_valid}/{K_FULL}" + (f" x{S} variants" if S > 1 else "")
              + (f" x{G} decode groups" if G > 1 else ""), flush=True)
        order_ids.append(str(study))
        if len(buf_meta[(0, 0)]) >= max(1, int(batch)):
            flush()
    flush()
    print(" ", fallback_report()
          + (f"  [padded counts each of the {S} variant(s) x {G} decode group(s) separately]"
             if S * G > 1 else ""))
    for k_, v_ in sorted(timers.items()):
        print(f"[time] stage=v2_{k_} elapsed={v_:.1f}s", flush=True)

    out: Dict[int, pd.DataFrame] = {}
    for v in range(S):
        frames = {i: pd.DataFrame(
            np.concatenate(per_member[(v, i)], axis=0) if per_member[(v, i)]
            else np.zeros((0, len(TARGETS)), np.float32),
            columns=TARGETS, index=order_ids) for i in range(len(members))}
        out[v] = _combine_v2(frames, order_ids, w, mode)
    return out


def _predict_cache(members, root, split, groups, batch, workers, device, study_ids, mode, w,
                   side_parquet, order_parquet, dicom_root, phases=(0.0,),
                   index=None) -> Dict[int, pd.DataFrame]:
    from .order import SliceOrderResolver

    if index is None:
        resolver = SliceOrderResolver.from_parquet(order_parquet, dicom_root=dicom_root,
                                                   split=split)
        index = build_index_v2(root, split=split, order=resolver)
    side_map = load_side_map(side_parquet)
    # Consistency with the DICOM path: a cache root IS one (crop_mm, size) rendering, so more
    # than one decode group cannot be served here, and a member whose recorded crop disagrees
    # with the root it is being scored on is the exact bug the DICOM grouping now prevents.
    dgroups = decode_groups_v2(members)
    print(f"[predict] cache path: {len(members)} v2 member(s) in {len(dgroups)} decode "
          f"group(s): {[(k[0], decode_size_for(members, ix), len(ix)) for k, ix in dgroups.items()]} "
          f"(crop_mm, decode_size, n_members)")
    root_crop, root_size = parse_cache_contract(root)
    if root_crop is not None:
        if len(dgroups) > 1:
            print(f"[predict] WARNING cache root {os.path.basename(root)} renders ONE crop "
                  f"({root_crop} mm); members span {sorted(k[0] for k in dgroups)} — every "
                  f"member is being scored on {root_crop} mm regardless of its metadata")
        for m in members:
            if abs(float(m.crop_mm) - float(root_crop)) > 1e-6:
                print(f"[predict] WARNING {m.name}: crop_mm={m.crop_mm} but cache root is "
                      f"{root_crop} mm — scoring it on an input contract it was not trained on")
            if root_size is not None and int(m.size) > int(root_size):
                print(f"[predict] WARNING {m.name}: size={m.size} > cache edge {root_size} "
                      f"— the bag will be UP-sampled")
    ids = list(index.keys()) if study_ids is None else \
        [str(s) for s in study_ids if str(s) in index]
    out: Dict[int, pd.DataFrame] = {}
    # No per-study decode here: the dataset reads ONE cached JPEG per picked slice, so a
    # variant costs one loader pass over the picked slices (there is nothing to re-decode).
    for v, ph in enumerate(phases):
        per_member: Dict[int, pd.DataFrame] = {}
        order_ids: Optional[List[str]] = None
        for sig, idxs in groups.items():
            m0 = members[idxs[0]]
            ds = StudyBagDatasetV2(ids, index, labels=None, size=m0.size, train=False,
                                   mean=np.asarray(m0.mean).ravel(),
                                   std=np.asarray(m0.std).ravel(),
                                   k_scale=1.0, side_map=side_map, phase=ph,
                                   slice_context=m0.slice_context)
            loader = DataLoader(ds, batch_size=max(1, int(batch)), shuffle=False,
                                num_workers=workers, pin_memory=str(device).startswith("cuda"))
            print(f"[predict] cache path: group size={m0.size} "
                  f"slice_context={m0.slice_context}, {len(ds)} studies from {root}"
                  + (f" [{variant_label(_lbl_idx(v, ph), ph)}]" if len(phases) > 1 else ""))
            got: List[str] = []
            chunks: Dict[int, List[np.ndarray]] = {i: [] for i in idxs}
            for x, sid_t, pos, valid, _y, _w, sid in loader:
                x = x.to(device, non_blocking=True)
                kw = dict(slot_id=sid_t.to(device), pos=pos.to(device), valid=valid.to(device))
                for i in idxs:
                    logits = _forward_with_guard(members[i].model, x, device, **kw)
                    chunks[i].append(torch.sigmoid(logits.float()).cpu().numpy())
                got.extend(list(sid))
            for i in idxs:
                arr = np.concatenate(chunks[i], axis=0) if chunks[i] else \
                    np.zeros((0, len(TARGETS)), np.float32)
                per_member[i] = pd.DataFrame(arr, columns=TARGETS, index=[str(s) for s in got])
            if order_ids is None:
                order_ids = [str(s) for s in got]
        out[v] = _combine_v2(per_member, order_ids or [], w, mode)
    print(" ", fallback_report())
    return out


# --------------------------------------------------------------------------------------
def run_submission_v2(
    input_root: Optional[Union[str, Mapping[int, str]]] = None,
    checkpoint_paths: Sequence[str] = (),
    output_path: str = "submission.csv",
    batch: int = 2,
    workers: int = 2,
    device: str = "cuda",
    mode: str = "prob",
    weights: Optional[Sequence[float]] = None,
    split: str = "test",
    side_parquet: Optional[str] = None,
    order_parquet: Optional[str] = None,
    dicom_root: Optional[str] = None,
    tta: int = 1,
    tta_combine: str = "rank",
) -> pd.DataFrame:
    """Survival flow: 0.5 file FIRST -> mount -> device probe -> members -> predict -> write."""
    t = time.time()
    if input_root is None:
        input_root = find_data_root("/kaggle/input")
        if input_root is None:
            raise FileNotFoundError("could not locate data mount (test.csv + test_series/)")
    csv_root = next(iter(input_root.values())) if isinstance(input_root, Mapping) else input_root
    print(f"[mount] input_root={input_root}")
    test_df = pd.read_csv(os.path.join(csv_root, f"{split}.csv"))
    print(f"[self-size] {split}.csv rows = {len(test_df)}")
    _stage_v2("mount_discovery", t)

    # 1) a valid all-0.5 submission exists from the first second
    write_benchmark_submission(test_df, output_path)
    print(f"[benchmark] wrote all-0.5 {output_path}")

    # 1b) device safety: an unsupported GPU arch must fall back to CPU, not crash the kernel.
    if str(device).startswith("cuda"):
        ok_cuda = False
        try:
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(0)
                sm = f"sm_{cap[0]}{cap[1]}"
                archs = list(torch.cuda.get_arch_list())
                if sm in archs:
                    _t = torch.zeros(8, device=device); _ = (_t + 1).sum().item()
                    ok_cuda = True
                else:
                    print(f"[device] GPU {sm} not in supported {archs}; using CPU")
            else:
                print("[device] cuda not available; using CPU")
        except Exception as e:
            print(f"[device] CUDA probe failed ({type(e).__name__}: {e}); using CPU")
        if not ok_cuda:
            device = "cpu"
    print(f"[device] using {device}")

    # 2) weights (pretrained=False) + advisory fingerprint check
    t = time.time()
    try:
        members = load_members_v2(checkpoint_paths, device=device)
    except Exception as e:
        if str(device).startswith("cuda"):
            print(f"[device] CUDA load failed ({type(e).__name__}: {e}); CPU fallback")
            device = "cpu"
            members = load_members_v2(checkpoint_paths, device=device)
        else:
            raise
    for m in members:
        if not check_fingerprint_v2(m, device):
            print(f"[fingerprint] {m.name} mismatch -> ADVISORY (cross-arch drift expected); "
                  f"proceeding with REAL predictions")
    _stage_v2("member_load[v2]", t)

    # 3) predict, self-sized to the hidden test's study list
    t = time.time()
    preds = predict_studies_v2(members, input_root, split, batch=batch, workers=workers,
                               device=device, study_ids=test_df[ID_COL].astype(str).tolist(),
                               mode=mode, weights=weights, side_parquet=side_parquet,
                               order_parquet=order_parquet, dicom_root=dicom_root,
                               tta=tta, tta_combine=tta_combine)
    _stage_v2("predict[v2]", t)

    # 4) rank + reindex + fillna(0.5) + write
    t = time.time()
    sub = write_submission(preds, test_df, output_path)
    _stage_v2("write", t)
    print(f"[submission] wrote {output_path} rows={len(sub)} cols={list(sub.columns)}")
    _stage_v2("total", _T0)
    return sub


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="v2 arm inference (local check)")
    ap.add_argument("--input-root", required=True, help="DICOM mount or data/cache_c140_384")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", default="submission_v2.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--member-mode", default="prob", choices=("prob", "rank"))
    ap.add_argument("--side-parquet", default=None)
    ap.add_argument("--order-parquet", default=None)
    ap.add_argument("--dicom-root", default=None)
    ap.add_argument("--tta", type=int, default=1,
                    help="S slot-phase variants (dataset_v2.variant_phases); 1 = off and "
                         "byte-identical to the pre-TTA path")
    ap.add_argument("--tta-combine", default="rank", choices=("rank", "prob"))
    a = ap.parse_args(argv)

    sub = run_submission_v2(input_root=a.input_root, checkpoint_paths=a.ckpt,
                            output_path=a.out, batch=a.batch, workers=a.workers,
                            device=a.device, mode=a.member_mode, split=a.split,
                            side_parquet=a.side_parquet, order_parquet=a.order_parquet,
                            dicom_root=a.dicom_root, tta=a.tta, tta_combine=a.tta_combine)
    print(sub.head().to_string())
    n_unique = sub[TARGETS].round(6).drop_duplicates().shape[0]
    print(f"[check] rows={len(sub)} cols={len(sub.columns)} distinct prediction rows={n_unique}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MemberV2", "load_members_v2", "predict_studies_v2", "run_submission_v2", "is_v2_meta",
    "check_fingerprint_v2", "compute_fingerprint_v2", "build_study_stack_dicom",
    "decode_study_series", "assemble_stack", "has_read_series_v2", "MODE_TAG_V2", "CACHE_SIZE",
    "decode_groups_v2", "decode_size_for", "context_for", "parse_cache_contract",
]
