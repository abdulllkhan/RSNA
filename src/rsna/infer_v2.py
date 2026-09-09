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
 7. one uint8 [K,384,384] canonical stack per study -> resized per member `size` -> normalised
    per member (mean,std) -> forwarded through every member of that size group.

Members of different `size` share the 384 decode and differ only by an INTER_AREA resize,
which is exactly what `dataset_v2` does when trained at `--size 336` off the 384 cache.

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
    fallback_report,
    load_side_map,
    plan_slots,
    select_slot_series,
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
                 pct: Sequence[float] = V2_PCT, backbone: str = ""):
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

    @property
    def signature(self) -> Tuple:
        """Members sharing this share ONE normalised tensor (decode is shared by all)."""
        return (self.size,
                tuple(np.asarray(self.mean).ravel().round(6).tolist()),
                tuple(np.asarray(self.std).ravel().round(6).tolist()))

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
                     pct=tuple(a.get("pct", V2_PCT)), backbone=cfg.backbone)
        print(f"[member] {m.name}: mode=v2 size={m.size} K_eval={m.k_eval} "
              f"(trained k_scale={m.k_scale}) crop_mm={m.crop_mm} pct={m.pct} "
              f"slot_prior={m.slot_prior} backbone={m.backbone}")
        members.append(m)
    return members


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


def build_study_stack_dicom(
    root: str,
    split: str,
    study: str,
    series_rows,
    crop_mm: float = V2_CROP_MM,
    pct: Sequence[float] = V2_PCT,
    verbose: bool = False,
    timers: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    """One study -> `(stack uint8 [K,384,384], slot_id [K], pos [K], valid [K], side, source)`.

    Steps 1-7 of the module docstring. Every chosen series is decoded once and in full
    (the percentile normalisation is per SERIES); nothing else is touched.

    `series_rows` is `group_series_rows(<split>_series.csv)` (or the raw DataFrame, which is
    grouped on the spot — pass the dict in a loop).
    """
    from .decode import read_series_v2  # lazy: the cache path must not need pydicom

    t = time.time()
    recs = _study_series_records(root, split, series_rows, study)
    sels = select_slot_series(recs, k_scale=1.0, train=False)          # eval = full 64
    need = {s.rec.series: s.rec for s in sels if s.rec is not None}
    if timers is not None:
        timers["plan"] = timers.get("plan", 0.0) + (time.time() - t)

    t = time.time()
    vols: Dict[str, np.ndarray] = {}
    anat: Dict[str, np.ndarray] = {}
    metas_all: List[List[dict]] = []
    for uid, rec in need.items():
        paths = list(rec.paths)
        ranks, _method = series_ranks_from_dicom(paths)
        vol, metas = read_series_v2(paths, size=CACHE_SIZE, crop_mm=float(crop_mm),
                                    pct=tuple(pct), verbose=verbose)
        vols[uid] = np.asarray(vol)
        # anat[r] = index into `paths` (and into `vol`, which read_series_v2 aligns to paths)
        anat[uid] = np.argsort(np.asarray(ranks), kind="stable").astype(np.int64)
        if metas:
            metas_all.append(list(metas))
    if timers is not None:
        timers["decode"] = timers.get("decode", 0.0) + (time.time() - t)

    t = time.time()
    side, source, _x = _side_from_metas(metas_all) if metas_all else (CANONICAL_SIDE, "no-meta", 0.0)
    picks = plan_slots(sels, side=side, train=False)
    k = len(picks)
    stack = np.zeros((k, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
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
        src = int(order[min(p.idx, len(order) - 1)])
        if src >= len(vol):
            continue
        img = vol[src]
        if p.flip:
            img = img[:, ::-1]                       # laterality canonicalisation
        stack[j] = img
        valid[j] = True
    if timers is not None:
        timers["assemble"] = timers.get("assemble", 0.0) + (time.time() - t)
    return stack, slot_id, pos, valid, side, source


def _stack_to_tensor(stack: np.ndarray, size: int, mean: np.ndarray, std: np.ndarray
                     ) -> np.ndarray:
    """uint8 [K,384,384] -> float32 [K,3,size,size], INTER_AREA resize + (x-mean)/std.

    Identical arithmetic to `dataset_hr._read_slice_img` + the dataset's normalisation, which
    is what a member trained at `--size 336` off the 384 cache saw.
    """
    k = stack.shape[0]
    out = np.empty((k, 3, size, size), dtype=np.float32)
    for j in range(k):
        img = stack[j]
        if img.shape != (size, size):
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        f = img.astype(np.float32) / 255.0
        out[j] = (np.stack([f, f, f], axis=0) - mean) / std
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
) -> pd.DataFrame:
    """Ensemble prediction -> DataFrame `[ID_COL + 12 columns]`, one row per study.

    Two input paths, chosen automatically:
      * a **DICOM mount** (Kaggle, or `/tmp/dicom_mount`): decode once per study, per-series
        percentile + 140 mm crop through `decode.read_series_v2`, side from the headers;
      * a **cache root** (`data/cache_c140_384`, split `train`): the training dataset itself,
        with `k_scale=1.0` and the side parquet — the OOF-style evaluation path.

    The returned frame is exactly the shape `infer_plane.rank_average_frames` consumes, so
    this arm joins the multi-arm rank ensemble with no adapter.
    """
    if not members:
        raise ValueError("no members")
    if mode not in ("prob", "rank"):
        raise ValueError(f"mode must be 'prob' or 'rank', got {mode!r}")
    w = np.ones(len(members)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(w) != len(members):
        raise ValueError("weights must match members")

    root = next(iter(input_root.values())) if isinstance(input_root, Mapping) else str(input_root)
    groups: Dict[Tuple, List[int]] = {}
    for i, m in enumerate(members):
        groups.setdefault(m.signature, []).append(i)
    print(f"[predict] {len(members)} v2 member(s) in {len(groups)} size group(s): "
          f"{[(sig[0], len(ix)) for sig, ix in groups.items()]} (size, n)")

    if _is_dicom_root(root, split):
        return _predict_dicom(members, root, split, groups, batch, device, study_ids, mode, w,
                              verbose_decode)
    return _predict_cache(members, root, split, groups, batch, workers, device, study_ids,
                          mode, w, side_parquet, order_parquet, dicom_root)


def _predict_dicom(members, root, split, groups, batch, device, study_ids, mode, w,
                   verbose_decode) -> pd.DataFrame:
    series_csv = os.path.join(root, f"{split}_series.csv")
    series_rows = group_series_rows(pd.read_csv(series_csv))
    base = os.path.join(root, f"{split}_series")
    have = set(os.listdir(base)) if os.path.isdir(base) else set()
    ids = [str(s) for s in (study_ids if study_ids is not None else sorted(series_rows))
           if str(s) in have]
    m0 = members[0]
    if any((m.crop_mm, m.pct) != (m0.crop_mm, m0.pct) for m in members):
        print(f"[predict] WARNING members disagree on (crop_mm, pct); decoding with "
              f"{m0.name}'s ({m0.crop_mm}, {m0.pct}) for all of them")
    print(f"[predict] DICOM path: {len(ids)} studies from {root} "
          f"(crop_mm={m0.crop_mm} pct={m0.pct} K={K_FULL})")

    timers: Dict[str, float] = {}
    per_member: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(members))}
    order_ids: List[str] = []
    buf: Dict[Tuple, List[np.ndarray]] = {sig: [] for sig in groups}
    buf_meta: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def flush():
        if not buf_meta:
            return
        sid = torch.from_numpy(np.stack([b[0] for b in buf_meta])).to(device)
        pos = torch.from_numpy(np.stack([b[1] for b in buf_meta])).to(device)
        val = torch.from_numpy(np.stack([b[2] for b in buf_meta])).to(device)
        t = time.time()
        for sig, idxs in groups.items():
            x = torch.from_numpy(np.stack(buf[sig])).to(device, non_blocking=True)
            for i in idxs:
                logits = _forward_with_guard(members[i].model, x, device,
                                             slot_id=sid, pos=pos, valid=val)
                per_member[i].append(torch.sigmoid(logits.float()).cpu().numpy())
            buf[sig].clear()
        timers["forward"] = timers.get("forward", 0.0) + (time.time() - t)
        buf_meta.clear()

    for n, study in enumerate(ids):
        stack, slot_id, pos, valid, side, source = build_study_stack_dicom(
            root, split, study, series_rows, crop_mm=m0.crop_mm, pct=m0.pct,
            verbose=verbose_decode, timers=timers)
        print(f"  [study {n + 1}/{len(ids)}] {study[-12:]} side={side} ({source}) "
              f"valid={int(valid.sum())}/{len(valid)}", flush=True)
        for sig, idxs in groups.items():
            mm = members[idxs[0]]
            buf[sig].append(_stack_to_tensor(stack, mm.size, mm.mean, mm.std))
        buf_meta.append((slot_id, pos, valid))
        order_ids.append(str(study))
        if len(buf_meta) >= max(1, int(batch)):
            flush()
    flush()
    print(" ", fallback_report())
    for k_, v_ in sorted(timers.items()):
        print(f"[time] stage=v2_{k_} elapsed={v_:.1f}s", flush=True)

    frames = {i: pd.DataFrame(
        np.concatenate(per_member[i], axis=0) if per_member[i]
        else np.zeros((0, len(TARGETS)), np.float32),
        columns=TARGETS, index=order_ids) for i in range(len(members))}
    return _combine_v2(frames, order_ids, w, mode)


def _predict_cache(members, root, split, groups, batch, workers, device, study_ids, mode, w,
                   side_parquet, order_parquet, dicom_root) -> pd.DataFrame:
    from .order import SliceOrderResolver

    resolver = SliceOrderResolver.from_parquet(order_parquet, dicom_root=dicom_root, split=split)
    index = build_index_v2(root, split=split, order=resolver)
    side_map = load_side_map(side_parquet)
    ids = list(index.keys()) if study_ids is None else \
        [str(s) for s in study_ids if str(s) in index]
    per_member: Dict[int, pd.DataFrame] = {}
    order_ids: Optional[List[str]] = None
    for sig, idxs in groups.items():
        m0 = members[idxs[0]]
        ds = StudyBagDatasetV2(ids, index, labels=None, size=m0.size, train=False,
                               mean=np.asarray(m0.mean).ravel(), std=np.asarray(m0.std).ravel(),
                               k_scale=1.0, side_map=side_map)
        loader = DataLoader(ds, batch_size=max(1, int(batch)), shuffle=False,
                            num_workers=workers, pin_memory=str(device).startswith("cuda"))
        print(f"[predict] cache path: group size={m0.size}, {len(ds)} studies from {root}")
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
    print(" ", fallback_report())
    return _combine_v2(per_member, order_ids or [], w, mode)


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
                               order_parquet=order_parquet, dicom_root=dicom_root)
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
    a = ap.parse_args(argv)

    sub = run_submission_v2(input_root=a.input_root, checkpoint_paths=a.ckpt,
                            output_path=a.out, batch=a.batch, workers=a.workers,
                            device=a.device, mode=a.member_mode, split=a.split,
                            side_parquet=a.side_parquet, order_parquet=a.order_parquet,
                            dicom_root=a.dicom_root)
    print(sub.head().to_string())
    n_unique = sub[TARGETS].round(6).drop_duplicates().shape[0]
    print(f"[check] rows={len(sub)} cols={len(sub.columns)} distinct prediction rows={n_unique}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MemberV2", "load_members_v2", "predict_studies_v2", "run_submission_v2", "is_v2_meta",
    "check_fingerprint_v2", "compute_fingerprint_v2", "build_study_stack_dicom",
    "has_read_series_v2", "MODE_TAG_V2", "CACHE_SIZE",
]
