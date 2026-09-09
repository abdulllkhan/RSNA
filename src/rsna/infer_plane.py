"""Inference for the Wave-3 "A3" plane-specialised checkpoints (any input resolution).

`infer.py` (224, `StudyBagDataset` + `model.StudyModel`) and `infer_hr.py` (size-aware, same
architecture) both feed ONE mixed-plane bag. An A3 checkpoint needs three per-plane bags plus
a plane mask, so it gets its own predictor — and, as with the HR arm, the existing modules are
left untouched (running trainings import them).

What this module does:
  * `load_members_plane` rebuilds `model_plane.StudyPlaneModel` from the checkpoint's own
    `cfg` and reads `args["size"]` / `args["k_per_plane"]` / `args["plane_mode"]` from it, so
    one call handles 224 and 384 members and the `planewise`/`concat` ablation alike;
  * `predict_studies_plane` groups members by their decode signature `(size, k_per_plane,
    mean, std)` and builds ONE DataLoader per group — each DICOM slice is decoded once per
    (resolution, slot layout), not once per member, the same trick `infer.py`/`infer_hr.py`
    use (it is what keeps a 5-fold ensemble affordable on Kaggle's CPU);
  * decoding goes through `dataset_plane` -> `dataset_hr.read_slice_hr` ->
    `decode.read_dicom_slice_float(path, size)` for `.dcm` and cached JPEG otherwise, so a
    raw Kaggle DICOM mount and a local `data/cache<size>` tree are interchangeable;
  * `mode="rank"` averages per-column percentile ranks instead of probabilities, and
    `rank_average_frames` combines FINISHED prediction frames coming from *different arms*
    (baseline `infer.predict_studies`, HR `infer_hr.predict_studies_hr`, and this one) —
    PLAN §9's combiner for heterogeneous members, since AUC only sees within-column order;
  * `plane_weights_of` reads the learned 12x3 routing table back out of a checkpoint (the
    interpretable artefact of this arm).

Kept identical to `infer.py`: the capability-aware autocast rule (bf16 on sm_80+, fp16 on
sm_70-75, fp32 elsewhere, with an fp32 recompute if a forward fails or returns non-finite
values), `pretrained=False` weight loading, the advisory fingerprint self-check, and the
submission-writing helpers (imported from `infer.py`, not duplicated).

Local check against the visible test DICOM:

    from src.rsna.infer_plane import load_members_plane, predict_studies_plane
    ms = load_members_plane(["models/effv2s_plane_k8x3_224_fold0.pt"], device="cpu")
    preds = predict_studies_plane("/tmp/dicom_mount", "test", ms, batch=2, workers=2, device="cpu")
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import TARGETS, ID_COL
from .dataset_plane import (
    DEFAULT_K_PER_PLANE,
    N_PLANES,
    PLANE_ORDER,
    StudyPlaneBagDataset,
    build_plane_index,
)
from .infer import (  # reused verbatim -- no duplication, no edit to infer.py
    FINGERPRINT_TOL,
    find_data_root,
    write_benchmark_submission,
    write_submission,
)
from .model_plane import build_plane_model, config_from_dict

DEFAULT_SIZE = 224


def _autocast(device: str):
    """Capability-aware autocast dtype -- the single biggest GPU-inference lever we have.

        sm >= 8.0        (Ampere+: GB10, A100, L4, ...)  -> bf16
        7.0 <= sm < 8.0  (Turing T4, Volta V100)         -> fp16
        anything else, or CPU                            -> no-op (plain fp32)

    Turing has NO cuDNN engine for a *bf16* conv ("GET was unable to find an engine to
    execute this computation"), which is why this used to fall all the way back to fp32 --
    but it has full *fp16* tensor cores, and fp16 is its fastest path by roughly 2x. Kaggle
    scores this competition on a T4 (we pin `--accelerator NvidiaTeslaT4`), so the fp16 rung
    is worth about half of the GPU stage. fp16's narrow dynamic range (max 65504) is
    backstopped per batch by `_forward_with_guard`."""
    if str(device).startswith("cuda"):
        try:
            major, _minor = torch.cuda.get_device_capability(0)
        except Exception:
            major = 0
        if major >= 8:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        if major >= 7:
            return torch.autocast("cuda", dtype=torch.float16)  # T4/V100 tensor cores
        return nullcontext()  # pre-Volta: no usable half path
    return nullcontext()


# One flag for the whole run: the fp32 backstop is a correctness net, not a per-batch event
# worth thousands of log lines. (In the inlined Kaggle bundle infer/infer_hr/infer_plane
# collapse into ONE namespace, so this really is one flag across all arms -- keep the three
# copies of this block textually identical.)
_FP32_FALLBACK = {"logged": False}


def _forward_with_guard(model, x, device, **kw):
    """`model(x, **kw)` under `_autocast`, with an fp32 recompute for the two ways half
    precision can fail on hardware we cannot test on locally:

      * RuntimeError      -- no engine for this dtype/layout on this GPU;
      * non-finite output -- fp16 overflows above 65504, so one unlucky activation can
        produce inf/NaN where bf16 or fp32 would not. That failure is SILENT and would
        poison a whole column's ranks, so it is checked rather than hoped about.

    Either way the batch is recomputed in plain fp32 (contiguous, no autocast) and the run
    carries on; the run is never allowed to die into the 0.5 file. Logged once."""
    try:
        with _autocast(device):
            logits = model(x, **kw)
        if torch.isfinite(logits).all():
            return logits
        why = "non-finite output"
    except RuntimeError as e:
        why = f"RuntimeError: {str(e)[:70]}"
    if not _FP32_FALLBACK["logged"]:
        _FP32_FALLBACK["logged"] = True
        print(f"[predict] autocast forward -> fp32 fallback ({why}); "
              f"further fallbacks on this run are silent")
    return model(x.float().contiguous(), **kw)


# --------------------------------------------------------------------------------------
# Members
# --------------------------------------------------------------------------------------
class MemberPlane:
    """One loaded A3 checkpoint + the preprocessing/slot layout it was trained with."""

    def __init__(self, model, mean, std, k_per_plane: Sequence[int], size: int, fingerprint,
                 fusion: str = "planewise", name: str = ""):
        self.model = model
        self.mean = mean
        self.std = std
        self.k_per_plane = tuple(int(v) for v in k_per_plane)
        self.k_total = int(sum(self.k_per_plane))
        self.size = int(size)
        self.fingerprint = fingerprint
        self.fusion = str(fusion)
        self.name = name

    @property
    def signature(self) -> Tuple:
        """Members sharing this decode signature can share one DataLoader."""
        return (self.size, self.k_per_plane,
                tuple(round(float(v), 6) for v in np.asarray(self.mean).ravel()),
                tuple(round(float(v), 6) for v in np.asarray(self.std).ravel()))

    def __repr__(self) -> str:
        return (f"MemberPlane({self.name or 'ckpt'}, size={self.size}, "
                f"k_per_plane={self.k_per_plane}, fusion={self.fusion})")


def load_members_plane(checkpoint_paths: Sequence[str], device: str = "cuda") -> List[MemberPlane]:
    """Load A3 checkpoints with `pretrained=False`; k_per_plane/size/mean/std come from the ckpt."""
    members: List[MemberPlane] = []
    for cp in checkpoint_paths:
        st = torch.load(cp, map_location=device, weights_only=False)
        cfg = config_from_dict(st["cfg"])
        cfg.pretrained = False  # NO network at inference
        model = build_plane_model(cfg).to(device)
        if str(device).startswith("cuda"):
            model = model.to(memory_format=torch.channels_last)
        model.load_state_dict(st["model"])
        model.eval()
        a = st.get("args", {}) or {}
        kpp = a.get("k_per_plane") or list(cfg.k_per_plane) or list(DEFAULT_K_PER_PLANE)
        m = MemberPlane(model, st["mean"], st["std"], kpp,
                        int(a.get("size", DEFAULT_SIZE)), np.asarray(st.get("fingerprint")),
                        fusion=str(a.get("plane_mode", cfg.fusion)), name=os.path.basename(cp))
        print(f"[member] {m.name}: size={m.size} k_per_plane={m.k_per_plane} "
              f"fusion={m.fusion} backbone={cfg.backbone}")
        members.append(m)
    return members


def plane_weights_of(checkpoint_path: str) -> pd.DataFrame:
    """The learned [12 x 3] per-finding plane weights stored in an A3 checkpoint."""
    st = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    w = np.asarray(st.get("plane_weights"))
    order = list(st.get("plane_order", PLANE_ORDER))
    if w.size == 0:
        raise KeyError(f"{checkpoint_path} has no 'plane_weights' (not an A3 checkpoint?)")
    return pd.DataFrame(w, index=list(st.get("targets", TARGETS)), columns=order)


def compute_fingerprint_plane(model, device: str, k_total: int, size: int) -> np.ndarray:
    """True-fp32 (TF32 disabled), no-autocast output on [1,k_total,3,size,size], all planes on."""
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        g = torch.Generator().manual_seed(1234)
        x = torch.randn(1, k_total, 3, size, size, generator=g).to(device)
        m = torch.ones(1, N_PLANES, dtype=torch.bool, device=device)
        with torch.no_grad():
            return torch.sigmoid(model(x, plane_mask=m).float()).cpu().numpy().ravel()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def check_fingerprint_plane(member: MemberPlane, device: str, tol: float = FINGERPRINT_TOL) -> bool:
    """Advisory self-check (see infer.py: a cross-arch mismatch must NOT abort the run)."""
    ref = np.asarray(member.fingerprint) if member.fingerprint is not None else np.zeros(0)
    if ref.size == 0:
        print(f"[fingerprint] {member.name}: none stored; skipping check")
        return True
    got = compute_fingerprint_plane(member.model, device, member.k_total, member.size)
    if got.shape != ref.shape:
        print(f"[fingerprint] {member.name}: shape {got.shape} vs stored {ref.shape} -> SKIP")
        return True
    ok = bool(np.allclose(got, ref, atol=tol))
    print(f"[fingerprint] {member.name}: max|diff|={float(np.max(np.abs(got - ref))):.2e} "
          f"tol={tol:.0e} -> {'OK' if ok else 'MISMATCH'}")
    return ok


# --------------------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------------------
def _root_for_size(input_root: Union[str, Mapping[int, str]], size: int) -> str:
    """One root (a raw DICOM mount decodes at any size) or a {size: root} map of caches."""
    if isinstance(input_root, Mapping):
        if size not in input_root:
            raise KeyError(f"no input root configured for size {size}: {dict(input_root)}")
        return input_root[size]
    return input_root


@torch.no_grad()
def predict_studies_plane(
    input_root: Union[str, Mapping[int, str]],
    split: str,
    members: Sequence[MemberPlane],
    batch: int = 8,
    workers: int = 8,
    device: str = "cuda",
    study_ids: Optional[Sequence[str]] = None,
    mode: str = "prob",
    weights: Optional[Sequence[float]] = None,
    series_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Ensemble prediction -> DataFrame [ID_COL + 12 columns], one row per study.

    Members are grouped by `(size, k_per_plane, mean, std)`; each group decodes its slices
    ONCE and every member of the group runs on the decoded batch. `mode="prob"` averages
    sigmoid probabilities; `mode="rank"` averages per-column percentile ranks (use when the
    members are heterogeneous — AUC is rank-based).
    """
    if not members:
        raise ValueError("no members")
    if mode not in ("prob", "rank"):
        raise ValueError(f"mode must be 'prob' or 'rank', got {mode!r}")
    w = np.ones(len(members)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(w) != len(members):
        raise ValueError("weights must match members")

    groups: Dict[Tuple, List[int]] = {}
    for i, m in enumerate(members):
        groups.setdefault(m.signature, []).append(i)
    print(f"[predict] {len(members)} plane member(s) in {len(groups)} decode group(s): "
          f"{[(sig[0], sig[1], len(ix)) for sig, ix in groups.items()]} (size, k_per_plane, n)")

    per_member: Dict[int, pd.DataFrame] = {}
    ordered_ids: Optional[List[str]] = None
    for sig, idxs in groups.items():
        size, kpp = sig[0], sig[1]
        m0 = members[idxs[0]]
        root = _root_for_size(input_root, size)
        index = build_plane_index(root, split=split, series_csv=series_csv, cache_path=None)
        ids_wanted = list(index.keys()) if study_ids is None else \
            [str(s) for s in study_ids if str(s) in index]
        ds = StudyPlaneBagDataset(ids_wanted, index, labels=None, k_per_plane=kpp, train=False,
                                  size=size, mean=m0.mean, std=m0.std)
        loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                            pin_memory=str(device).startswith("cuda"))
        print(f"[predict] group size={size} k_per_plane={kpp}: {len(ds)} studies from {root}")
        ids: List[str] = []
        chunks: Dict[int, List[np.ndarray]] = {i: [] for i in idxs}
        for x, _y, mask, sid in loader:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            for i in idxs:
                # fp32 recompute on a dead engine or a non-finite fp16 output (see
                # _forward_with_guard); logged once, never fatal.
                logits = _forward_with_guard(members[i].model, x, device, plane_mask=mask)
                chunks[i].append(torch.sigmoid(logits.float()).cpu().numpy())
            ids.extend(list(sid))
        for i in idxs:
            arr = np.concatenate(chunks[i], axis=0) if chunks[i] else \
                np.zeros((0, len(TARGETS)), np.float32)
            per_member[i] = pd.DataFrame(arr, columns=TARGETS, index=[str(s) for s in ids])
        if ordered_ids is None:
            ordered_ids = [str(s) for s in ids]

    # Intersect: only studies every group produced (a partial local cache may drop some).
    common = [s for s in (ordered_ids or []) if all(s in df.index for df in per_member.values())]
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


# --------------------------------------------------------------------------------------
# Mixed-arm combination (PLAN §9: per-column percentile-rank average)
# --------------------------------------------------------------------------------------
def rank_average_frames(
    frames: Sequence[pd.DataFrame],
    weights: Optional[Sequence[float]] = None,
    per_column_weights: Optional[Mapping[str, Sequence[float]]] = None,
) -> pd.DataFrame:
    """Per-column percentile-rank average of prediction frames from DIFFERENT arms.

    Each frame is `[ID_COL + 12 TARGETS]` as produced by `infer.predict_studies`,
    `infer_hr.predict_studies_hr` or `predict_studies_plane`. Frames are inner-joined on
    `ID_COL` (order taken from the first), each column is converted to percentile ranks
    WITHIN its frame, then averaged. `per_column_weights` optionally overrides `weights` for
    named findings — e.g. give the plane arm more weight on MCL / the OA columns:

        rank_average_frames([base, plane], weights=[1, 1],
                            per_column_weights={"MCL": [0.4, 1.6], "Lateral OA": [0.4, 1.6]})
    """
    if not frames:
        raise ValueError("no frames")
    w = np.ones(len(frames)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(w) != len(frames):
        raise ValueError("weights must match frames")
    ids = [str(s) for s in frames[0][ID_COL].astype(str)]
    for f in frames[1:]:
        keep = set(f[ID_COL].astype(str))
        ids = [s for s in ids if s in keep]
    ranked = []
    for f in frames:
        g = f.copy()
        g[ID_COL] = g[ID_COL].astype(str)
        g = g.set_index(ID_COL).loc[ids][list(TARGETS)]
        ranked.append(g.rank(pct=True))
    out = pd.DataFrame(0.0, index=ids, columns=list(TARGETS))
    for t in TARGETS:
        cw = np.asarray(per_column_weights[t], dtype=np.float64) \
            if per_column_weights and t in per_column_weights else w
        if len(cw) != len(frames):
            raise ValueError(f"per_column_weights[{t!r}] must match frames")
        acc = np.zeros(len(ids), dtype=np.float64)
        for r, cwi in zip(ranked, cw):
            acc += cwi * r[t].values
        out[t] = acc / max(1e-12, float(cw.sum()))
    out = out.reset_index().rename(columns={"index": ID_COL})
    return out[[ID_COL] + list(TARGETS)]


# --------------------------------------------------------------------------------------
# Submission flow
# --------------------------------------------------------------------------------------
def run_submission_plane(
    input_root: Optional[Union[str, Mapping[int, str]]] = None,
    checkpoint_paths: Sequence[str] = (),
    output_path: str = "submission.csv",
    batch: int = 8,
    workers: int = 8,
    device: str = "cuda",
    mode: str = "prob",
    weights: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Survival flow (0.5 file first -> discover mount -> predict -> rank/reindex), A3 members.

    Mirrors `infer.run_submission` / `infer_hr.run_submission_hr` step for step; only the
    member loading and the prediction call are plane-aware.
    """
    if input_root is None:
        input_root = find_data_root("/kaggle/input")
        if input_root is None:
            raise FileNotFoundError("could not locate data mount (test.csv + test_series/)")
    csv_root = next(iter(input_root.values())) if isinstance(input_root, Mapping) else input_root
    print(f"[mount] input_root={input_root}")
    test_df = pd.read_csv(os.path.join(csv_root, "test.csv"))
    print(f"[self-size] test.csv rows = {len(test_df)}")

    # 1) valid 0.5 file FIRST
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
    try:
        members = load_members_plane(checkpoint_paths, device=device)
    except Exception as e:
        if str(device).startswith("cuda"):
            print(f"[device] CUDA load failed ({type(e).__name__}: {e}); CPU fallback")
            device = "cpu"
            members = load_members_plane(checkpoint_paths, device=device)
        else:
            raise
    for m in members:
        if not check_fingerprint_plane(m, device):
            print(f"[fingerprint] {m.name} mismatch -> ADVISORY (cross-arch drift expected); "
                  f"proceeding with REAL predictions")

    # 3) predict, self-sized to the hidden test's study list
    preds = predict_studies_plane(input_root, "test", members, batch=batch, workers=workers,
                                  device=device, study_ids=test_df[ID_COL].astype(str).tolist(),
                                  mode=mode, weights=weights)

    # 4) rank + reindex + fillna(0.5) + write
    sub = write_submission(preds, test_df, output_path)
    print(f"[submission] wrote {output_path} rows={len(sub)} cols={list(sub.columns)}")
    return sub


__all__ = [
    "MemberPlane",
    "load_members_plane",
    "predict_studies_plane",
    "run_submission_plane",
    "compute_fingerprint_plane",
    "check_fingerprint_plane",
    "rank_average_frames",
    "plane_weights_of",
]
