"""Size-aware inference for a MIXED-resolution ensemble (224 baseline + Wave-3 384/512).

`infer.py` decodes every slice at 224 (`dataset.StudyBagDataset`), which is exactly right for
the baseline checkpoints and exactly wrong for a 384 one. This module is the size-aware twin;
`infer.py` is left untouched (running trainings import it) and stays the 224-only path.

What is different from `infer.py`:
  * `load_members_hr` reads the input resolution from the checkpoint itself
    (`ckpt["args"]["size"]`, written by `train_hr.py`; absent -> 224, so every baseline
    checkpoint keeps working);
  * `predict_studies_hr` groups members by their decode signature `(size, k, mean, std)` and
    builds ONE loader per group — the same "decode once, run every member on the batch" trick
    `infer.py` uses, generalised: a 5-fold 384 ensemble still decodes each slice once, and a
    mixed 224+384 ensemble decodes each slice exactly twice (once per resolution) instead of
    once per member;
  * decoding goes through `dataset_hr.read_slice_hr` -> `decode.read_dicom_slice(path, size)`
    for `.dcm` and a cached JPEG otherwise, so the Kaggle DICOM mount and a local
    `data/cache<size>` tree are interchangeable;
  * `mode="rank"` averages per-column percentile ranks instead of probabilities — the
    PLAN §9 combiner for heterogeneous arms (AUC only cares about within-column order).

Kept identical to `infer.py`: the capability-aware autocast rule (bf16 on sm_80+, fp16 on
sm_70-75, fp32 elsewhere, with an fp32 recompute if a forward fails or returns non-finite
values), `pretrained=False` weight loading, the fingerprint self-check (now size-aware), and
the submission-writing helpers, which are imported from `infer.py` rather than duplicated.

Local check against the visible test DICOM (or its decoded cache):

    from src.rsna.infer_hr import load_members_hr, predict_studies_hr
    ms = load_members_hr(["models/effv2s_k24_384_fold0.pt"], device="cpu")
    preds = predict_studies_hr("/tmp/cache384", "test", ms, batch=2, workers=2, device="cpu")
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
from .dataset_hr import StudyBagDatasetHR, build_hr_index
from .infer import (  # reused verbatim -- no duplication, no edit to infer.py
    FINGERPRINT_TOL,
    find_data_root,
    write_benchmark_submission,
    write_submission,
)
from .model import ModelConfig, build_model

DEFAULT_SIZE = 224  # checkpoints without args['size'] are the 224 baseline


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
class MemberHR:
    """One loaded checkpoint + the preprocessing it was trained with (incl. `size`)."""

    def __init__(self, model, mean, std, k: int, size: int, fingerprint, name: str = ""):
        self.model = model
        self.mean = mean
        self.std = std
        self.k = int(k)
        self.size = int(size)
        self.fingerprint = fingerprint
        self.name = name

    @property
    def signature(self) -> Tuple:
        """Members sharing this decode signature can share one DataLoader."""
        return (self.size, self.k,
                tuple(round(float(v), 6) for v in np.asarray(self.mean).ravel()),
                tuple(round(float(v), 6) for v in np.asarray(self.std).ravel()))

    def __repr__(self) -> str:
        return f"MemberHR({self.name or 'ckpt'}, size={self.size}, k={self.k})"


def load_members_hr(checkpoint_paths: Sequence[str], device: str = "cuda") -> List[MemberHR]:
    """Load checkpoints with `pretrained=False`, taking k/size/mean/std from each ckpt."""
    members: List[MemberHR] = []
    for cp in checkpoint_paths:
        st = torch.load(cp, map_location=device, weights_only=False)
        cfg = ModelConfig(**{k: v for k, v in st["cfg"].items()
                             if k in ModelConfig.__dataclass_fields__})
        cfg.pretrained = False  # NO network at inference
        # Unconditional, exactly like `infer.load_members` -- NOT gated on cuda. This builds a
        # `model.StudyModel`, whose forward does `x.to(memory_format=channels_last)` with no
        # device test, so a cuda-gated loader leaves NCHW weights meeting NHWC activations on
        # the CPU-fallback path: a reorder on every conv, every batch. (The plane loader is
        # deliberately left cuda-gated: `model_plane.StudyPlaneModel.forward` gates its own
        # input conversion on `flat.is_cuda`, so there both sides are already NCHW on CPU.)
        model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
        model.load_state_dict(st["model"])
        model.eval()
        a = st.get("args", {}) or {}
        m = MemberHR(model, st["mean"], st["std"], a.get("k", 24),
                     int(a.get("size", DEFAULT_SIZE)), np.asarray(st.get("fingerprint")),
                     name=os.path.basename(cp))
        print(f"[member] {m.name}: size={m.size} k={m.k} backbone={cfg.backbone}")
        members.append(m)
    return members


def compute_fingerprint_hr(model, device: str, k: int, size: int) -> np.ndarray:
    """True-fp32 (TF32 disabled), no-autocast output on a fixed synthetic [1,k,3,size,size]."""
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        g = torch.Generator().manual_seed(1234)
        x = torch.randn(1, k, 3, size, size, generator=g).to(device)
        with torch.no_grad():
            return torch.sigmoid(model(x).float()).cpu().numpy().ravel()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def check_fingerprint_hr(member: MemberHR, device: str, tol: float = FINGERPRINT_TOL) -> bool:
    """Advisory self-check (see infer.py: a cross-arch mismatch must NOT abort the run)."""
    if member.fingerprint is None or np.asarray(member.fingerprint).size == 0:
        print(f"[fingerprint] {member.name}: none stored; skipping check")
        return True
    got = compute_fingerprint_hr(member.model, device, member.k, member.size)
    ref = np.asarray(member.fingerprint)
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
    """`input_root` is either one root (raw DICOM mount: any size decodes from it) or a
    {size: root} map (local caches: 224 -> processed_jpeg224, 384 -> data/cache384, ...)."""
    if isinstance(input_root, Mapping):
        if size not in input_root:
            raise KeyError(f"no input root configured for size {size}: {dict(input_root)}")
        return input_root[size]
    return input_root


@torch.no_grad()
def predict_studies_hr(
    input_root: Union[str, Mapping[int, str]],
    split: str,
    members: Sequence[MemberHR],
    batch: int = 8,
    workers: int = 8,
    device: str = "cuda",
    study_ids: Optional[Sequence[str]] = None,
    mode: str = "prob",
    weights: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Ensemble prediction -> DataFrame [ID_COL + 12 columns], one row per study.

    Members are grouped by `(size, k, mean, std)`; each group decodes its slices ONCE and
    every member of the group runs on the decoded batch. `mode="prob"` averages sigmoid
    probabilities (the baseline's combiner); `mode="rank"` averages per-column percentile
    ranks (use for heterogeneous arms — AUC is rank-based).
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
    print(f"[predict] {len(members)} member(s) in {len(groups)} decode group(s): "
          f"{[ (sig[0], sig[1], len(ix)) for sig, ix in groups.items() ]} (size, k, n)")

    per_member: Dict[int, pd.DataFrame] = {}
    ordered_ids: Optional[List[str]] = None
    for sig, idxs in groups.items():
        size, k = sig[0], sig[1]
        m0 = members[idxs[0]]
        root = _root_for_size(input_root, size)
        index = build_hr_index(root, split=split, cache_path=None)
        ids_wanted = list(index.keys()) if study_ids is None else \
            [str(s) for s in study_ids if str(s) in index]
        ds = StudyBagDatasetHR(ids_wanted, index, labels=None, k=k, train=False, size=size,
                               mean=m0.mean, std=m0.std)
        loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                            pin_memory=str(device).startswith("cuda"))
        print(f"[predict] group size={size} k={k}: {len(ds)} studies from {root}")
        ids: List[str] = []
        chunks: Dict[int, List[np.ndarray]] = {i: [] for i in idxs}
        for x, _y, sid in loader:
            x = x.to(device, non_blocking=True)
            for i in idxs:
                # fp32 recompute on a dead engine or a non-finite fp16 output (see
                # _forward_with_guard); logged once, never fatal.
                logits = _forward_with_guard(members[i].model, x, device)
                chunks[i].append(torch.sigmoid(logits.float()).cpu().numpy())
            ids.extend(list(sid))
        for i in idxs:
            arr = np.concatenate(chunks[i], axis=0) if chunks[i] else \
                np.zeros((0, len(TARGETS)), np.float32)
            per_member[i] = pd.DataFrame(arr, columns=TARGETS, index=[str(s) for s in ids])
        if ordered_ids is None:
            ordered_ids = [str(s) for s in ids]

    # Intersect: only studies every group produced (all groups see the same study list, but a
    # partial local cache may drop some -> stay strictly aligned rather than guess).
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


def run_submission_hr(
    input_root: Optional[Union[str, Mapping[int, str]]] = None,
    checkpoint_paths: Sequence[str] = (),
    output_path: str = "submission.csv",
    batch: int = 8,
    workers: int = 8,
    device: str = "cuda",
    mode: str = "prob",
    weights: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Survival flow (0.5 file first -> discover mount -> predict -> rank/reindex), size-aware.

    Mirrors `infer.run_submission` step for step; only the member loading and the prediction
    call are the size-aware ones. `input_root=None` discovers the Kaggle mount, where every
    member decodes from the same raw DICOM tree at its own resolution.
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
        members = load_members_hr(checkpoint_paths, device=device)
    except Exception as e:
        if str(device).startswith("cuda"):
            print(f"[device] CUDA load failed ({type(e).__name__}: {e}); CPU fallback")
            device = "cpu"
            members = load_members_hr(checkpoint_paths, device=device)
        else:
            raise
    for m in members:
        if not check_fingerprint_hr(m, device):
            print(f"[fingerprint] {m.name} mismatch -> ADVISORY (cross-arch drift expected); "
                  f"proceeding with REAL predictions")

    # 3) predict, self-sized to the hidden test's study list
    preds = predict_studies_hr(input_root, "test", members, batch=batch, workers=workers,
                               device=device, study_ids=test_df[ID_COL].astype(str).tolist(),
                               mode=mode, weights=weights)

    # 4) rank + reindex + fillna(0.5) + write
    sub = write_submission(preds, test_df, output_path)
    print(f"[submission] wrote {output_path} rows={len(sub)} cols={list(sub.columns)}")
    return sub


__all__ = [
    "MemberHR",
    "load_members_hr",
    "predict_studies_hr",
    "run_submission_hr",
    "compute_fingerprint_hr",
    "check_fingerprint_hr",
]
