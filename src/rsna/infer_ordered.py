"""Inference for the anatomically-ORDERED slot-sampler arm — size-aware AND order-aware.

`infer_hr.predict_studies_hr` samples each study's K slots evenly over the series'
**lexicographic** file list. That is right for the 224/384 arms and **wrong** for an ordered
checkpoint, which was trained on slots evenly spaced in ANATOMICAL rank: the model would still
produce plausible numbers, from a slice distribution it never saw (anatomically an arbitrary
subset — lex order == anatomical order in 0/400 train and 0/15 test series). So the arm needs
its own predictor, and its checkpoints must be routed here by their metadata
(`args["order"]` + `args["coverage"]`, or `args["mode"] == "ordered"`) — not by filename, and
not by `size`, which a 384 ordered checkpoint shares with the plain 384 HR arm.

Everything that is not the slot choice is `infer_hr`'s code, imported rather than copied: the
capability-aware autocast (bf16 sm_80+ / fp16 T4-V100 / fp32 else) with the per-batch fp32
recompute guard, the `pretrained=False` weight load, the size-aware fingerprint check, and
`infer`'s submission writers. `mode="rank"` per-column percentile-rank averaging is kept for
combining members across arms (PLAN §9).

SLICE ORDER AT INFERENCE. The Kaggle mount IS DICOM, so the order comes from the headers
themselves (`order.SliceOrderResolver` -> `pydicom.dcmread(stop_before_pixels=True,
specific_tags=[InstanceNumber, ImagePositionPatient, ImageOrientationPatient])`), ~0.1 ms per
slice and cached per series — the whole 557-slice visible-test mount audits in 0.5 s including
interpreter start. No parquet is needed or possible for the hidden test; do NOT ship
`data/series_order.parquet`. Predicting from a LOCAL JPEG cache instead? Pass
`order_parquet=` (the train map) or `dicom_root=` (a raw mirror), or the slots silently
degrade to lexicographic.

COST. Identical to the HR arm: K slots, one decode per slot, one backbone forward per slot.
The only addition is 3 header tags per slice, once per series.

Local check against the visible test DICOM:

    .venv/bin/python -m src.rsna.infer_ordered --input-root /tmp/dicom_mount --split test \
        --ckpt models/effv2s_k24_384_ord_fold0.pt --device cpu --batch 1 --workers 2 \
        --out /tmp/sub_ord.csv
"""
from __future__ import annotations

import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import TARGETS, ID_COL
from .dataset_hr import build_hr_index
from .dataset_ordered import (
    DEFAULT_CENTER_FRAC,
    DEFAULT_COVERAGE,
    MODE_TAG,
    StudyBagDatasetOrdered,
)
from .infer import (  # reused verbatim -- no duplication, no edit to infer.py
    find_data_root,
    write_benchmark_submission,
    write_submission,
)
from .infer_hr import (  # reused verbatim -- no duplication, no edit to infer_hr.py
    DEFAULT_SIZE,
    _forward_with_guard,
    check_fingerprint_hr,
    compute_fingerprint_hr,
)
from .model import ModelConfig, build_model
from .order import DEFAULT_ORDER_MODE, SliceOrderResolver


# --------------------------------------------------------------------------------------
# Arm detection
# --------------------------------------------------------------------------------------
def is_ordered_meta(args: Optional[Mapping] = None, cfg: Optional[Mapping] = None,
                    keys: Optional[Sequence[str]] = None) -> bool:
    """Is this checkpoint an ordered-sampler one? From its OWN metadata, never its filename.

    `train_ordered.py` writes `args["mode"] = "ordered"` AND `args["order"]` +
    `args["coverage"]`; no other trainer in the repo writes any of those three keys
    (`train_25d` writes `order_mode` / `order_parquet`, which are different names, and
    `mode="2.5d"`). Either signal is accepted so a hand-edited or slimmed checkpoint that
    lost `mode` still routes correctly.

    This is the predicate the multi-arm notebook's `detect_arm` must consult BEFORE its
    `size != 224 -> hr` rule (see `reports/arm_ordered.md`).
    """
    a = dict(args or {})
    if str(a.get("mode", "")).strip().lower() == MODE_TAG:
        return True
    return ("order" in a) and ("coverage" in a)


# --------------------------------------------------------------------------------------
# Members
# --------------------------------------------------------------------------------------
class MemberOrdered:
    """One loaded ordered-sampler checkpoint + the sampling it was trained with."""

    def __init__(self, model, mean, std, k: int, size: int, fingerprint, name: str = "",
                 order_mode: str = DEFAULT_ORDER_MODE, coverage: str = DEFAULT_COVERAGE,
                 center_frac: float = DEFAULT_CENTER_FRAC):
        self.model = model
        self.mean = mean
        self.std = std
        self.k = int(k)
        self.size = int(size)
        self.fingerprint = fingerprint
        self.name = name
        self.order_mode = str(order_mode)
        self.coverage = str(coverage)
        self.center_frac = float(center_frac)

    @property
    def signature(self) -> Tuple:
        """Members sharing this decode signature can share ONE DataLoader (decode once)."""
        return (self.size, self.k, self.order_mode, self.coverage,
                round(float(self.center_frac), 6),
                tuple(round(float(v), 6) for v in np.asarray(self.mean).ravel()),
                tuple(round(float(v), 6) for v in np.asarray(self.std).ravel()))

    def __repr__(self) -> str:
        return (f"MemberOrdered({self.name or 'ckpt'}, size={self.size}, k={self.k}, "
                f"order={self.order_mode}, coverage={self.coverage})")


def load_members_ordered(checkpoint_paths: Sequence[str], device: str = "cuda",
                         strict_mode: bool = False) -> List[MemberOrdered]:
    """Load ordered checkpoints with `pretrained=False`, reading size/k/order/coverage from each.

    A checkpoint without the ordered metadata belongs to a different arm; sampling it here
    would feed it slots it never trained on. Warned by default, fatal with `strict_mode=True`.
    """
    members: List[MemberOrdered] = []
    for cp in checkpoint_paths:
        st = torch.load(cp, map_location=device, weights_only=False)
        a = st.get("args", {}) or {}
        if not is_ordered_meta(a):
            msg = (f"{os.path.basename(cp)}: args has mode={a.get('mode')!r} "
                   f"order={a.get('order')!r} coverage={a.get('coverage')!r} — this is NOT an "
                   f"ordered-sampler checkpoint")
            if strict_mode:
                raise ValueError(msg)
            print(f"[member] WARNING {msg}; loading it anyway (its slots will be chosen in "
                  f"anatomical order, which it was not trained on)")
        cfg = ModelConfig(**{k: v for k, v in st["cfg"].items()
                             if k in ModelConfig.__dataclass_fields__})
        cfg.pretrained = False  # NO network at inference
        model = build_model(cfg).to(device)
        # Unconditional (matching infer.load_members / infer_hr.load_members_hr):
        # `model.StudyModel.forward` converts activations to channels_last with no device
        # test, so cuda-gating the weights would leave NCHW weights meeting NHWC activations
        # on the CPU fallback path — a reorder on every conv, every batch.
        model = model.to(memory_format=torch.channels_last)
        model.load_state_dict(st["model"])
        model.eval()
        m = MemberOrdered(model, st["mean"], st["std"], a.get("k", 24),
                          int(a.get("size", DEFAULT_SIZE)), np.asarray(st.get("fingerprint")),
                          name=os.path.basename(cp),
                          order_mode=str(a.get("order", DEFAULT_ORDER_MODE)),
                          coverage=str(a.get("coverage", DEFAULT_COVERAGE)),
                          center_frac=float(a.get("center_frac", DEFAULT_CENTER_FRAC)))
        print(f"[member] {m.name}: mode=ordered size={m.size} k={m.k} order={m.order_mode} "
              f"coverage={m.coverage} backbone={cfg.backbone}")
        members.append(m)
    return members


# `check_fingerprint_hr` / `compute_fingerprint_hr` work unchanged: the fingerprint is a
# synthetic [1,k,3,size,size] tensor, the same shape for both arms.
check_fingerprint_ordered = check_fingerprint_hr
compute_fingerprint_ordered = compute_fingerprint_hr


# --------------------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------------------
def _root_for_size(input_root: Union[str, Mapping[int, str]], size: int) -> str:
    """One root (raw DICOM mount: every size decodes from it) or a `{size: root}` map."""
    if isinstance(input_root, Mapping):
        if size not in input_root:
            raise KeyError(f"no input root configured for size {size}: {dict(input_root)}")
        return input_root[size]
    return input_root


def build_infer_order(
    split: str = "test",
    mode: str = DEFAULT_ORDER_MODE,
    order_parquet: Optional[str] = None,
    dicom_root: Optional[str] = None,
) -> SliceOrderResolver:
    """The inference-side order resolver: headers by default, parquet/mirror when given.

    On Kaggle nothing is passed: the mount is `.dcm`, so the resolver reads the three ordering
    tags straight from each series (cached per series). Locally, a JPEG cache root needs
    `order_parquet=` or `dicom_root=` or the slots would be lexicographic.
    """
    return SliceOrderResolver.from_parquet(order_parquet, dicom_root=dicom_root, split=split,
                                           mode=mode)


@torch.no_grad()
def predict_studies_ordered(
    input_root: Union[str, Mapping[int, str]],
    split: str,
    members: Sequence[MemberOrdered],
    batch: int = 4,
    workers: int = 4,
    device: str = "cuda",
    study_ids: Optional[Sequence[str]] = None,
    mode: str = "prob",
    weights: Optional[Sequence[float]] = None,
    order: Optional[SliceOrderResolver] = None,
    order_parquet: Optional[str] = None,
    dicom_root: Optional[str] = None,
) -> pd.DataFrame:
    """Ensemble prediction -> DataFrame `[ID_COL + 12 columns]`, one row per study.

    Members are grouped by `(size, k, order_mode, coverage, center_frac, mean, std)`; each
    group decodes its slices ONCE and every member of the group runs on the decoded batch —
    `infer_hr`'s decode-once trick. `mode="prob"` averages sigmoid probabilities;
    `mode="rank"` averages per-column percentile ranks (the PLAN §9 combiner for
    heterogeneous members).

    The returned frame is exactly the shape `infer_plane.rank_average_frames` consumes, so
    this arm plugs into the multi-arm rank ensemble with no adapter.
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
    print(f"[predict] {len(members)} ordered member(s) in {len(groups)} decode group(s): "
          f"{[(sig[0], sig[1], sig[2], sig[3], len(ix)) for sig, ix in groups.items()]} "
          f"(size, k, order, coverage, n)")

    per_member: Dict[int, pd.DataFrame] = {}
    ordered_ids: Optional[List[str]] = None
    for sig, idxs in groups.items():
        size, k, order_mode, coverage, center_frac = sig[0], sig[1], sig[2], sig[3], sig[4]
        m0 = members[idxs[0]]
        root = _root_for_size(input_root, size)
        index = build_hr_index(root, split=split, cache_path=None)
        ids_wanted = list(index.keys()) if study_ids is None else \
            [str(s) for s in study_ids if str(s) in index]
        # Built in THIS process so the per-series order cache is shared copy-on-write by the
        # DataLoader workers rather than rebuilt in each of them.
        res = order or build_infer_order(split=split, mode=order_mode,
                                         order_parquet=order_parquet, dicom_root=dicom_root)
        ds = StudyBagDatasetOrdered(ids_wanted, index, labels=None, k=k, train=False, size=size,
                                    mean=m0.mean, std=m0.std, order=res, coverage=coverage,
                                    center_frac=center_frac)
        loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                            pin_memory=str(device).startswith("cuda"))
        print(f"[predict] group size={size} k={k} order={order_mode} coverage={coverage}: "
              f"{len(ds)} studies from {root}")
        ids: List[str] = []
        chunks: Dict[int, List[np.ndarray]] = {i: [] for i in idxs}
        for x, _y, sid in loader:
            x = x.to(device, non_blocking=True)
            for i in idxs:
                logits = _forward_with_guard(members[i].model, x, device)
                chunks[i].append(torch.sigmoid(logits.float()).cpu().numpy())
            ids.extend(list(sid))
        print(" ", res.report())
        for i in idxs:
            arr = np.concatenate(chunks[i], axis=0) if chunks[i] else \
                np.zeros((0, len(TARGETS)), np.float32)
            per_member[i] = pd.DataFrame(arr, columns=TARGETS, index=[str(s) for s in ids])
        if ordered_ids is None:
            ordered_ids = [str(s) for s in ids]

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


def run_submission_ordered(
    input_root: Optional[Union[str, Mapping[int, str]]] = None,
    checkpoint_paths: Sequence[str] = (),
    output_path: str = "submission.csv",
    batch: int = 4,
    workers: int = 4,
    device: str = "cuda",
    mode: str = "prob",
    weights: Optional[Sequence[float]] = None,
    split: str = "test",
    order_parquet: Optional[str] = None,
    dicom_root: Optional[str] = None,
) -> pd.DataFrame:
    """Survival flow (0.5 file first -> mount -> device probe -> predict -> rank -> write).

    Mirrors `infer_hr.run_submission_hr` step for step; only the member loading and the
    prediction call are the ordered-sampler ones.
    """
    if input_root is None:
        input_root = find_data_root("/kaggle/input")
        if input_root is None:
            raise FileNotFoundError("could not locate data mount (test.csv + test_series/)")
    csv_root = next(iter(input_root.values())) if isinstance(input_root, Mapping) else input_root
    print(f"[mount] input_root={input_root}")
    test_df = pd.read_csv(os.path.join(csv_root, f"{split}.csv"))
    print(f"[self-size] {split}.csv rows = {len(test_df)}")

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
        members = load_members_ordered(checkpoint_paths, device=device)
    except Exception as e:
        if str(device).startswith("cuda"):
            print(f"[device] CUDA load failed ({type(e).__name__}: {e}); CPU fallback")
            device = "cpu"
            members = load_members_ordered(checkpoint_paths, device=device)
        else:
            raise
    for m in members:
        if not check_fingerprint_ordered(m, device):
            print(f"[fingerprint] {m.name} mismatch -> ADVISORY (cross-arch drift expected); "
                  f"proceeding with REAL predictions")

    # 3) predict, self-sized to the hidden test's study list
    preds = predict_studies_ordered(input_root, split, members, batch=batch, workers=workers,
                                    device=device,
                                    study_ids=test_df[ID_COL].astype(str).tolist(),
                                    mode=mode, weights=weights, order_parquet=order_parquet,
                                    dicom_root=dicom_root)

    # 4) rank + reindex + fillna(0.5) + write
    sub = write_submission(preds, test_df, output_path)
    print(f"[submission] wrote {output_path} rows={len(sub)} cols={list(sub.columns)}")
    return sub


# --------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Local CLI — the smoke path against `/tmp/dicom_mount` or a decoded cache."""
    import argparse

    ap = argparse.ArgumentParser(description="ordered-sampler arm inference (local check)")
    ap.add_argument("--input-root", required=True, help="DICOM mount or data/cache<size>")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", default="submission_ordered.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--member-mode", default="prob", choices=("prob", "rank"))
    ap.add_argument("--order-parquet", default=None,
                    help="only for a JPEG cache root; the DICOM mount needs nothing")
    ap.add_argument("--dicom-root", default=None)
    a = ap.parse_args(argv)

    sub = run_submission_ordered(input_root=a.input_root, checkpoint_paths=a.ckpt,
                                 output_path=a.out, batch=a.batch, workers=a.workers,
                                 device=a.device, mode=a.member_mode, split=a.split,
                                 order_parquet=a.order_parquet, dicom_root=a.dicom_root)
    print(sub.head().to_string())
    n_unique = sub[TARGETS].round(6).drop_duplicates().shape[0]
    print(f"[check] rows={len(sub)} cols={len(sub.columns)} distinct prediction rows={n_unique}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MemberOrdered",
    "load_members_ordered",
    "predict_studies_ordered",
    "run_submission_ordered",
    "build_infer_order",
    "is_ordered_meta",
    "check_fingerprint_ordered",
    "compute_fingerprint_ordered",
    "MODE_TAG",
]
