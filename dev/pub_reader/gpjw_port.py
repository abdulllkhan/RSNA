"""Verbatim port of goodpjw2008's public 2.5D ConvNeXt-T knee reader (inference path only).

Source: Kaggle dataset goodpjw2008/rsna-knee-2-5d-convnext-reader, version 1 (2026-10-05),
licence Apache-2.0, files preprocess.py / knee.py / infer.py (sha256 in reports/pub_reader_validation.md).
Copyright (c) goodpjw2008; reproduced here under the Apache License 2.0 for local validation.

Why a port and not an import: the downloaded directory is untrusted data and is never put on
sys.path. Every function/class below is copied unchanged; `dev/pub_reader/verify_port.py` proves it
by comparing ASTs with the downloaded originals (parsing only, nothing downloaded is executed).
Deliberate differences (all outside the copied bodies):
  * `load_models` loads with `torch.load(..., weights_only=True)` (the checkpoints were audited:
    only OrderedDict / FloatStorage / _rebuild_tensor_v2 are referenced) instead of weights_only=False.
  * `build_study_table_fallback` = the variant from the author's 0.954 notebook (nb
    goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954 cell 13): a missing fat-sat or
    non-fat-sat slot is filled with its counterpart's candidates. NOT used by the primary run.
  * the training-only pieces (StudyDataset, train loop, label building) are not ported.
"""
import math
import os

import cv2
import numpy as np
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- preprocess.py
MM_PER_PX = 0.4
SIZE = 384  # 153.6 mm field of view
MAX_SLICES = 64

# plane -> (normal axis, slice-order sign, (col axis, sign), (row axis, sign)) in LPS coordinates
# Sagittal: slices right->left, image columns anterior->posterior, rows superior->inferior
# Coronal:  slices anterior->posterior, columns patient-right->left, rows superior->inferior
# Axial:    slices superior->inferior, columns patient-right->left, rows anterior->posterior
CANON = {
    0: (+1, (1, +1), (2, -1)),
    1: (+1, (0, +1), (2, -1)),
    2: (-1, (0, +1), (1, +1)),
}
PLANES = ["Sagittal", "Coronal", "Axial"]


def _read(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    slope, inter = ds.get("RescaleSlope"), ds.get("RescaleIntercept")
    if slope is not None and inter is not None:
        arr = arr * float(slope) + float(inter)
    iop = [float(v) for v in ds.ImageOrientationPatient] if "ImageOrientationPatient" in ds else None
    ipp = [float(v) for v in ds.ImagePositionPatient] if "ImagePositionPatient" in ds else None
    ps = [float(v) for v in ds.PixelSpacing] if "PixelSpacing" in ds else None
    inst = float(ds.get("InstanceNumber") or 0)
    frames = list(arr) if arr.ndim == 3 else [arr]
    return [(f, iop, ipp, ps, inst + i * 1e-3) for i, f in enumerate(frames)]


def _canon_slice(img, iop, ps, plane):
    """Reorient one slice to the canonical in-plane orientation; returns (img, (row_mm, col_mm))."""
    ps = ps or [MM_PER_PX * img.shape[0] / SIZE] * 2  # no spacing: assume the image spans the crop
    if iop is None:
        return img, ps
    r, c = np.array(iop[:3]), np.array(iop[3:])
    _, (cax, csgn), (rax, rsgn) = CANON[plane]
    if abs(r[cax]) < abs(r[rax]):  # rows and columns are swapped relative to canonical
        img, r, c, ps = img.T, c, r, ps[::-1]
    if r[cax] * csgn < 0:
        img = img[:, ::-1]
    if c[rax] * rsgn < 0:
        img = img[::-1]
    return img, ps


def _resample(img, ps):
    """Centre crop to SIZE*MM_PER_PX mm, resample to MM_PER_PX, zero-pad to SIZE x SIZE."""
    out = np.zeros((SIZE, SIZE), np.float32)
    h, w = img.shape
    ch = min(h, int(round(SIZE * MM_PER_PX / ps[0])))
    cw = min(w, int(round(SIZE * MM_PER_PX / ps[1])))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
    nh = max(1, min(SIZE, int(round(ch * ps[0] / MM_PER_PX))))
    nw = max(1, min(SIZE, int(round(cw * ps[1] / MM_PER_PX))))
    interp = cv2.INTER_AREA if nh < ch else cv2.INTER_LINEAR
    res = cv2.resize(crop, (nw, nh), interpolation=interp)
    oy, ox = (SIZE - nh) // 2, (SIZE - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = res
    return out


def load_series(sdir, plane_name=None):
    """Returns (uint8 volume [n, SIZE, SIZE], info dict). Unreadable slices are skipped."""
    slices, bad = [], 0
    for fn in os.listdir(sdir):
        try:
            slices.extend(_read(os.path.join(sdir, fn)))
        except Exception:
            bad += 1
    info = {"n_raw": len(slices), "n_bad": bad}
    if not slices:
        return np.zeros((1, SIZE, SIZE), np.uint8), info

    iop = next((s[1] for s in slices if s[1] is not None), None)
    if iop is not None:
        normal = np.cross(iop[:3], iop[3:])
        plane = int(np.abs(normal).argmax())
    else:
        normal, plane = None, PLANES.index(plane_name) if plane_name in PLANES else 0
    info["plane"] = PLANES[plane]

    if normal is not None and all(s[2] is not None for s in slices):
        sgn = CANON[plane][0] * np.sign(normal[plane])
        slices.sort(key=lambda s: float(np.dot(s[2], normal)) * sgn)
        pos = [float(np.dot(s[2], normal)) for s in slices]
        info["slice_gap"] = float(np.median(np.abs(np.diff(pos)))) if len(pos) > 1 else 0.0
    else:
        slices.sort(key=lambda s: s[4])
    if len(slices) > MAX_SLICES:
        keep = np.linspace(0, len(slices) - 1, MAX_SLICES).round().astype(int)
        slices = [slices[i] for i in keep]

    vol = np.stack([_resample(*_canon_slice(s[0], s[1], s[3], plane)) for s in slices])
    lo, hi = np.percentile(vol[:, ::4, ::4], [1, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    info["n"] = len(vol)
    return (vol * 255).round().astype(np.uint8), info


# ----------------------------------------------------------------------------- knee.py
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# slot = plane index + 3 * (1 - fat_suppressed): fat-suppressed planes first
N_SLOTS = 6
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def slot_of(plane, fat_sup):
    return PLANES.index(plane) + 3 * (1 - int(fat_sup))


def build_study_table(series_df, n_slices):
    """series_df: StudyInstanceUID, SeriesInstanceUID, Fat_Suppression, Anatomical_Plane.
    n_slices: dict series -> cached slice count. Returns {study: [[series ids of slot 0], ...]}
    with each slot's candidates sorted by slice count (largest first)."""
    table = {}
    for st, sr, fs, pl in zip(series_df.StudyInstanceUID, series_df.SeriesInstanceUID,
                              series_df.Fat_Suppression, series_df.Anatomical_Plane):
        if n_slices.get(sr, 0) < 3:
            continue
        table.setdefault(st, [[] for _ in range(N_SLOTS)])[slot_of(pl, fs)].append(sr)
    for slots in table.values():
        for cands in slots:
            cands.sort(key=lambda s: -n_slices[s])
    return table


def build_study_table_fallback(series_df, n_slices):
    """0.954-notebook variant (cell 13): as build_study_table, then a missing fat-sat / non-fat-sat
    slot of a plane is filled with its counterpart's candidates."""
    table = {}
    for st, sr, fs, pl in zip(series_df.StudyInstanceUID, series_df.SeriesInstanceUID,
                              series_df.Fat_Suppression, series_df.Anatomical_Plane):
        if n_slices.get(sr, 0) < 3:
            continue
        table.setdefault(st, [[] for _ in range(N_SLOTS)])[slot_of(pl, fs)].append(sr)
    for slots in table.values():
        for cands in slots:
            cands.sort(key=lambda s: -n_slices[s])
        # Intelligent anatomical plane fallback: if fat-sat is missing, use non-fat-sat (and vice-versa)
        for p in range(3):
            fs_s, nonfs_s = p, p + 3
            if not slots[fs_s] and slots[nonfs_s]:
                slots[fs_s] = list(slots[nonfs_s])
            elif not slots[nonfs_s] and slots[fs_s]:
                slots[nonfs_s] = list(slots[fs_s])
    return table


def window_centres(n, k, train, lo=0.04, hi=0.96):
    """k slice indices (centres of 3-slice windows) spread over the series."""
    a, b = max(1.0, lo * (n - 1)), min(n - 2.0, hi * (n - 1))
    if b < a:
        a = b = (n - 1) / 2
    if train:
        edges = np.linspace(a, b, k + 1)
        c = edges[:-1] + np.random.rand(k) * (edges[1:] - edges[:-1])
    else:
        c = np.linspace(a, b, k)
    return np.clip(np.round(c).astype(int), 1, max(1, n - 2))


def make_views(x, res, train, group, fov=0.92):
    """x: uint8 [N, 3, H, W] with N = n_series * group. Crops a centred `fov` fraction of the cached
    field of view and resizes to res; in training adds rotation/scale/shift and gain/gamma jitter
    that is shared within a series."""
    n = x.shape[0]
    ns = n // group
    dev = x.device
    x = x.float().div_(255.0)
    if train:
        ang = (torch.rand(ns, device=dev) - 0.5) * 2 * math.radians(12)
        sc = fov * (1 + (torch.rand(ns, device=dev) - 0.5) * 0.24)
        tx = (torch.rand(ns, device=dev) - 0.5) * 0.12
        ty = (torch.rand(ns, device=dev) - 0.5) * 0.12
        gamma = torch.exp((torch.rand(ns, device=dev) - 0.5) * 0.5)
        gain = 1 + (torch.rand(ns, device=dev) - 0.5) * 0.2
    else:
        ang = torch.zeros(ns, device=dev)
        sc = torch.full((ns,), fov, device=dev)
        tx = ty = torch.zeros(ns, device=dev)
        gamma = gain = None
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    theta = theta.repeat_interleave(group, 0)
    grid = F.affine_grid(theta, (n, 3, res, res), align_corners=False)
    if x.shape[-1] > 1.5 * res:  # avoid aliasing when shrinking a lot
        x = F.avg_pool2d(x, 2) if x.shape[-1] >= 2 * res else F.interpolate(x, scale_factor=0.75, mode="bilinear", antialias=True)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if train:
        g = gamma.repeat_interleave(group).view(n, 1, 1, 1)
        x = x.clamp_min(1e-4).pow(g) * gain.repeat_interleave(group).view(n, 1, 1, 1)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


class KneeNet(nn.Module):
    """2D backbone on 3-slice windows -> tokens tagged with series slot and slice position ->
    small transformer over all windows of the study -> one attention pooling per finding."""

    def __init__(self, backbone="convnext_tiny", pretrained=True, dim=384, depth=2, drop=0.2, drop_path=0.1):
        super().__init__()
        kw = {"drop_path_rate": drop_path} if drop_path else {}
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(self.enc.num_features, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Embedding(N_SLOTS, dim)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False) if depth else None
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, len(LABELS)))
        self.cls_w = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(len(LABELS)))

    def head(self, feat, mask, pos):
        """feat [B, S, K, D], mask [B, S] bool, pos [B, S, K] in 0..1 -> logits [B, 12]."""
        b, s, k, _ = feat.shape
        slot = torch.arange(s, device=feat.device).view(1, s, 1).expand(b, s, k)
        h = self.proj(feat) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))
        h = h.view(b, s * k, -1)
        pad = ~mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
        pad = pad & ~pad.all(1, keepdim=True)  # a study with no series at all attends to everything
        if self.tf is not None:
            h = self.tf(h, src_key_padding_mask=pad)
        a = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)  # [B, T, C]
        pooled = torch.einsum("btc,btd->bcd", a.to(h.dtype), h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b

    def forward(self, x, mask, pos, res=256, train=False):
        """x uint8 [B, S, K, 3, H, W]."""
        b, s, k = x.shape[:3]
        keep = mask.view(-1)  # run the backbone only on series that exist
        xv = x.view(b * s, k, *x.shape[3:])[keep].flatten(0, 1)
        f = self.enc(make_views(xv, res, train, k).contiguous(memory_format=torch.channels_last))
        feat = f.new_zeros(b * s, k, f.shape[-1])
        feat[keep] = f.view(-1, k, f.shape[-1])
        return self.head(feat.view(b, s, k, -1), mask, pos)


# ----------------------------------------------------------------------------- infer.py
class DicomStudyDataset(torch.utils.data.Dataset):
    """Loads every selected series of a study once; windows are cut per model setting later."""

    def __init__(self, studies, table, root, k):
        self.studies, self.table, self.root, self.k = list(studies), table, root, k

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        slots = self.table.get(st, [[] for _ in range(N_SLOTS)])
        x = np.zeros((N_SLOTS, self.k, 3, SIZE, SIZE), np.uint8)
        mask, pos = np.zeros(N_SLOTS, bool), np.zeros((N_SLOTS, self.k), np.float32)
        for s, cands in enumerate(slots):
            if not cands:
                continue
            try:
                vol, _ = load_series(f"{self.root}/{st}/{cands[0]}")
            except Exception:
                continue
            n = len(vol)
            if n < 3:
                continue
            c = window_centres(n, self.k, False)
            x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])
            mask[s], pos[s] = True, c / max(n - 1, 1)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos)


def load_models(ckpts, device="cuda"):
    """infer.py load_models, but with weights_only=True (safe unpickler) and the args returned."""
    models = []
    for path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        # the pretrained tag after the dot is irrelevant here and unknown to older timm versions
        m = KneeNet(ck["args"]["backbone"].split(".")[0], pretrained=False)
        m.load_state_dict(ck["model"])
        models.append((m.to(device).to(memory_format=torch.channels_last).eval(), ck["args"]["res"], ck["args"]))
    return models


def series_file_counts(se, root):
    """infer.py run(): n_files[sr] = min(#files in the series dir, MAX_SLICES)."""
    n_files = {}
    for st, sr in zip(se.StudyInstanceUID, se.SeriesInstanceUID):
        d = f"{root}/{st}/{sr}"
        n_files[sr] = min(len(os.listdir(d)), MAX_SLICES) if os.path.isdir(d) else 0
    return n_files


def study_windows(vol_of, st, slots, k):
    """infer.py DicomStudyDataset.__getitem__ body, with the series decode delegated to `vol_of`
    (a memoising wrapper around load_series, so two variants of one study share one decode).
    vol_of(path) returns the uint8 volume or raises like load_series does."""
    x = np.zeros((N_SLOTS, k, 3, SIZE, SIZE), np.uint8)
    mask, pos = np.zeros(N_SLOTS, bool), np.zeros((N_SLOTS, k), np.float32)
    for s, cands in enumerate(slots):
        if not cands:
            continue
        try:
            vol = vol_of(cands[0])
        except Exception:
            continue
        n = len(vol)
        if n < 3:
            continue
        c = window_centres(n, k, False)
        x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])
        mask[s], pos[s] = True, c / max(n - 1, 1)
    return x, mask, pos
