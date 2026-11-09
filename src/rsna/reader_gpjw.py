"""The `reader` arm: goodpjw2008's public 2.5D ConvNeXt-T knee reader (verbatim inference port + our arm runner).

Source: Kaggle dataset goodpjw2008/rsna-knee-2-5d-convnext-reader, version 1 (2026-10-05): preprocess.py, knee.py,
infer.py and the checkpoints cnxt_v0_fold{0,1,2}.pt (sha256 below and in reports/pub_reader_validation.md).
Copyright (c) goodpjw2008. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

Modifications (ours):
  * the block between the two "verbatim port" rulers is dev/pub_reader/gpjw_port.py byte-for-byte: every ported
    constant/function/class is AST-identical to the author's file (dev/sub13/verify_reader_module.py re-proves it
    for THIS file); `load_models` loads with weights_only=True; `series_file_counts` / `study_windows` are the
    released infer.py run() / DicomStudyDataset.__getitem__ bodies factored out (proven byte-identical);
  * everything below the second ruler is ours: checkpoint discovery + validation, the instrumented study dataset
    (COUNTERS ONLY -- the windows come from the verbatim functions), the batched forward, the robustness guard.

PRE-REGISTERED configuration (EXPERIMENTS.md A18; reports/pub_reader_validation.md) -- do not change:
  the 3 public checkpoints, k = 12 windows per series (the released infer.py), fp16 autocast on CUDA, probability
  mean over the 3 checkpoints, ONE pass (S = 1, no TTA). Guard: the arm is dropped if > 2 % of the test studies are
  invalid (0 valid slots, or the forward raised) or > 2 % of the selected series hit >= 1 decode error; below that,
  an invalid study is simply absent from the arm's frame and the across-arm average for that row uses the other
  arms only (combine_partial.rank_average_frames_partial).

In the Kaggle bundle this file runs in its OWN module namespace (`reader_gpjw`), not flattened into the bundle:
several of its names (N_SLOTS, SIZE, MEAN, STD, PLANES, load_series, ...) mean something else to the other arms.
"""
import contextlib
import gc
import glob
import hashlib
import re
import time

import pandas as pd

# ================================================================ verbatim port (dev/pub_reader/gpjw_port.py) ====
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
# ================================================================ end of the verbatim port ========================


# ----------------------------------------------------------------------------- the `reader` arm (ours)
READER_K = 12                       # PRE-REGISTERED: the released infer.py (= the author's k_eval)
READER_RES = 256
READER_NAME = "cnxt_v0"
READER_BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"
READER_FOLDS = (0, 1, 2)
READER_SHA256 = {                   # the validated files (Kaggle dataset v1; reports/pub_reader_validation.md §1)
    0: "0b64baa9d1c29c8d5c4dffe06d9ab2c941119b37738e5a2346e882ab694e7b80",
    1: "61696b6e67474695242fc8049e96e2ea13987db6e20d04410dc2961a689585cb",
    2: "f5cf19cf591600c1c1a04f1aa5ff467751ef01513426a4ea3baa789443656cb6",
}
READER_DATASET = "rsna-knee-2-5d-convnext-reader"
READER_OWNER = "goodpjw2008"
READER_FILE_RE = re.compile(r"^cnxt_v0_fold(\d+)\.pt$")
GUARD_MAX_INVALID_FRAC = 0.02       # PRE-REGISTERED (A18)
GUARD_MAX_SERIES_ERR_FRAC = 0.02    # PRE-REGISTERED (A18)
ID_COL = "StudyInstanceUID"
_PRUNE = {"train_series", "test_series", "competitions"}   # never descend into a DICOM tree


def is_reader_file(path):
    """File-name half of the member test (the args half is check_reader_args)."""
    return READER_FILE_RE.match(os.path.basename(str(path))) is not None


def _hint_dirs(root):
    return ([os.path.join(root, READER_DATASET),
             os.path.join(root, "datasets", READER_OWNER, READER_DATASET)]
            + sorted(glob.glob(os.path.join(root, "*" + READER_DATASET + "*")))
            + sorted(glob.glob(os.path.join(root, "datasets", "*", "*" + READER_DATASET + "*"))))


def _files_under(d, max_depth):
    """Reader checkpoints at most `max_depth` directory levels below `d`; DICOM trees are pruned."""
    hits = []
    for dp, dn, fns in os.walk(d):
        rel = os.path.relpath(dp, d)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        dn[:] = [] if depth >= max_depth else sorted(x for x in dn if x not in _PRUNE)
        hits += [os.path.join(dp, fn) for fn in sorted(fns) if is_reader_file(fn)]
    return hits


def find_reader_checkpoints(roots=("/kaggle/input",)):
    """Shallow search for the attached public reader dataset. It mounts either at <root>/<slug> or at
    <root>/datasets/<owner>/<slug>: both are probed by name first; only if neither holds a checkpoint is a
    depth-bounded (4 levels), DICOM-pruned walk of the roots tried. Never a full walk of /kaggle/input (341 s)."""
    t0 = time.time()
    roots = [r for i, r in enumerate(roots) if r and os.path.isdir(r)
             and os.path.realpath(r) not in {os.path.realpath(q) for q in roots[:i] if q}]
    hits, seen, how = [], set(), "hint"
    for root in roots:
        for d in _hint_dirs(root):
            rd = os.path.realpath(d)
            if rd in seen or not os.path.isdir(rd):
                continue
            seen.add(rd)
            hits += _files_under(d, 3)
    if not hits:
        how = "bounded-walk"
        for root in roots:
            hits += _files_under(root, 4)
    by_name = {}
    for p in hits:                              # one file per name (a dataset visible through two paths)
        by_name.setdefault(os.path.basename(p), p)
    out = [by_name[k] for k in sorted(by_name)]
    print(f"[time] stage=reader_weights_search elapsed={time.time() - t0:.1f}s via={how} found={len(out)}",
          flush=True)
    return out


def reader_ckpt_args(path):
    """A reader checkpoint's `args`, read with the SAFE unpickler (weights_only=True)."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    try:
        return dict(ck.get("args") or {})
    finally:
        del ck


def check_reader_args(path, args):
    """-> fold. Raises ValueError unless the file name AND the checkpoint's own args are the validated reader's."""
    name = os.path.basename(str(path))
    m = READER_FILE_RE.match(name)
    if m is None:
        raise ValueError(f"{name}: not a cnxt_v0_fold<k>.pt file")
    fold = int(m.group(1))
    want = {"name": READER_NAME, "backbone": READER_BACKBONE, "res": READER_RES, "k_eval": READER_K, "fold": fold}
    bad = {k: (args.get(k), v) for k, v in want.items() if args.get(k) != v}
    if bad:
        raise ValueError(f"{name}: args differ from the validated reader (got, want): {bad}")
    if fold not in READER_FOLDS:
        raise ValueError(f"{name}: fold {fold} is not one of the validated {READER_FOLDS}")
    return fold


def _sha256(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


def load_reader_members(paths, device):
    """The 3 validated checkpoints: name + args + sha256 checked, then loaded STRICTLY with weights_only=True
    through the port's `load_models`. Raises unless exactly folds 0, 1, 2 load (the pre-registered arm is all
    three; anything else must drop the arm, not silently shrink it)."""
    members = {}
    for p in sorted(paths):
        name = os.path.basename(p)
        fold = check_reader_args(p, reader_ckpt_args(p))
        sha = _sha256(p)
        if sha != READER_SHA256[fold]:
            raise ValueError(f"{name}: sha256 {sha[:16]} != validated {READER_SHA256[fold][:16]}")
        if fold in members:
            raise ValueError(f"duplicate reader fold {fold}: {name}")
        (model, res, a), = load_models([p], device=device)
        n_par = sum(t.numel() for t in model.parameters())
        print(f"[member] reader {name}: name={a['name']} fold={a['fold']} backbone={a['backbone']} res={res} "
              f"k_eval={a['k_eval']} labels={a.get('labels')} params={n_par / 1e6:.2f}M sha256={sha[:16]} "
              f"weights_only=True strict=True", flush=True)
        members[fold] = (model, int(res))
    if sorted(members) != list(READER_FOLDS):
        raise ValueError(f"the reader arm needs folds {READER_FOLDS}; loaded {sorted(members)}")
    return [(f, members[f][0], members[f][1]) for f in READER_FOLDS]


def usable_series_rows(se):
    """Rows build_study_table can place (plane in PLANES, Fat_Suppression 0/1). On a clean CSV (every
    competition CSV we have) this keeps every row, unchanged and in order; a malformed row would otherwise
    raise inside the verbatim `slot_of` and take the whole arm down."""
    fs = pd.to_numeric(se["Fat_Suppression"], errors="coerce")
    keep = se["Anatomical_Plane"].isin(PLANES).values & fs.isin([0, 1]).values
    return se[keep], int((~keep).sum())


class ReaderStudyDataset(torch.utils.data.Dataset):
    """The released infer.py DicomStudyDataset.__getitem__ (through the ported `study_windows`) plus per-series
    decode COUNTERS. The counters sit in the `vol_of` wrapper around the verbatim `load_series`; the windows,
    mask and positions are byte-identical to DicomStudyDataset's (dev/sub13/verify_reader_module.py)."""

    def __init__(self, studies, table, root, k=READER_K):
        self.studies, self.table, self.root, self.k = list(studies), table, root, k

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        slots = self.table.get(st, [[] for _ in range(N_SLOTS)])
        series = []

        def vol_of(sr):
            rec = {"series": str(sr), "n_raw": 0, "n_bad": 0, "n": 0, "error": ""}
            series.append(rec)
            try:
                vol, info = load_series(f"{self.root}/{st}/{sr}")
            except Exception as e:      # study_windows skips the slot exactly as DicomStudyDataset does
                rec["error"] = f"{type(e).__name__}: {e}"[:200]
                raise
            rec["n_raw"], rec["n_bad"], rec["n"] = int(info.get("n_raw", 0)), int(info.get("n_bad", 0)), len(vol)
            return vol

        x, mask, pos = study_windows(vol_of, st, slots, self.k)
        return (st, torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos),
                {"slot_cands": int(sum(1 for c in slots if c)), "series": series})


def _collate(items):
    return ([it[0] for it in items], torch.stack([it[1] for it in items]), torch.stack([it[2] for it in items]),
            torch.stack([it[3] for it in items]), [it[4] for it in items])


def _worker_init(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def _amp(device):
    """The released infer.py runs torch.autocast("cuda", dtype=torch.float16) -- fp16 on EVERY CUDA GPU (not the
    bf16 our other arms pick on sm_80+): the validated reference was produced that way. CPU: plain fp32."""
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


_FP32_RECOMPUTES = {"n": 0}


@torch.no_grad()
def reader_forward(model, res, x, mask, pos, device):
    """Sigmoid probabilities [B, 12] computed as the released infer.py does (`m(x, mask, pos, res=res).float()
    .sigmoid()` under fp16 autocast). Non-finite half-precision logits -> the batch is recomputed in fp32; still
    non-finite -> FloatingPointError (the caller retries per study and marks a failing study invalid)."""
    with _amp(device):
        logits = model(x, mask, pos, res=res)
        p = logits.float().sigmoid()
    if not bool(torch.isfinite(logits).all()):
        _FP32_RECOMPUTES["n"] += 1
        logits = model(x, mask, pos, res=res)
        p = logits.float().sigmoid()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("non-finite reader logits in fp32")
    return p


def _forward_rows(members, x, mask, pos, rows, device):
    """Forward rows `rows` of one collated batch through every member -> ({row: [p per member]}, {row: error}).
    One forward per member for the whole sub-batch; if any raises, every study is retried alone, and a study
    whose own forward raises is INVALID (it drops out of the arm's frame)."""
    cuda = str(device).startswith("cuda")

    def run(sel):
        if len(sel) == x.shape[0]:
            xb, mb, pb = x, mask, pos
        else:
            ix = torch.as_tensor(sel, dtype=torch.long)
            xb, mb, pb = x.index_select(0, ix), mask.index_select(0, ix), pos.index_select(0, ix)
        xb, mb, pb = xb.to(device, non_blocking=cuda), mb.to(device), pb.to(device)
        return [reader_forward(m, res, xb, mb, pb, device).cpu().numpy() for _, m, res in members]

    try:
        outs = run(rows)
        return {j: [o[i] for o in outs] for i, j in enumerate(rows)}, {}
    except Exception as e:
        print(f"[reader] batch forward raised ({type(e).__name__}: {str(e)[:120]}) -> retrying {len(rows)} "
              f"study(ies) one by one", flush=True)
    ok, bad = {}, {}
    for j in rows:
        if cuda:
            torch.cuda.empty_cache()
        try:
            outs = run([j])
            ok[j] = [o[0] for o in outs]
        except Exception as e:
            bad[j] = f"{type(e).__name__}: {str(e)[:160]}"
    return ok, bad


def reader_guard(stats, n_test):
    """PRE-REGISTERED (A18): drop the arm if invalid studies > 2 % of the test studies OR series with >= 1 decode
    error > 2 % of the reader's selected series. A study is invalid if it has 0 valid slots or its forward raised."""
    n_valid = int(stats["valid"].sum()) if len(stats) else 0
    invalid = int(n_test) - n_valid
    n_series = int(stats["series_selected"].sum()) if len(stats) else 0
    n_err = int(stats["series_err"].sum()) if len(stats) else 0
    inv_frac = invalid / max(1, int(n_test))
    err_frac = n_err / n_series if n_series else 1.0
    ok = int(n_test) > 0 and inv_frac <= GUARD_MAX_INVALID_FRAC and err_frac <= GUARD_MAX_SERIES_ERR_FRAC
    return {"studies": int(n_test), "valid": n_valid, "invalid": invalid, "invalid_frac": inv_frac,
            "series": n_series, "series_err": n_err, "series_err_frac": err_frac,
            "slots_mean": float(stats["slots_valid"].sum()) / max(1, int(n_test)) if len(stats) else 0.0,
            "guard_ok": bool(ok)}


def predict_reader(paths, root, split, study_ids, device="cuda", batch=4, workers=4, deadline=None):
    """Run the arm on `study_ids` of the mount `root` (<split>_series.csv + <split>_series/).

    Returns {"frame": [ID_COL + 12] probability mean of the 3 checkpoints for the VALID studies only,
             "per_ckpt": {fold: frame}, "stats": per-study counters, "summary": reader_guard(...), "guard_ok": bool}.
    Raises on anything that is not a per-study problem (missing/mismatched checkpoint, CSV, DataLoader, the
    `deadline` [epoch seconds] passing): the caller drops the whole arm."""
    t0 = time.time()
    if deadline is not None and time.time() > deadline:
        raise TimeoutError("reader deadline already passed before the arm started")
    members = load_reader_members(paths, device)
    try:
        print(f"[arm reader] {len(members)} member(s) loaded (folds {[f for f, _, _ in members]}): k={READER_K} "
              f"res={READER_RES} fp16-autocast={str(device).startswith('cuda')} probability mean, S=1, "
              f"batch={batch}, workers={workers}, device={device}", flush=True)
        print(f"[time] stage=reader_member_load elapsed={time.time() - t0:.1f}s", flush=True)
        t1 = time.time()
        ids = [str(s) for s in study_ids]
        sroot = os.path.join(root, f"{split}_series")
        se = pd.read_csv(os.path.join(root, f"{split}_series.csv"))
        se = se[se.StudyInstanceUID.astype(str).isin(set(ids))]
        se, n_drop = usable_series_rows(se)
        if n_drop:
            print(f"[reader] WARNING {n_drop} series row(s) with an unusable Anatomical_Plane/Fat_Suppression "
                  f"left out of the slot table", flush=True)
        n_files = series_file_counts(se, sroot)
        table = build_study_table(se, n_files)
        print(f"[reader] slot table: {len(ids)} studies, {len(se)} series rows, {len(table)} studies with >= 1 "
              f"candidate series", flush=True)
        print(f"[time] stage=reader_table elapsed={time.time() - t1:.1f}s", flush=True)

        cuda = str(device).startswith("cuda")
        kw = dict(batch_size=max(1, int(batch)), num_workers=max(0, int(workers)), collate_fn=_collate,
                  pin_memory=cuda)
        if int(workers) > 0:
            kw.update(worker_init_fn=_worker_init, prefetch_factor=2)
        dl = torch.utils.data.DataLoader(ReaderStudyDataset(ids, table, sroot, READER_K), **kw)
        probs, rows = {}, []
        wait_s = fwd_s = 0.0
        t2 = t_log = time.time()
        it = iter(dl)
        while True:
            tw = time.time()
            try:
                sts, x, mask, pos, infos = next(it)
            except StopIteration:
                break
            wait_s += time.time() - tw
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"reader deadline passed after {len(rows)}/{len(ids)} studies")
            has_slot = mask.any(1).tolist()
            todo = [j for j, h in enumerate(has_slot) if h]
            tf = time.time()
            ok, bad = _forward_rows(members, x, mask, pos, todo, device) if todo else ({}, {})
            fwd_s += time.time() - tf
            for j, st in enumerate(sts):
                sel = infos[j]["series"]
                if j in ok:
                    probs[st] = ok[j]
                rows.append({
                    ID_COL: st,
                    "slot_cands": infos[j]["slot_cands"],
                    "slots_valid": int(mask[j].sum()),
                    "series_selected": len(sel),
                    "series_err": sum(1 for r in sel if r["error"] or r["n_bad"] > 0),
                    "series_short": sum(1 for r in sel if not r["error"] and r["n"] < 3),
                    "slices_bad": sum(r["n_bad"] for r in sel),
                    "slices_used": sum(min(r["n"], MAX_SLICES) for r in sel if r["n"] >= 3),
                    "forward": "ok" if j in ok else ("no-slots" if not has_slot[j] else "raised"),
                    "valid": j in ok,
                    "error": " | ".join([r["error"] for r in sel if r["error"]] + ([bad[j]] if j in bad else []))[:300],
                })
            if time.time() - t_log > 120:
                el = time.time() - t2
                print(f"[reader] {len(rows)}/{len(ids)} studies {el:.0f}s ({len(rows) / max(el, 1e-9):.2f} st/s) "
                      f"loader-wait {wait_s:.0f}s forward {fwd_s:.0f}s", flush=True)
                t_log = time.time()
        print(f"[time] stage=reader_predict elapsed={time.time() - t2:.1f}s loader_wait={wait_s:.1f}s "
              f"forward={fwd_s:.1f}s studies={len(rows)}", flush=True)
    finally:
        for _, m, _ in members:
            try:
                m.to("cpu")
            except Exception:
                pass
        del members
        gc.collect()
        if str(device).startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    stats = pd.DataFrame(rows, columns=[ID_COL, "slot_cands", "slots_valid", "series_selected", "series_err",
                                        "series_short", "slices_bad", "slices_used", "forward", "valid", "error"])
    summ = reader_guard(stats, len(ids))
    valid_ids = [s for s in ids if s in probs]
    per_ckpt = {}
    for k, f in enumerate(READER_FOLDS):
        arr = np.stack([probs[s][k] for s in valid_ids]).astype(np.float32) if valid_ids \
            else np.zeros((0, len(LABELS)), np.float32)
        d = pd.DataFrame(arr, columns=LABELS)
        d.insert(0, ID_COL, valid_ids)
        per_ckpt[f] = d
    mean = np.stack([per_ckpt[f][LABELS].values for f in READER_FOLDS]).mean(0) if valid_ids \
        else np.zeros((0, len(LABELS)), np.float32)
    frame = pd.DataFrame(mean.astype(np.float32), columns=LABELS)
    frame.insert(0, ID_COL, valid_ids)
    print(f"[reader] studies={summ['studies']} valid={summ['valid']} slots_mean={summ['slots_mean']:.2f} "
          f"series={summ['series']} series_err={summ['series_err']} guard={'ok' if summ['guard_ok'] else 'TRIPPED'}",
          flush=True)
    print(f"[reader] invalid={summ['invalid']} ({summ['invalid_frac']:.2%} of studies; drop if > 2%) "
          f"no-slot={int((stats['forward'] == 'no-slots').sum())} forward-raised={int((stats['forward'] == 'raised').sum())} "
          f"series_err={summ['series_err']} ({summ['series_err_frac']:.2%} of selected series; drop if > 2%) "
          f"slices_bad={int(stats['slices_bad'].sum())} short_series={int(stats['series_short'].sum())} "
          f"fp32_recomputes={_FP32_RECOMPUTES['n']}", flush=True)
    if not summ["guard_ok"]:
        print(f"[reader] GUARD TRIPPED: invalid studies {summ['invalid']}/{summ['studies']} "
              f"({summ['invalid_frac']:.2%}), series with decode errors {summ['series_err']}/{summ['series']} "
              f"({summ['series_err_frac']:.2%}); limits 2% / 2% -> the reader arm is DROPPED and the submission is "
              f"the other arms alone", flush=True)
    print(f"[time] stage=reader_total elapsed={time.time() - t0:.1f}s", flush=True)
    return {"frame": frame, "per_ckpt": per_ckpt, "stats": stats, "summary": summ, "guard_ok": summ["guard_ok"]}


def write_reader_outputs(res, out_dir, prefix="reader"):
    """Per-study counters + per-checkpoint and mean probabilities as CSVs (diagnostics; never read back)."""
    os.makedirs(out_dir, exist_ok=True)
    res["stats"].to_csv(os.path.join(out_dir, f"{prefix}_study_stats.csv"), index=False)
    for f, d in res["per_ckpt"].items():
        d.to_csv(os.path.join(out_dir, f"{prefix}_pred_fold{f}.csv"), index=False, float_format="%.8g")
    res["frame"].to_csv(os.path.join(out_dir, f"{prefix}_pred_mean.csv"), index=False, float_format="%.8g")
