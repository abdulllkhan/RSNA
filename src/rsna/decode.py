"""Size-parametric DICOM slice decoder — the ONE rendering path for cache + inference.

Wave-3 needs the raw competition DICOM rendered at 384px / 512px, using *exactly* the
recipe that was pixel-validated against the community 224px JPEGs we trained the baseline
on (reverse-engineered + brute-force verified; MAE ~1.1/255, corr > 0.998 — see
`reports/dicom_inference_fix.md`). The recipe:

    ds  = pydicom.dcmread(path)
    px  = ds.pixel_array.astype(float32)
    (multi-frame -> middle frame; RGB -> luminance)
    px  = px * RescaleSlope + RescaleIntercept        # identity on this data (1 / 0)
    if PhotometricInterpretation == "MONOCHROME1": px = px.max() - px
    lo, hi = px.min(), px.max()                        # PER-IMAGE min-max (the key choice;
    norm   = (px - lo) / (hi - lo)                     #  NOT window/level, NOT percentile)
    img    = cv2.resize(norm * 255, (size, size), INTER_AREA)   # plain squish, no aspect crop
    img    = round(clip(img, 0, 255)).astype(uint8)

`read_dicom_slice(path, 224)` reproduces `rsna.dataset._read_slice_dicom(path) * 255`
bit-for-bit (verified on the /tmp/dicom_mount sample by `dev/test_decode.py`). Keeping this
one function as the single code path for BOTH the offline cache build (`dev/decode_raw.py`)
and inference guarantees train/infer consistency at every resolution.

This module is intentionally standalone: it imports nothing from the rest of the package
(pydicom is imported lazily inside the function), so it can be inlined into a Kaggle
submission notebook or imported by a cache builder without dragging in torch.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

cv2.setNumThreads(0)  # avoid oversubscription inside DataLoader / multiprocessing workers

DICOM_EXTS = (".dcm",)

# Default output edge; the baseline (and the community JPEG set) is 224.
DEFAULT_SIZE = 224


def _to_2d(px: np.ndarray, samples_per_pixel: Optional[int]) -> Optional[np.ndarray]:
    """Collapse a 3-D pixel_array to a single 2-D plane.

    Colour (SamplesPerPixel == 3, shape (H, W, 3)) -> ITU-R BT.601 luminance.
    Multi-frame stack (shape (frames, H, W))       -> the middle frame.

    Matches `dataset._read_slice_dicom`, which disambiguates on `shape[-1] == 3` alone;
    we additionally consult SamplesPerPixel so a genuine 3-frame grayscale stack is not
    mistaken for an RGB image. Neither branch is reachable on this competition's MR data
    (every slice is a 2-D MONOCHROME2 frame), so behaviour is identical in practice.
    """
    if px.ndim == 2:
        return px
    if px.ndim != 3:
        return None
    is_colour = px.shape[-1] == 3 and (samples_per_pixel is None or int(samples_per_pixel) == 3)
    if is_colour:
        return px[..., 0] * 0.299 + px[..., 1] * 0.587 + px[..., 2] * 0.114
    return px[px.shape[0] // 2]


def _decode_plane(path: str, verbose: bool = True) -> Tuple[Optional[np.ndarray], Optional[object]]:
    """Steps 0-2 of the recipe: a full-resolution float32 plane BEFORE any normalisation.

    `pixel_array -> float32` (3-D collapsed by `_to_2d`), Modality LUT
    (`RescaleSlope`/`RescaleIntercept`), MONOCHROME1 inversion. Returns `(plane, ds)` so the
    caller can also read header tags (PixelSpacing, Rows/Columns, Laterality) off the SAME
    parse; `(None, ds_or_None)` when the slice is undecodable — never raises.

    Shared by both normalisation paths so they cannot drift: v1 (`_decode_normalized`) then
    applies a PER-IMAGE min-max, v2 (`read_series_v2`) a PER-SERIES percentile. Splitting
    this out changed no v1 arithmetic (proved bit-identical in `reports/decode_v2.md` §5).
    """
    try:
        import pydicom  # lazy: keep the JPEG path and importers free of a hard dependency

        ds = pydicom.dcmread(path)
        px = ds.pixel_array.astype(np.float32)
    except Exception as e:  # unreadable / missing pixel data -> caller skips this slice
        if verbose:
            print(f"[decode] skip {os.path.basename(path)}: {type(e).__name__}: {e}")
        return None, None

    px = _to_2d(px, getattr(ds, "SamplesPerPixel", None))
    if px is None or px.ndim != 2 or px.size == 0:
        if verbose:
            print(f"[decode] skip {os.path.basename(path)}: unusable pixel shape")
        return None, ds
    px = px.astype(np.float32, copy=False)

    # 1. Modality LUT (identity on this data: RescaleSlope=1, RescaleIntercept=0).
    slope = getattr(ds, "RescaleSlope", 1)
    inter = getattr(ds, "RescaleIntercept", 0)
    try:
        px = px * float(slope if slope is not None else 1) + float(inter if inter is not None else 0)
    except Exception:
        pass

    # 2. MONOCHROME1 stores "high value = black" -> invert so bright = high everywhere.
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        px = px.max() - px
    return px, ds


def _decode_normalized(path: str, verbose: bool = True) -> Optional[np.ndarray]:
    """Everything up to (but excluding) the resize: a full-resolution float32 plane whose
    values span [0, 255] after per-image min-max. Returns None if undecodable (never raises).

    Split out so a caller wanting several output sizes pays the DICOM parse + normalisation
    (~11 ms/slice, the dominant cost) ONCE — see `read_dicom_slices`.
    """
    px, _ = _decode_plane(path, verbose)
    if px is None:
        return None

    # 3. Per-image min-max to [0,1] (the validated choice; window/level & percentile clip
    #    were 10-40x worse against the reference JPEGs).
    lo = float(px.min())
    hi = float(px.max())
    if hi <= lo:  # constant image (all-black slice) -> zeros, still a valid in-range frame
        norm = np.zeros_like(px, dtype=np.float32)
    else:
        norm = (px - lo) / (hi - lo)
    return (norm * 255.0).astype(np.float32)


def _resize_u8(nrm: np.ndarray, size: int) -> np.ndarray:
    """4. Resize the normalised plane to the square target with INTER_AREA, quantise to 8-bit."""
    resized = cv2.resize(nrm, (size, size), interpolation=cv2.INTER_AREA)
    return np.clip(np.round(resized), 0, 255).astype(np.uint8)


def read_dicom_slice(
    path: str,
    size: int = DEFAULT_SIZE,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    """Decode one DICOM slice to a `uint8 [size, size]` grayscale image.

    Args:
        path: path to a `.dcm` file.
        size: output edge length in pixels (224 / 384 / 512 ...). Always a square:
              non-square natives (e.g. 640x1280) are squished, exactly as the 224px
              training set was built — no aspect-preserving crop or pad.
        verbose: print a one-line reason when a slice cannot be decoded.

    Returns:
        `np.uint8` array of shape (size, size), or `None` if the slice is undecodable
        (missing/corrupt pixel data, unsupported transfer syntax, non-image object).
        Callers pad/backfill; this function never raises.
    """
    nrm = _decode_normalized(path, verbose)
    return None if nrm is None else _resize_u8(nrm, size)


def read_dicom_slices(path: str, sizes, verbose: bool = True) -> dict:
    """Decode one slice to SEVERAL sizes, sharing the parse+normalisation.

    Returns `{size: uint8[size,size]}` — empty dict if the slice is undecodable. Each entry
    is bit-identical to `read_dicom_slice(path, size)`; this is purely the cheap path for
    the cache builder (2 sizes: ~12 ms/slice instead of ~23 ms).
    """
    nrm = _decode_normalized(path, verbose)
    if nrm is None:
        return {}
    return {int(s): _resize_u8(nrm, int(s)) for s in sizes}


def read_dicom_slice_float(path: str, size: int = DEFAULT_SIZE, verbose: bool = True) -> Optional[np.ndarray]:
    """`read_dicom_slice` as `float32` in [0,1] — drop-in for `dataset._read_slice_dicom`."""
    u8 = read_dicom_slice(path, size=size, verbose=verbose)
    return None if u8 is None else u8.astype(np.float32) / 255.0


def encode_jpeg(img: np.ndarray, quality: int = 95) -> Optional[bytes]:
    """Encode a `uint8 [H,W]` array as a grayscale JPEG. Returns None on failure."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


# ==========================================================================================
# v2 — physical-millimetre crop + per-SERIES percentile normalisation
# ==========================================================================================
# WHY (reports/public_solutions.md §2(c) and §4 #1): every 0.92+ public solution feeds the
# network a fixed PHYSICAL box (CROP_MM 130-140) resized to 336-384 px, normalised by a
# percentile taken over the WHOLE SERIES — not the full frame squished and min-maxed per
# slice, which is what v1 above does.
#
#   * mm crop — `(Rows x PixelSpacing)` has median ~160 mm and ranges 70-320 mm on this data,
#     so a squish puts the same anatomy at a different pixel scale in every series. A 140 mm
#     centre box is inside the FOV of ~99 % of series and always contains the joint, and it
#     fixes mm/px at 140/384 = 0.36 mm — under the 0.5 mm/px Nyquist limit for a 1 mm tear
#     (a 224 px full-frame squish gives ~0.71 mm/px).
#   * per-series percentile — "percentiles over the SERIES, not per slice: preserves
#     inter-slice contrast" (sadamtorres). Per-slice min-max rescales every slice to its own
#     extremes, so a bright artefact on one slice silently changes the apparent intensity of
#     everything else in the stack and the model cannot compare slices.
#
# v1 (`read_dicom_slice` / `read_dicom_slices`) is UNCHANGED and still the path used by the
# `data/cache384` / `data/cache512` caches and every checkpoint trained on them.

DEFAULT_CROP_MM = 140.0
DEFAULT_PCT = (1.0, 99.0)

# Laterality (0020,0060) / ImageLaterality (0020,0062) — read for the laterality audit.
TAG_LATERALITY = (0x0020, 0x0060)
TAG_IMAGE_LATERALITY = (0x0020, 0x0062)

# Exact percentiles need the whole series in one array; a pathological series (hundreds of
# 1024^2 slices) would be GBs. Above this many elements the percentile is taken on a fixed
# stride (deterministic, no RNG) instead. 64 M float32 = 256 MB; the median series is
# 30 slices of 512^2 = 7.9 M, and exactly 2 of the 24,386 train series exceed the cap
# (one 1/2 stride, one 1/3 stride) -- see reports/decode_v2.md.
_PCT_MAX_ELEMS = 64_000_000

_V2_WARNED: Dict[str, bool] = {}


def _warn_once(key: str, msg: str) -> None:
    """Print `msg` the first time `key` is seen (per process). Fallbacks stay loud but cheap."""
    if not _V2_WARNED.get(key):
        _V2_WARNED[key] = True
        print(f"[decode] {msg}")


def pixel_spacing_of(ds) -> Tuple[Tuple[float, float], str]:
    """`((row_mm, col_mm), source)` from a parsed dataset.

    DICOM `PixelSpacing` is `[row spacing, column spacing]` — row spacing is the physical
    distance between adjacent ROWS (i.e. the mm covered by one step down the image), so the
    number of rows in a `crop_mm` box is `crop_mm / row_mm`. Falls back to
    `ImagerPixelSpacing`, then to 1.0 mm with a one-time warning (`source='default'`), which
    makes the crop degenerate into "crop_mm pixels" rather than crashing.
    """
    for attr, src in (("PixelSpacing", "PixelSpacing"), ("ImagerPixelSpacing", "ImagerPixelSpacing")):
        v = getattr(ds, attr, None)
        try:
            if v is not None and len(v) >= 2:
                r, c = float(v[0]), float(v[1])
                if r > 0 and c > 0:
                    return (r, c), src
        except Exception:
            pass
    _warn_once("no_pixel_spacing",
               "WARN no PixelSpacing/ImagerPixelSpacing on a slice -> assuming 1.0 mm/px "
               "(the mm crop degenerates to a pixel crop for it)")
    return (1.0, 1.0), "default"


def _spacing_pair(pixel_spacing) -> Tuple[float, float]:
    """Coerce a caller-supplied spacing to a positive `(row_mm, col_mm)`; 1.0 mm on junk."""
    try:
        if pixel_spacing is None:
            raise ValueError
        r, c = float(pixel_spacing[0]), float(pixel_spacing[1])
        if not (r > 0 and c > 0) or not (np.isfinite(r) and np.isfinite(c)):
            raise ValueError
        return r, c
    except Exception:
        _warn_once("bad_spacing_arg",
                   "WARN crop_resize_v2 got a missing/invalid pixel_spacing -> 1.0 mm/px")
        return 1.0, 1.0


def _plane_list(vol) -> List[np.ndarray]:
    """`[H,W]` planes from a `[n,H,W]` array, a `[H,W]` array, or a sequence of planes.

    A sequence is accepted (a superset of the documented `[n,H,W]`) because slices in one
    series occasionally differ in Rows/Columns and then cannot be stacked.
    """
    if isinstance(vol, np.ndarray):
        if vol.ndim == 3:
            return [vol[i] for i in range(vol.shape[0])]
        if vol.ndim == 2:
            return [vol]
        raise ValueError(f"normalize_series_pct: expected 2-D or 3-D array, got ndim={vol.ndim}")
    return [np.asarray(p) for p in vol]


def series_pct_bounds(planes: Sequence[np.ndarray], pct=DEFAULT_PCT) -> Tuple[float, float]:
    """`(lo, hi)` intensity bounds: the `pct` percentiles over EVERY pixel of the series."""
    planes = [np.asarray(p) for p in planes if p is not None and np.asarray(p).size]
    if not planes:
        return 0.0, 0.0
    total = int(sum(p.size for p in planes))
    if total <= _PCT_MAX_ELEMS:
        flat = np.concatenate([p.reshape(-1) for p in planes])
    else:  # deterministic stride subsample; 2 of 24,386 train series hit this
        step = int(np.ceil(total / _PCT_MAX_ELEMS))
        _warn_once("pct_subsample",
                   f"series has {total} pixels > {_PCT_MAX_ELEMS} -> percentile on a 1/{step} stride")
        flat = np.concatenate([p.reshape(-1)[::step] for p in planes])
    lo, hi = np.percentile(flat.astype(np.float32, copy=False), [float(pct[0]), float(pct[1])])
    return float(lo), float(hi)


def _apply_pct_u8(px: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip one plane to `[lo, hi]` and rescale to `uint8`. Constant series (`hi <= lo`) -> zeros."""
    if hi <= lo:
        return np.zeros(px.shape, dtype=np.uint8)
    nrm = (np.clip(px, lo, hi) - lo) * (255.0 / (hi - lo))
    return np.clip(np.round(nrm), 0, 255).astype(np.uint8)


def normalize_series_pct(vol: np.ndarray, pct=DEFAULT_PCT) -> np.ndarray:
    """Per-SERIES percentile normalisation: `float32 [n,H,W]` -> `uint8 [n,H,W]`.

    `vol` is the stack of planes produced by steps 0-2 of the recipe (`_decode_plane`, i.e.
    what `_decode_normalized` holds BEFORE its per-image min-max: rescale applied,
    MONOCHROME1 inverted, 3-D collapsed). The `pct` percentiles are taken over the whole
    series at once, so a slice's grey level means the same thing everywhere in the stack.

    A constant series (`hi <= lo`, e.g. an all-black stack) returns zeros — a valid,
    in-range frame, matching v1's handling of a constant image.

    A list of planes is also accepted; the return is then a list-shaped stack only if the
    planes share a shape, so ragged series should be normalised via `series_pct_bounds` +
    `_apply_pct_u8` directly (which is what `read_series_v2` does).
    """
    planes = _plane_list(vol)
    if not planes:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    lo, hi = series_pct_bounds(planes, pct)
    return np.stack([_apply_pct_u8(np.asarray(p, dtype=np.float32), lo, hi) for p in planes])


def crop_resize_v2(
    img: np.ndarray,
    pixel_spacing,
    size: int = 384,
    crop_mm: float = DEFAULT_CROP_MM,
) -> np.ndarray:
    """Centre-crop a fixed PHYSICAL box out of `img` and resize it to `size x size`.

    Args:
        img: `uint8 [H, W]` (normalised full frame — `normalize_series_pct` output).
        pixel_spacing: `(row_mm, col_mm)`, i.e. DICOM `PixelSpacing` order. `None`/invalid
            falls back to 1.0 mm/px with a one-time warning.
        size: output edge in pixels.
        crop_mm: box edge in millimetres (140 = the Raptor/CoAtNet arm, 130 = the DINO arm).

    The box is `round(crop_mm/row_mm)` rows x `round(crop_mm/col_mm)` cols, each CLAMPED to
    the image. When the field of view is smaller than `crop_mm` in one axis (or the voxels
    are anisotropic) the clamped rectangle is NOT square, and it is then stretched to the
    square output — the crop keeps as much real anatomy as exists and the aspect distortion
    is confined to the minority of series that are short in one axis (measured in
    `reports/decode_v2.md` §3). The alternative, padding to square, would put the joint
    off-centre relative to every other series.

    Note the resize is always INTER_AREA, as in the public implementations. For a series
    whose crop is smaller than `size` px (large voxels) that is an upsample, where INTER_AREA
    degrades to nearest-neighbour; `reports/decode_v2.md` §3 reports how often that happens.
    """
    img = np.asarray(img)
    if img.ndim != 2 or img.size == 0:
        raise ValueError(f"crop_resize_v2: expected a non-empty 2-D image, got shape {img.shape}")
    h, w = img.shape
    row_mm, col_mm = _spacing_pair(pixel_spacing)
    ch = max(1, min(int(round(float(crop_mm) / row_mm)), h))
    cw = max(1, min(int(round(float(crop_mm) / col_mm)), w))
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    crop = img[r0:r0 + ch, c0:c0 + cw]
    size = int(size)
    if crop.shape != (size, size):
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(crop, dtype=np.uint8)


def slice_meta_v2(ds, rows: Optional[int] = None, cols: Optional[int] = None) -> dict:
    """Ordering + geometry + laterality metadata from an ALREADY-PARSED dataset.

    Keys are a superset of `order.read_slice_meta`'s (`instance_number`, `ipp`, `iop`) —
    same tags, same types, read off the parse `_decode_plane` already did instead of a second
    `dcmread` (`dev/decode_raw_v2.py --check-meta` asserts the two agree). Extra keys:
    `pixel_spacing` `(row_mm, col_mm)`, `spacing_source`, `rows`, `cols`, `laterality_tag`
    (`'L'`/`'R'`/`None` from Laterality else ImageLaterality) and `fov_mm`
    `(rows*row_mm, cols*col_mm)`.

    `rows`/`cols` default to the header's Rows/Columns; pass the decoded plane's shape when
    it can differ (multi-frame, colour).
    """
    meta: dict = {"instance_number": None, "ipp": None, "iop": None}
    try:
        v = getattr(ds, "InstanceNumber", None)
        meta["instance_number"] = None if v is None else int(v)
    except Exception:
        pass
    for key, attr, n in (("ipp", "ImagePositionPatient", 3), ("iop", "ImageOrientationPatient", 6)):
        try:
            v = getattr(ds, attr, None)
            if v is not None and len(v) == n:
                meta[key] = np.asarray([float(x) for x in v], dtype=np.float64)
        except Exception:
            pass

    (row_mm, col_mm), src = pixel_spacing_of(ds)
    meta["pixel_spacing"] = (row_mm, col_mm)
    meta["spacing_source"] = src
    if rows is None:
        rows = int(getattr(ds, "Rows", 0) or 0)
    if cols is None:
        cols = int(getattr(ds, "Columns", 0) or 0)
    meta["rows"], meta["cols"] = int(rows), int(cols)

    lat = None
    for attr in ("Laterality", "ImageLaterality"):
        try:
            v = getattr(ds, attr, None)
            if v:
                s = str(v).strip().upper()
                if s in ("L", "R"):
                    lat = s
                    break
        except Exception:
            pass
    meta["laterality_tag"] = lat
    meta["fov_mm"] = (meta["rows"] * row_mm, meta["cols"] * col_mm)
    return meta


def read_series_v2(
    paths: Sequence[str],
    size: int = 384,
    crop_mm: float = DEFAULT_CROP_MM,
    pct=DEFAULT_PCT,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[Optional[dict]]]:
    """Decode ONE series: `uint8 [n, size, size]` + per-slice metadata, in `paths` order.

    The v2 rendering contract end to end:

        _decode_plane      per slice: parse, 3-D -> 2-D, rescale, MONOCHROME1 invert
        series_pct_bounds  ONE (lo, hi) from the `pct` percentiles of the WHOLE series
        _apply_pct_u8      per slice: clip to (lo, hi) -> uint8
        crop_resize_v2     per slice: centre `crop_mm` box (that slice's own PixelSpacing)
                           -> INTER_AREA -> (size, size)

    Normalisation precedes the crop, so `lo/hi` are the percentiles of the FULL frames (the
    documented contract). Every slice of the series is decoded — sampling is the dataset's
    job, and the percentile needs the whole stack anyway.

    Returns:
        `(vol, metas)` with `len(metas) == len(paths)`. An undecodable slice is an all-zero
        frame in `vol` with `metas[i] is None`; the count is reported once at the end (per
        call) when `verbose`, not once per slice.
    """
    paths = list(paths)
    n = len(paths)
    size = int(size)
    vol = np.zeros((n, size, size), dtype=np.uint8)
    metas: List[Optional[dict]] = [None] * n
    if n == 0:
        return vol, metas

    planes: List[Optional[np.ndarray]] = []
    failed: List[str] = []
    for i, p in enumerate(paths):
        px, ds = _decode_plane(p, verbose=False)
        if px is None:
            planes.append(None)
            failed.append(os.path.basename(p))
            continue
        planes.append(px)
        metas[i] = slice_meta_v2(ds, rows=px.shape[0], cols=px.shape[1])

    lo, hi = series_pct_bounds([p for p in planes if p is not None], pct)
    for i, px in enumerate(planes):
        if px is None:
            continue
        u8 = _apply_pct_u8(px, lo, hi)
        vol[i] = crop_resize_v2(u8, metas[i]["pixel_spacing"], size=size, crop_mm=crop_mm)

    if failed and verbose:
        head = ", ".join(failed[:3]) + (" ..." if len(failed) > 3 else "")
        print(f"[decode] read_series_v2: {len(failed)}/{n} slice(s) undecodable -> zeros ({head})")
    return vol, metas
