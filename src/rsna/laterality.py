"""Which knee is this? — study-level laterality and the canonicalisation it implies.

WHY THIS FILE EXISTS (reports/public_solutions.md §4 #3)
-------------------------------------------------------
Four of the twelve targets are side-specific (Medial/Lateral Meniscus, Medial/Lateral OA).
A left knee is the mirror image of a right one, so if the training set mixes both and nothing
normalises them, "medial" is on the left half of the image in half the studies and on the
right half in the other half — those four labels are learned as noise, and the failure is
invisible in the loss curve. Every 0.93+ public solution canonicalises every study to a RIGHT
knee before the network sees it, and consequently uses NO horizontal-flip augmentation (a
flip would put the nuisance axis straight back).

WHERE THE SIDE COMES FROM
-------------------------
1. **The DICOM tag** — `Laterality` (0020,0060) or `ImageLaterality` (0020,0062). Present
   and unambiguous on a minority of studies here (it is very often the empty string), but
   when it is present it is the ground truth, and it is what the geometry rule below is
   validated against (`dev/laterality_audit.py`).
2. **Geometry**, for everything else. DICOM patient coordinates are LPS: +x = patient LEFT,
   +y = posterior, +z = superior. A knee sits off the midline, so the sign of the image's
   x coordinate says which leg it is.

   The trap `sadamtorres` measured: `ImagePositionPatient` is the centre of the first
   **corner** voxel, not the centre of the image. For a sagittal knee series the in-plane
   axes carry a large x component, so the corner can sit on the opposite side of the midline
   from the anatomy. Correcting to the image CENTRE

       centre = IPP + row_dir * row_mm * (Rows/2) + col_dir * col_mm * (Cols/2)
       row_dir = IOP[3:6]  (direction of increasing ROW index, spacing PixelSpacing[0])
       col_dir = IOP[0:3]  (direction of increasing COLUMN index, spacing PixelSpacing[1])

   takes agreement with the populated tag from **58.8 % to 96.9 %** in his write-up. Our own
   measurement on this training set is in `reports/decode_v2.md` §6 — the sign convention
   below (`LEFT_IS_POSITIVE_X`) was CHECKED against the tags rather than assumed.

3. **Dead zone / default.** Near the midline (|x| < `DEAD_ZONE_MM`) the geometry is not
   trustworthy, so a study with no tag there is called `'R'` with `source='default'` —
   i.e. left untouched by the canonicalisation, the safe direction: a wrong flip is worse
   than no flip.

The decision is taken per STUDY (median x over every slice of every series), not per series:
all series of one exam image the same knee, and the median is robust to a stray localiser.

WHAT CANONICALISATION MEANS
---------------------------
`canonical_ops(side)` returns the two operations the public solutions apply for a left knee:
horizontally flip the CORONAL and AXIAL planes (medial/lateral is the image's left-right
axis there), and REVERSE the sagittal slice order (medial/lateral is the through-plane axis
there, so it is the stack that must be mirrored, not the picture). It must be applied
identically in training and at inference or it is worse than doing nothing.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# DICOM patient coordinates are LPS: +x is the patient's LEFT. Empirically re-verified
# against the populated Laterality tags on train (dev/laterality_audit.py); flip this to
# False if a future data set disagrees and the audit will follow.
LEFT_IS_POSITIVE_X = True

# |x_centre| below this is too close to the midline to trust; no tag -> 'default'.
DEAD_ZONE_MM = 20.0

DEFAULT_SIDE = "R"

# Planes mirrored left<->right for a left knee; the sagittal stack is reversed instead.
FLIP_LR_PLANES: Tuple[str, ...] = ("Coronal", "Axial")


def slice_centre_lps(meta: Optional[dict]) -> Optional[np.ndarray]:
    """Patient-space (LPS, mm) coordinate of the CENTRE of one slice, or None.

    `meta` is a `decode.slice_meta_v2` dict (`ipp`, `iop`, `pixel_spacing`, `rows`, `cols`).
    Returns `IPP + row_dir*row_mm*(Rows/2) + col_dir*col_mm*(Cols/2)` — see the module
    docstring for why the uncorrected corner IPP is not good enough.
    """
    if not meta:
        return None
    ipp, iop = meta.get("ipp"), meta.get("iop")
    if ipp is None or iop is None:
        return None
    try:
        ipp = np.asarray(ipp, dtype=np.float64).reshape(3)
        iop = np.asarray(iop, dtype=np.float64).reshape(6)
        row_mm, col_mm = (float(x) for x in meta.get("pixel_spacing", (1.0, 1.0)))
        rows, cols = int(meta.get("rows", 0) or 0), int(meta.get("cols", 0) or 0)
    except Exception:
        return None
    if not np.all(np.isfinite(ipp)) or not np.all(np.isfinite(iop)):
        return None
    col_dir, row_dir = iop[0:3], iop[3:6]
    return ipp + row_dir * row_mm * (rows / 2.0) + col_dir * col_mm * (cols / 2.0)


def slice_corner_x(meta: Optional[dict]) -> Optional[float]:
    """Uncorrected `IPP[0]` — the naive corner rule, kept only so the audit can quantify it."""
    if not meta or meta.get("ipp") is None:
        return None
    try:
        return float(np.asarray(meta["ipp"], dtype=np.float64).reshape(3)[0])
    except Exception:
        return None


def _iter_metas(series_metas: Sequence[Sequence[Optional[dict]]]):
    for series in series_metas or ():
        for meta in series or ():
            if meta:
                yield meta


def tag_side(series_metas: Sequence[Sequence[Optional[dict]]]) -> Tuple[Optional[str], int, bool]:
    """`(side, n_slices_with_tag, consistent)` from the Laterality / ImageLaterality tags.

    `side` is None when no slice carries one. `consistent` is False when slices of the same
    study disagree, in which case the caller must fall back to geometry.
    """
    votes: Dict[str, int] = {}
    for meta in _iter_metas(series_metas):
        v = meta.get("laterality_tag")
        if v in ("L", "R"):
            votes[v] = votes.get(v, 0) + 1
    if not votes:
        return None, 0, True
    n = sum(votes.values())
    if len(votes) == 1:
        return next(iter(votes)), n, True
    return max(votes, key=votes.get), n, False


def study_x_centre(series_metas: Sequence[Sequence[Optional[dict]]], corner: bool = False) -> float:
    """Median x (mm, LPS) over every slice of every series of one study; NaN if unavailable.

    `corner=True` uses the raw `IPP[0]` instead of the centre-corrected coordinate — the
    comparison `dev/laterality_audit.py` reports.
    """
    xs: List[float] = []
    for meta in _iter_metas(series_metas):
        if corner:
            v = slice_corner_x(meta)
        else:
            c = slice_centre_lps(meta)
            v = None if c is None else float(c[0])
        if v is not None and np.isfinite(v):
            xs.append(v)
    return float(np.median(xs)) if xs else float("nan")


def side_from_x(x_centre_mm: float, dead_zone_mm: float = DEAD_ZONE_MM) -> Optional[str]:
    """`'L'`/`'R'` from a signed x, or None inside the dead zone / for NaN."""
    if x_centre_mm is None or not np.isfinite(x_centre_mm) or abs(x_centre_mm) < dead_zone_mm:
        return None
    positive_is_left = bool(LEFT_IS_POSITIVE_X)
    is_left = (x_centre_mm > 0) if positive_is_left else (x_centre_mm < 0)
    return "L" if is_left else "R"


def side_from_series_meta(
    series_metas: List[List[dict]],
    dead_zone_mm: float = DEAD_ZONE_MM,
) -> Tuple[str, str, float]:
    """Which knee one STUDY images: `(side, source, x_centre_mm)`.

    Args:
        series_metas: one list of `decode.slice_meta_v2` dicts per series of the study
            (`None` entries — undecodable slices — are ignored).
        dead_zone_mm: |x| below which the geometry is not trusted (default 20 mm).

    Returns:
        `side`   — `'L'` or `'R'`;
        `source` — `'tag'` (a consistent Laterality/ImageLaterality tag decided it),
                   `'geometry'` (the centre-corrected median x decided it),
                   `'default'` (no tag and |x| inside the dead zone, or no geometry at all
                   -> `'R'`, i.e. the canonicalisation leaves the study alone);
        `x_centre_mm` — the median centre-corrected x, kept for auditing even when the tag
                   decided (NaN if no slice had usable geometry).
    """
    x = study_x_centre(series_metas, corner=False)
    tside, n_tag, consistent = tag_side(series_metas)
    if tside is not None and consistent and n_tag > 0:
        return tside, "tag", x
    gside = side_from_x(x, dead_zone_mm)
    if gside is not None:
        return gside, "geometry", x
    return DEFAULT_SIDE, "default", x


def canonical_ops(side: str) -> dict:
    """The operations that turn a `side` study into the canonical RIGHT knee.

    `{'flip_lr_planes': ('Coronal', 'Axial'), 'reverse_sagittal': True}` for `'L'`;
    `{'flip_lr_planes': (), 'reverse_sagittal': False}` (no-ops) for `'R'` or anything else.

    Coronal/axial images carry medial<->lateral across the image width, so they are mirrored
    (`torch.flip(img, dims=[-1])`); a sagittal series carries it through-plane, so its slice
    ORDER is reversed instead. Applying only one of the two is worse than applying neither.
    """
    if str(side).upper() == "L":
        return {"flip_lr_planes": FLIP_LR_PLANES, "reverse_sagittal": True}
    return {"flip_lr_planes": (), "reverse_sagittal": False}
