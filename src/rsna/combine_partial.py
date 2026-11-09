"""Across-arm rank average with a per-row fallback for an OPTIONAL arm that may lack some studies.

The multi-arm notebook combines finished arm frames with `infer_plane.rank_average_frames`: frames are
inner-joined on the study id, each column is turned into percentile ranks within its frame, and the ranks are
averaged with the arm weights. A study missing from ANY frame drops out of the join and later falls back to 0.5
via `write_submission`'s reindex. That is the right behaviour for our own arms (each covers every study or fails
as a whole) but not for the public `reader` arm, which can lose single studies (no decodable slot, a failed
forward) -- one bad study must not cost that study the other arms' predictions.

`rank_average_frames_partial` keeps `rank_average_frames`' semantics for the REQUIRED frames and adds the
fallback for OPTIONAL ones (EXPERIMENTS A18, pre-registered):
  * the study set is the inner join of the REQUIRED frames, ordered as the first required frame;
  * if every optional frame covers that set, the function returns `rank_average_frames(frames, weights)` itself:
    the same call on the same inputs, so the result is the existing code path, bit for bit;
  * otherwise an optional frame is percentile-ranked within the studies it has, and a study it lacks is averaged
    over the arms it has, with their weights renormalised -- so that row stays on the same [0, 1] rank scale.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import ID_COL, TARGETS
from .infer_plane import rank_average_frames


def rank_average_frames_partial(
    frames: Sequence[pd.DataFrame],
    weights: Sequence[float],
    optional: Sequence[bool],
    log: Optional[callable] = print,
) -> pd.DataFrame:
    """Per-column rank average of `frames` ([ID_COL + TARGETS] each) with per-row fallback for optional ones."""
    if not frames:
        raise ValueError("no frames")
    w = np.asarray(weights, dtype=np.float64)
    if len(w) != len(frames) or len(optional) != len(frames):
        raise ValueError("weights / optional must match frames")
    req = [i for i, o in enumerate(optional) if not o]
    if not req:
        raise ValueError("at least one frame must be required")
    ids = [str(s) for s in frames[req[0]][ID_COL].astype(str)]
    for i in req[1:]:
        keep = set(frames[i][ID_COL].astype(str))
        ids = [s for s in ids if s in keep]
    have = {i: set(frames[i][ID_COL].astype(str)) for i, o in enumerate(optional) if o}
    lacking = {i: [s for s in ids if s not in h] for i, h in have.items()}
    if not any(lacking.values()):
        return rank_average_frames(frames, weights=list(weights))
    if log:
        for i, miss in lacking.items():
            log(f"[combine] optional frame {i} covers {len(ids) - len(miss)}/{len(ids)} studies; the other "
                f"{len(miss)} average the remaining arms only (weights renormalised)")
    acc = np.zeros((len(ids), len(TARGETS)), dtype=np.float64)
    den = np.zeros((len(ids), len(TARGETS)), dtype=np.float64)
    for i, (f, wi) in enumerate(zip(frames, w)):
        g = f.copy()
        g[ID_COL] = g[ID_COL].astype(str)
        g = g.set_index(ID_COL)
        if optional[i]:
            sub = [s for s in ids if s in have[i]]
            r = g.loc[sub][list(TARGETS)].rank(pct=True).reindex(ids).values
            present = ~np.isnan(r)
            acc += wi * np.where(present, r, 0.0)
            den += wi * present
        else:
            acc += wi * g.loc[ids][list(TARGETS)].rank(pct=True).values
            den += wi
    out = pd.DataFrame(acc / np.maximum(den, 1e-12), columns=list(TARGETS))
    out.insert(0, ID_COL, ids)
    return out[[ID_COL] + list(TARGETS)]
