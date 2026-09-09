"""Plane-aware study model — the Wave-3 "A3" arm.

Baseline (`model.StudyModel`):

    [B, 24, 3, S, S] --backbone--> [B, 24, D] --ONE gated-attention pool--> [B, D]
                                                            --Linear(D, 12)--> 12 logits

The 24 slices are mixed across planes, so a single softmax has to serve all 12 findings at
once. This module keeps the SAME per-slice backbone and the SAME gated-attention mechanism
(`model.GatedAttentionPool` is imported, not re-implemented) and changes only the pooling
topology:

    [B, K1+K2+K3, 3, S, S] --shared backbone--> [B, Ktot, D]
        split by plane -> Sagittal [B,K1,D] | Coronal [B,K2,D] | Axial [B,K3,D]
        3 SEPARATE gated-attention pools    -> z_sag, z_cor, z_ax   (each [B, D])
        per-finding plane weighting         -> w[b, c, :] = softmax_p(P[c, p])   (P: 12x3, learned)
                                               z_c[b] = sum_p w[b,c,p] * z_p[b]  ([B, 12, D])
        12 per-finding linear heads         -> logits [B, 12]

`plane_mask` ([B, 3] bool, from `dataset_plane.StudyPlaneBagDataset`) is applied in BOTH
places: an absent plane is excluded from the per-finding softmax (`-inf` before softmax) so
its filler slices can never contribute, and, in `fusion="concat"`, its embedding is zeroed.

`fusion` ablation:
  * `"planewise"` (default, the hypothesis): the 12x3 learned plane logits above. The softmax
    is a *per-finding routing table* — `plane_weight_table()` reads it out, which is the
    interpretable artefact the report wants ("which plane did MCL choose?"). Head parameters:
    12xD + 12, i.e. EXACTLY the baseline's `Linear(D, 12)` budget.
  * `"concat"` (control): plain `Linear(3D, 12)` over the concatenated plane embeddings. It
    can express anything the planewise head can (and more), but spends 3x the head params
    and gets no explicit per-finding routing — the ablation that says whether the structure,
    not just the plane-separated pooling, is what helps.

Everything else matches the baseline so the arms stay comparable: same `tf_efficientnetv2_s`,
same `timm` data config, same LayerNorm+Dropout before the head, channels_last on CUDA only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn

from . import TARGETS
from .dataset_plane import DEFAULT_K_PER_PLANE, N_PLANES, PLANE_ORDER
from .model import GatedAttentionPool  # reused verbatim (Ilse et al. gated attention MIL)

FUSIONS = ("planewise", "concat")


@dataclass
class PlaneModelConfig:
    backbone: str = "tf_efficientnetv2_s"
    pretrained: bool = True
    attn_dim: int = 256
    dropout: float = 0.2
    n_classes: int = len(TARGETS)
    k_per_plane: Tuple[int, int, int] = DEFAULT_K_PER_PLANE
    fusion: str = "planewise"
    plane_temp: float = 1.0  # softmax temperature on the plane logits (>1 = softer routing)

    def __post_init__(self):
        self.k_per_plane = tuple(int(v) for v in self.k_per_plane)
        if len(self.k_per_plane) != N_PLANES:
            raise ValueError(f"k_per_plane needs {N_PLANES} entries, got {self.k_per_plane}")
        if self.fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}, got {self.fusion!r}")


def _masked_plane_softmax(logits: torch.Tensor, mask: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """softmax over the plane axis with absent planes removed.

    logits: [B, C, P]; mask: [B, P] bool. A row whose mask is all-False (impossible for a
    non-empty study, but cheap to guard) falls back to a uniform distribution instead of NaN.
    """
    m = mask[:, None, :].expand_as(logits)
    x = (logits / max(1e-6, float(temp))).masked_fill(~m, float("-inf"))
    empty = ~mask.any(dim=1)                       # [B]
    if bool(empty.any()):
        x = torch.where(empty[:, None, None].expand_as(x), torch.zeros_like(x), x)
    return torch.softmax(x, dim=-1)


class StudyPlaneModel(nn.Module):
    """Shared per-slice backbone -> 3 per-plane attention pools -> per-finding plane fusion."""

    def __init__(self, cfg: PlaneModelConfig):
        super().__init__()
        self.cfg = cfg
        self.k_per_plane = tuple(cfg.k_per_plane)
        self.k_total = int(sum(self.k_per_plane))
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=cfg.pretrained, num_classes=0, global_pool="avg"
        )
        self.feat_dim = self.backbone.num_features
        # One gated-attention pool per plane: what makes a slice "informative" differs by
        # plane (a good coronal slice is not a good sagittal slice), so they do not share.
        self.attn = nn.ModuleList(
            [GatedAttentionPool(self.feat_dim, cfg.attn_dim) for _ in range(N_PLANES)]
        )
        self.norm = nn.LayerNorm(self.feat_dim)
        self.drop = nn.Dropout(cfg.dropout)

        if cfg.fusion == "planewise":
            # Learned per-finding routing over planes (softmaxed, masked). Zero-init => the
            # model STARTS as a uniform plane average (a strict generalisation of the
            # baseline's mixed pooling) and has to earn any specialisation.
            self.plane_logits = nn.Parameter(torch.zeros(cfg.n_classes, N_PLANES))
            # 12 independent linear heads, each on its own finding-specific embedding.
            bound = 1.0 / (self.feat_dim ** 0.5)
            self.head_w = nn.Parameter(torch.empty(cfg.n_classes, self.feat_dim).uniform_(-bound, bound))
            self.head_b = nn.Parameter(torch.zeros(cfg.n_classes))
        else:  # concat control
            self.concat_norm = nn.LayerNorm(self.feat_dim * N_PLANES)
            self.head = nn.Linear(self.feat_dim * N_PLANES, cfg.n_classes)

    # ---- plumbing ---------------------------------------------------------------------
    def data_config(self):
        """timm data config (mean/std/input_size) for consistent preprocessing."""
        return timm.data.resolve_model_data_config(self.backbone)

    def _split_planes(self, feats: torch.Tensor) -> List[torch.Tensor]:
        """[B, Ktot, D] -> [ [B,K1,D], [B,K2,D], [B,K3,D] ] in canonical PLANE_ORDER."""
        sizes = [k for k in self.k_per_plane]
        if feats.shape[1] != sum(sizes):
            raise ValueError(f"expected {sum(sizes)} slices/study, got {feats.shape[1]}")
        out, off = [], 0
        for k in sizes:
            out.append(feats[:, off:off + k] if k > 0 else feats[:, off:off])
            off += k
        return out

    # ---- interpretation ---------------------------------------------------------------
    @torch.no_grad()
    def plane_weight_table(self) -> np.ndarray:
        """[12, 3] per-finding plane weights (rows sum to 1), all planes assumed present.

        `planewise`: the softmaxed learned routing table — the direct read-out.
        `concat`:    the L2 norm of each plane's block of the head weight, normalised. Not a
                     softmax, but the same question ("how much does this finding lean on this
                     plane?") answered for the control arm, so the two are comparable.
        """
        if self.cfg.fusion == "planewise":
            w = torch.softmax(self.plane_logits.detach().float() / max(1e-6, self.cfg.plane_temp), dim=-1)
            return w.cpu().numpy()
        W = self.head.weight.detach().float()  # [C, 3D]
        blocks = W.reshape(W.shape[0], N_PLANES, self.feat_dim).norm(dim=-1)  # [C, 3]
        return (blocks / blocks.sum(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()

    def plane_weight_report(self, targets: Sequence[str] = TARGETS) -> Dict[str, Dict[str, float]]:
        """`{finding: {plane: weight}}` — what gets printed each epoch / stored in the ckpt."""
        tab = self.plane_weight_table()
        return {t: {p: round(float(tab[i, j]), 4) for j, p in enumerate(PLANE_ORDER)}
                for i, t in enumerate(targets)}

    # ---- forward ----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        plane_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """x: [B, Ktot, 3, S, S]; plane_mask: [B, 3] bool (None -> all planes present)."""
        b, ktot = x.shape[0], x.shape[1]
        if plane_mask is None:
            plane_mask = torch.ones(b, N_PLANES, dtype=torch.bool, device=x.device)
        else:
            plane_mask = plane_mask.to(device=x.device, dtype=torch.bool)
            # A plane with zero allocated slots is never present, whatever the loader says.
            alloc = torch.tensor([k > 0 for k in self.k_per_plane], device=x.device)
            plane_mask = plane_mask & alloc[None, :]

        flat = x.reshape(b * ktot, *x.shape[2:])
        if flat.is_cuda:  # channels_last is a CUDA-only win; on CPU it costs a copy
            flat = flat.to(memory_format=torch.channels_last)
        feats = self.backbone(flat).reshape(b, ktot, self.feat_dim)

        zs, attns = [], []
        for pi, fp in enumerate(self._split_planes(feats)):
            if fp.shape[1] == 0:  # plane not allocated any slots
                zs.append(torch.zeros(b, self.feat_dim, device=x.device, dtype=feats.dtype))
                attns.append(torch.zeros(b, 0, device=x.device, dtype=feats.dtype))
                continue
            z, a = self.attn[pi](fp)
            zs.append(z)
            attns.append(a)
        Z = torch.stack(zs, dim=1)  # [B, 3, D]

        if self.cfg.fusion == "planewise":
            Zn = self.drop(self.norm(Z))                                  # [B, 3, D]
            pl = self.plane_logits.to(Zn.dtype)[None].expand(b, -1, -1)   # [B, C, 3]
            w = _masked_plane_softmax(pl, plane_mask, self.cfg.plane_temp)
            Zc = torch.einsum("bcp,bpd->bcd", w.to(Zn.dtype), Zn)         # [B, C, D]
            logits = torch.einsum("bcd,cd->bc", Zc, self.head_w.to(Zc.dtype)) + self.head_b.to(Zc.dtype)
        else:
            Zm = Z * plane_mask[:, :, None].to(Z.dtype)                   # zero absent planes
            flatz = Zm.reshape(b, N_PLANES * self.feat_dim)
            logits = self.head(self.drop(self.concat_norm(flatz)))
            w = _masked_plane_softmax(
                torch.zeros(b, self.cfg.n_classes, N_PLANES, device=x.device, dtype=Z.dtype),
                plane_mask, 1.0)

        if return_aux:
            return logits, dict(plane_weights=w, attn=attns, plane_embeddings=Z)
        return logits


def build_plane_model(cfg: Optional[PlaneModelConfig] = None, **kwargs) -> StudyPlaneModel:
    if cfg is None:
        cfg = PlaneModelConfig(**kwargs)
    return StudyPlaneModel(cfg)


def config_from_dict(d: Dict) -> PlaneModelConfig:
    """Rebuild a config from a checkpoint's `cfg` dict, ignoring unknown//legacy keys."""
    fields = PlaneModelConfig.__dataclass_fields__
    return PlaneModelConfig(**{k: v for k, v in dict(d).items() if k in fields})


__all__ = [
    "PlaneModelConfig",
    "StudyPlaneModel",
    "build_plane_model",
    "config_from_dict",
    "FUSIONS",
    "PLANE_ORDER",
    "N_PLANES",
]
