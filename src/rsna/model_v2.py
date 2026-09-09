"""Wave-4 "v2" study model — per-slice backbone -> per-FINDING gated attention over 64 slots.

    x [B, K, 3, S, S]  --shared backbone (chunked, optionally grad-checkpointed)-->  [B, K, D]
                       + slot embedding (5)  + within-slot position embedding (16 buckets)
    per-finding gated attention over the K slices, with
        * an additive FIXED anatomical slot-prior bias   SLOT_PRIOR[f, slot] * strength
        * a mask that removes the slots this study has no series for
    -> [B, 12, D]  --12 independent linear heads-->  logits [B, 12]

WHY EACH PIECE IS THERE (reports/public_solutions.md)
-----------------------------------------------------
* **Per-finding attention MIL** is the pooling every 0.92+ solution uses; the Raptor arm
  (`coatnet_rmlp_2_rw_384`, LB 0.924 as a single model) is literally
  `att: Linear(F,256) -> Tanh -> Linear(256,12)`, `softmax(dim=1)`,
  `einsum("bkn,bkf->bnf")` (§1). We keep our GATED variant (`tanh(Vh) * sigmoid(Uh)`,
  Ilse et al., as in `model.GatedAttentionPool` / `model_plane`) and make the output
  per-finding, so each of the 12 findings pools its own weighted combination of slices.
* **The fixed slot prior** (§2d.1) — `SLOT_PRIOR_TABLE` + `SLOT_PRIOR_STRENGTH = 0.55`,
  i.e. exp(0.55) ~ 1.73x the attention weight of an unpreferred slot. It is a *prior*, not a
  gate: an unpreferred slot can still win if the evidence is there. `--slot-prior 0` disables.
* **Slot + position embeddings** — the bag is a fixed layout (18 Sag-fluid, 14 Sag-nonfluid,
  ...), so the network can be told which sequence a slice came from and where in the stack it
  sat, instead of having to infer both from pixels. `sadamtorres`'s arm does the same with a
  plane embedding + normalised slice position (§1, table row).

SLOT_PRIOR MAPPING. The public table is over SIX slots (plane x fat-sat); ours has FIVE
(plane x fluid, Axial merged). The mapping specified for this arm:

    ACL                -> both Sagittal
    MCL                -> both Coronal
    Medial/Lateral Meniscus -> both Sagittal + both Coronal
    Medial/Lateral OA  -> both Coronal + Sagittal-nonfluid
    PF OA              -> both Sagittal + Axial
    Effusion, Synovitis-> Sagittal-fluid + Axial
    Baker's            -> Sagittal-fluid
    Contusion          -> Sagittal-fluid + Coronal-fluid + Axial
    Fracture           -> all five

MEMORY KNOBS. `--slice-chunk N` forwards the B*K slices through the backbone N at a time
(bounds peak activation memory without changing the maths); `--grad-ckpt` recomputes each
chunk's activations in the backward pass (roughly trades 1.3x compute for ~K/chunk less
activation memory). Both are no-ops at inference except for the chunking itself.

A NOTE ON THE PARAMETER NAMED `plane_logits`. It is the (optional, zero-initialised) LEARNED
per-finding x slot bias added to the fixed prior — the same object `model_plane` calls
`plane_logits`, over slots instead of planes. It is deliberately given that name because
`dev/slim_ckpt.py` (which this package may not edit) routes a checkpoint down its
arm-agnostic branch when `"plane_logits" in ckpt["model"]`; every other branch tries to
rebuild a `model.StudyModel` and would fail on a v2 state dict. It is frozen at zero unless
`--learn-slot-bias 1`, so by default it contributes exactly nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt_util

from . import TARGETS
from .dataset_v2 import K_FULL, N_SLOTS, SLOT_NAMES, SLOTS  # noqa: F401

# Slot indices: 0 Sag-fluid | 1 Sag-nonfluid | 2 Cor-fluid | 3 Cor-nonfluid | 4 Axial-any
SLOT_PRIOR_SLOTS: Dict[str, Tuple[int, ...]] = {
    "ACL": (0, 1),
    "MCL": (2, 3),
    "Medial Meniscus": (0, 1, 2, 3),
    "Lateral Meniscus": (0, 1, 2, 3),
    "Medial OA": (1, 2, 3),
    "Lateral OA": (1, 2, 3),
    "PF OA": (0, 1, 4),
    "Effusion": (0, 4),
    "Synovitis": (0, 4),
    "Baker's": (0,),
    "Contusion": (0, 2, 4),
    "Fracture": (0, 1, 2, 3, 4),
}
DEFAULT_SLOT_PRIOR_STRENGTH = 0.55   # exp(0.55) ~ 1.73x an unpreferred slot
DEFAULT_POS_BUCKETS = 16

# ViT-family names need `img_size` (and often `dynamic_img_size`) to run off their native res,
# and must NOT be fed channels_last activations.
_VIT_TAGS = ("vit_", "deit", "beit", "eva", "dinov2", "siglip", "clip")


def slot_prior_table(strength: float = DEFAULT_SLOT_PRIOR_STRENGTH,
                     targets: Sequence[str] = TARGETS) -> np.ndarray:
    """`[12, 5]` additive attention bias: `strength` on preferred slots, 0 elsewhere."""
    tab = np.zeros((len(targets), N_SLOTS), dtype=np.float32)
    for i, t in enumerate(targets):
        for s in SLOT_PRIOR_SLOTS.get(t, ()):
            tab[i, s] = float(strength)
    return tab


def format_slot_prior(strength: float = DEFAULT_SLOT_PRIOR_STRENGTH) -> str:
    """The table as printed at model build (the artefact the report quotes)."""
    tab = slot_prior_table(strength)
    w = max(len(t) for t in TARGETS)
    head = " " * (w + 2) + "  ".join(f"{n:>12}" for n in SLOT_NAMES)
    lines = [f"[model_v2] SLOT_PRIOR (strength={strength:g}; . = 0, x = +{strength:g})", head]
    for i, t in enumerate(TARGETS):
        cells = "  ".join(f"{('x' if tab[i, s] else '.'):>12}" for s in range(N_SLOTS))
        lines.append(f"  {t:<{w}}{cells}")
    lines.append(f"  slots: {list(zip(SLOT_NAMES, [s[2] for s in SLOTS]))} -> K_full={K_FULL}")
    return "\n".join(lines)


@dataclass
class ModelV2Config:
    backbone: str = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
    pretrained: bool = True
    size: int = 384
    attn_dim: int = 256
    dropout: float = 0.2
    n_classes: int = len(TARGETS)
    n_slots: int = N_SLOTS
    pos_buckets: int = DEFAULT_POS_BUCKETS
    slot_prior: float = DEFAULT_SLOT_PRIOR_STRENGTH
    learn_slot_bias: bool = False
    grad_ckpt: bool = False
    slice_chunk: int = 0            # 0 = all B*K slices in one backbone call
    channels_last: Optional[bool] = None   # None -> auto (CNN yes, ViT no)


def _is_vit(name: str) -> bool:
    n = name.lower()
    return any(t in n for t in _VIT_TAGS)


def _create_backbone(cfg: ModelV2Config) -> nn.Module:
    """`timm.create_model` with the `img_size` / `dynamic_img_size` cascade for ViTs.

    Tried in order, so one code path serves CoAtNet-384, EfficientNetV2 and DINOv2-ViT:
      1. `img_size=S, dynamic_img_size=True`  (ViTs off their native resolution)
      2. `img_size=S`                          (models with a fixed-size positional grid)
      3. plain                                 (CNNs, which do not take `img_size`)

    `global_pool="avg"` is forced only for the CNNs (the convention `model.py` and
    `model_plane.py` use). A ViT is left on its own default (`token` for the DINOv2 weights):
    asking timm for `avg` on a ViT swaps `norm` for `fc_norm` in the head, and the DINOv2
    state dict then fails to load with "Missing key(s): fc_norm.weight" — i.e. the choice is
    not cosmetic, it decides whether the pretrained weights load at all.
    """
    base = dict(pretrained=cfg.pretrained, num_classes=0)
    if not _is_vit(cfg.backbone):
        base["global_pool"] = "avg"
    attempts: List[dict] = []
    if _is_vit(cfg.backbone):
        attempts.append(dict(img_size=int(cfg.size), dynamic_img_size=True))
    attempts.append(dict(img_size=int(cfg.size)))
    attempts.append({})
    last: Optional[Exception] = None
    for extra in attempts:
        try:
            m = timm.create_model(cfg.backbone, **base, **extra)
            if extra:
                print(f"[model_v2] backbone {cfg.backbone} built with {extra}")
            return m
        except (TypeError, RuntimeError, ValueError, KeyError) as e:
            last = e
    raise RuntimeError(f"could not build backbone {cfg.backbone!r}: {last}")


class PerFindingGatedAttention(nn.Module):
    """Gated attention MIL (Ilse et al.) with ONE softmax per finding.

    `model.GatedAttentionPool` produces a single [B,K] attention map; this produces [B,K,C]
    — twelve maps sharing the gate features, which is exactly what the public per-finding
    MIL heads do (`Linear(F,256) -> Tanh -> Linear(256,12)`, softmax over K).
    """

    def __init__(self, in_dim: int, attn_dim: int = 256, n_classes: int = len(TARGETS)):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, n_classes)

    def forward(self, h: torch.Tensor, bias: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None):
        # h: [B,K,D] -> a: [B,K,C]
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))
        if bias is not None:
            a = a + bias.to(a.dtype)
        if mask is not None:
            # A study with NO valid slice at all (impossible for a real study, cheap to
            # guard) would give an all -inf row -> NaN. Treat it as "everything valid".
            m = mask | (~mask.any(dim=1, keepdim=True))
            a = a.masked_fill(~m[..., None], float("-inf"))
        a = torch.softmax(a, dim=1)
        # `softmax` is an fp32 op under autocast while `h` stays bf16; cast explicitly rather
        # than relying on einsum's autocast policy, so the no-autocast fp32 fallback path in
        # `infer_hr._forward_with_guard` behaves identically.
        z = torch.einsum("bkc,bkd->bcd", a.to(h.dtype), h)      # [B, C, D]
        return z, a


class StudyModelV2(nn.Module):
    """Backbone -> (+slot,+position embeddings) -> per-finding gated attention -> 12 heads."""

    def __init__(self, cfg: ModelV2Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = _create_backbone(cfg)
        self.feat_dim = int(self.backbone.num_features)
        self.slot_emb = nn.Embedding(cfg.n_slots, self.feat_dim)
        self.pos_emb = nn.Embedding(cfg.pos_buckets, self.feat_dim)
        nn.init.normal_(self.slot_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.norm = nn.LayerNorm(self.feat_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.attn = PerFindingGatedAttention(self.feat_dim, cfg.attn_dim, cfg.n_classes)

        # Fixed anatomical prior (buffer -> travels in the state_dict, never trained).
        self.register_buffer("slot_prior",
                             torch.from_numpy(slot_prior_table(cfg.slot_prior)), persistent=True)
        # Learned per-finding x slot bias; see the module docstring for the name.
        self.plane_logits = nn.Parameter(torch.zeros(cfg.n_classes, cfg.n_slots),
                                         requires_grad=bool(cfg.learn_slot_bias))
        # 12 independent linear heads on the 12 pooled vectors (the baseline's param budget).
        bound = 1.0 / (self.feat_dim ** 0.5)
        self.head_w = nn.Parameter(torch.empty(cfg.n_classes, self.feat_dim).uniform_(-bound, bound))
        self.head_b = nn.Parameter(torch.zeros(cfg.n_classes))
        self._channels_last = (not _is_vit(cfg.backbone)) if cfg.channels_last is None \
            else bool(cfg.channels_last)

    # ---- plumbing --------------------------------------------------------------------
    def data_config(self):
        return timm.data.resolve_model_data_config(self.backbone)

    def n_params(self) -> Dict[str, int]:
        bb = sum(p.numel() for p in self.backbone.parameters())
        tot = sum(p.numel() for p in self.parameters())
        return {"backbone": bb, "head": tot - bb, "total": tot}

    def param_groups(self, lr: float, lr_backbone_mult: float = 0.1):
        """Layer-wise LR: the pretrained backbone is ADAPTED (lr x mult), the head is trained."""
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        bb_ids = {id(p) for p in self.backbone.parameters()}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in bb_ids]
        return [{"params": bb, "lr": lr * float(lr_backbone_mult)},
                {"params": rest, "lr": lr}]

    @torch.no_grad()
    def slot_bias_table(self) -> np.ndarray:
        """`[12, 5]` total additive attention bias actually in use (fixed + learned)."""
        return (self.slot_prior.detach().float()
                + self.plane_logits.detach().float()).cpu().numpy()

    # ---- forward ---------------------------------------------------------------------
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """[B,K,3,S,S] -> [B,K,D], chunked over slices and optionally grad-checkpointed."""
        b, k = x.shape[0], x.shape[1]
        flat = x.reshape(b * k, *x.shape[2:])
        if self._channels_last and flat.is_cuda:
            flat = flat.to(memory_format=torch.channels_last)
        chunk = int(self.cfg.slice_chunk) or (b * k)
        use_ckpt = bool(self.cfg.grad_ckpt) and self.training and torch.is_grad_enabled()
        outs: List[torch.Tensor] = []
        for i in range(0, b * k, chunk):
            sub = flat[i:i + chunk]
            if use_ckpt:
                outs.append(ckpt_util.checkpoint(self.backbone, sub, use_reentrant=False))
            else:
                outs.append(self.backbone(sub))
        feats = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        return feats.reshape(b, k, self.feat_dim)

    def forward(
        self,
        x: torch.Tensor,
        slot_id: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """x [B,K,3,S,S]; slot_id [B,K] int; pos [B,K] float in [0,1]; valid [B,K] bool.

        `slot_id=None` reconstructs the canonical layout for K == the scaled bag size, so a
        synthetic fingerprint tensor can be forwarded without carrying the side tensors.
        """
        b, k = x.shape[0], x.shape[1]
        dev = x.device
        if slot_id is None:
            from .dataset_v2 import slot_id_vector
            sid = slot_id_vector(k / float(K_FULL))
            if len(sid) != k:                       # non-canonical K -> spread over the slots
                sid = np.floor(np.linspace(0, N_SLOTS - 1e-6, k)).astype(np.int64)
            slot_id = torch.from_numpy(sid).to(dev)[None].expand(b, -1)
        slot_id = slot_id.to(device=dev, dtype=torch.long)
        if pos is None:
            pos = torch.linspace(0.0, 1.0, k, device=dev)[None].expand(b, -1)
        pos = pos.to(device=dev, dtype=torch.float32)
        if valid is None:
            valid = torch.ones(b, k, dtype=torch.bool, device=dev)
        valid = valid.to(device=dev, dtype=torch.bool)

        feats = self._features(x)                                          # [B,K,D]
        buckets = torch.clamp((pos * self.cfg.pos_buckets).long(), 0, self.cfg.pos_buckets - 1)
        feats = feats + self.slot_emb(slot_id).to(feats.dtype) \
                      + self.pos_emb(buckets).to(feats.dtype)
        h = self.drop(self.norm(feats))

        bias_tab = self.slot_prior.to(dev) + self.plane_logits.to(dev)     # [C, S]
        bias = bias_tab.t()[slot_id]                                       # [B, K, C]
        z, a = self.attn(h, bias=bias, mask=valid)                         # [B,C,D], [B,K,C]
        logits = torch.einsum("bcd,cd->bc", z, self.head_w.to(z.dtype)) + self.head_b.to(z.dtype)
        if return_aux:
            return logits, dict(attn=a, pooled=z, feats=feats)
        return logits


def build_model_v2(cfg: Optional[ModelV2Config] = None, verbose: bool = True, **kwargs):
    if cfg is None:
        cfg = ModelV2Config(**kwargs)
    model = StudyModelV2(cfg)
    if verbose:
        print(format_slot_prior(cfg.slot_prior), flush=True)
        p = model.n_params()
        print(f"[model_v2] backbone={cfg.backbone} size={cfg.size} feat_dim={model.feat_dim} "
              f"params: backbone={p['backbone'] / 1e6:.1f}M head={p['head'] / 1e6:.2f}M "
              f"total={p['total'] / 1e6:.1f}M | channels_last={model._channels_last} "
              f"grad_ckpt={cfg.grad_ckpt} slice_chunk={cfg.slice_chunk or 'all'} "
              f"learn_slot_bias={cfg.learn_slot_bias}", flush=True)
    return model


def config_from_dict_v2(d: Dict) -> ModelV2Config:
    """Rebuild a config from a checkpoint's `cfg` dict, ignoring unknown/legacy keys."""
    fields = ModelV2Config.__dataclass_fields__
    return ModelV2Config(**{k: v for k, v in dict(d).items() if k in fields})


# --------------------------------------------------------------------------------------
# Fingerprint (shared by train_v2 and infer_v2 -- one definition, one synthetic input)
# --------------------------------------------------------------------------------------
def fingerprint_slot_ids(k: int) -> np.ndarray:
    """Canonical `slot_id` for a bag of `k` slices (the scaled slot layout)."""
    from .dataset_v2 import slot_id_vector
    sid = slot_id_vector(float(k) / float(K_FULL))
    if len(sid) != k:                    # non-canonical K -> spread evenly over the slots
        sid = np.floor(np.linspace(0, N_SLOTS - 1e-6, k)).astype(np.int64)
    return sid.astype(np.int64)


def fingerprint_pos(k: int) -> np.ndarray:
    """Canonical within-slot positions `(j+0.5)/n` for a bag of `k` slices."""
    sid = fingerprint_slot_ids(k)
    pos = np.zeros(k, dtype=np.float32)
    for s in np.unique(sid):
        m = sid == s
        n = int(m.sum())
        pos[m] = (np.arange(n, dtype=np.float32) + 0.5) / n
    return pos


def compute_fingerprint_v2(model, device: str, k: int, size: int) -> np.ndarray:
    """True-fp32 (TF32 off), no-autocast output on a fixed synthetic bag.

    The synthetic input carries the canonical `slot_id` / `pos` / `valid` for this K, so the
    fingerprint exercises the slot and position embeddings and the prior bias as well as the
    backbone: a checkpoint loaded with the wrong slot layout is caught, not just a wrong
    backbone. Portable across CPU/GPU, which is why it is the ckpt's stored self-check.
    """
    was_training = model.training
    model.eval()
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        g = torch.Generator().manual_seed(1234)
        x = torch.randn(1, k, 3, size, size, generator=g).to(device)
        sid = torch.from_numpy(fingerprint_slot_ids(k)).to(device)[None]
        pos = torch.from_numpy(fingerprint_pos(k)).to(device)[None]
        val = torch.ones(1, k, dtype=torch.bool, device=device)
        with torch.no_grad():
            return torch.sigmoid(model(x, sid, pos, val).float()).cpu().numpy().ravel()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cudnn
        if was_training:
            model.train()


__all__ = [
    "ModelV2Config", "StudyModelV2", "PerFindingGatedAttention", "build_model_v2",
    "config_from_dict_v2", "slot_prior_table", "format_slot_prior", "SLOT_PRIOR_SLOTS",
    "DEFAULT_SLOT_PRIOR_STRENGTH", "DEFAULT_POS_BUCKETS",
    "compute_fingerprint_v2", "fingerprint_slot_ids", "fingerprint_pos",
]
