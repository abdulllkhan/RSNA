"""Per-slice timm backbone -> gated attention pooling over slices -> 12-way head.

Input:  [B, K, 3, 224, 224]
Flow:   reshape to [B*K, 3, 224, 224] -> backbone -> [B*K, D] -> [B, K, D]
        -> gated attention (Ilse et al. MIL) over K slices -> [B, D] study embedding
        -> Linear(D, 12) logits (sigmoid at inference).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn

from . import TARGETS


@dataclass
class ModelConfig:
    backbone: str = "tf_efficientnetv2_s"
    pretrained: bool = True
    attn_dim: int = 256
    dropout: float = 0.2
    n_classes: int = len(TARGETS)


class GatedAttentionPool(nn.Module):
    """Gated attention MIL pooling (Ilse, Tomczak, Welling 2018).

    a_k = softmax_k( w^T ( tanh(V h_k) * sigmoid(U h_k) ) ) ; z = sum_k a_k h_k
    """

    def __init__(self, in_dim: int, attn_dim: int = 256):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: [B, K, D]
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))  # [B, K, 1]
        a = a.squeeze(-1)  # [B, K]
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))
        a = torch.softmax(a, dim=1)  # [B, K]
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)  # [B, D]
        return z, a


class StudyModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=cfg.pretrained, num_classes=0, global_pool="avg"
        )
        self.feat_dim = self.backbone.num_features
        self.attn = GatedAttentionPool(self.feat_dim, cfg.attn_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.feat_dim, cfg.n_classes),
        )

    def data_config(self):
        """timm data config (mean/std/input_size) for consistent preprocessing."""
        return timm.data.resolve_model_data_config(self.backbone)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # x: [B, K, 3, 224, 224]
        b, k = x.shape[0], x.shape[1]
        x = x.reshape(b * k, *x.shape[2:])
        x = x.to(memory_format=torch.channels_last)
        feats = self.backbone(x)              # [B*K, D]
        feats = feats.reshape(b, k, self.feat_dim)
        z, a = self.attn(feats)               # [B, D], [B, K]
        logits = self.head(z)                 # [B, 12]
        if return_attn:
            return logits, a
        return logits


def build_model(cfg: Optional[ModelConfig] = None, **kwargs) -> StudyModel:
    if cfg is None:
        cfg = ModelConfig(**kwargs)
    return StudyModel(cfg)
