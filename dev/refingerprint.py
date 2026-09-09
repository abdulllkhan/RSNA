"""Recompute a portable (true-fp32, TF32-disabled) fingerprint on a trained checkpoint
and re-save it in place. Also cross-checks GPU vs CPU agreement so we know the tolerance
holds on Kaggle's T4/P100 (which do true fp32, no TF32).

Run AFTER training, when the GPU is free:
    .venv/bin/python dev/refingerprint.py models/effv2s_k24_fold0.pt
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from src.rsna.infer import compute_fingerprint
from src.rsna.model import ModelConfig, build_model


def load_model(cp, device):
    st = torch.load(cp, map_location=device, weights_only=False)
    cfg = ModelConfig(**{k: v for k, v in st["cfg"].items()
                         if k in ModelConfig.__dataclass_fields__})
    cfg.pretrained = False
    model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(st["model"])
    model.eval()
    return st, model, st["args"].get("k", 24)


def main():
    cp = sys.argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st, model, k = load_model(cp, dev)
    old = np.asarray(st.get("fingerprint"))
    fp_gpu = compute_fingerprint(model, dev, k)
    print("old(stored) vs new(true-fp32) max|diff| =",
          f"{float(np.max(np.abs(old - fp_gpu))):.3e}" if old is not None and old.size else "n/a")

    # cross-device check (Kaggle T4/P100 do true fp32)
    if dev == "cuda":
        _st2, model_cpu, _k = load_model(cp, "cpu")
        fp_cpu = compute_fingerprint(model_cpu, "cpu", k)
        print(f"GPU-fp32 vs CPU-fp32 max|diff| = {float(np.max(np.abs(fp_gpu - fp_cpu))):.3e} "
              f"(must be < 2e-3 for the check to hold cross-hardware)")

    st["fingerprint"] = fp_gpu
    torch.save(st, cp)
    print(f"re-saved true-fp32 fingerprint into {cp}")


if __name__ == "__main__":
    main()
