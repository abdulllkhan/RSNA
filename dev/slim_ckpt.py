"""Slim a training checkpoint to inference-only keys (CPU-only; safe while the GPU trains).
Keeps exactly what src/rsna/infer.load_members reads: model, cfg, mean, std, args, fingerprint.
Recomputes a portable true-fp32 fingerprint on CPU (advisory at inference).
Usage: .venv/bin/python dev/slim_ckpt.py models/effv2s_k24_fold1.pt   (writes <name>.slim.pt)
"""
import os, sys
import numpy as np, torch
from src.rsna.infer import compute_fingerprint, load_members
from src.rsna.model import ModelConfig, build_model

KEEP = ["model", "cfg", "mean", "std", "args", "fingerprint"]

def main(cp):
    st = torch.load(cp, map_location="cpu", weights_only=False)
    print("full keys:", sorted(st.keys()), "| size MB:", round(os.path.getsize(cp)/1e6, 1))
    is_plane = bool(st.get("args", {}).get("plane_mode")) or "plane_logits" in st.get("model", {})
    keep = KEEP + (["plane_weights", "plane_order"] if is_plane else [])
    slim = {kk: st[kk] for kk in keep if kk in st}
    if is_plane:
        # Arm-agnostic path: the baseline model class cannot host a plane state_dict; keep the
        # fingerprint stored at training time (advisory at inference) and skip the rebuild.
        print("plane checkpoint: keeping stored fingerprint; no rebuild")
    else:
        cfg = ModelConfig(**{k: v for k, v in st["cfg"].items() if k in ModelConfig.__dataclass_fields__})
        cfg.pretrained = False
        model = build_model(cfg).to("cpu"); model.load_state_dict(st["model"]); model.eval()
        slim["fingerprint"] = compute_fingerprint(model, "cpu", st["args"].get("k", 24))
    out = cp[:-3] + ".slim.pt"
    torch.save(slim, out)
    print("slim keys:", sorted(slim.keys()), "| size MB:", round(os.path.getsize(out)/1e6, 1))
    if is_plane:
        chk = torch.load(out, map_location="cpu", weights_only=False)
        assert "model" in chk and "plane_logits" in chk["model"] and "cfg" in chk and "args" in chk
        print("verify (plane) OK: keys present; size", chk["args"].get("size"), "| k_per_plane", chk["args"].get("k_per_plane"))
    else:
        m = load_members([out], device="cpu")   # verify it loads through the real inference path
        print("verify load_members OK; k =", m[0].k, "| fingerprint len =", len(m[0].fingerprint))
    return out

if __name__ == "__main__":
    main(sys.argv[1])
