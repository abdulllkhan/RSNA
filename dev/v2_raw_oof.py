"""Build a RAW-weights checkpoint from a train_v2 `_last.pt` (whose `model` key is the EMA) and write the
fold's OOF with those raw weights via infer_v2 on the cache.
Usage: PYTHONPATH=. .venv/bin/python dev/v2_raw_oof.py models/v2_effv2s384_fold0_last.pt [--device cuda] [--limit N] [--out models/v2_effv2s384_fold0.pt]
Prints steven-OOF macro and gold-fold macro; writes data/oof/<name>_fold<k>.parquet (unless --limit)."""
import argparse, os, sys, numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from src.rsna.model_v2 import build_model_v2, config_from_dict_v2, compute_fingerprint_v2
from src.rsna.infer_v2 import load_members_v2, predict_studies_v2
T=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
ap=argparse.ArgumentParser(); ap.add_argument("last"); ap.add_argument("--device", default="cuda"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=None); ap.add_argument("--batch", type=int, default=2); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--labels", default="data/labels/consensus_v1.parquet"); ap.add_argument("--cache-root", default="data/cache_c140_384")
ap.add_argument("--source", default="raw_model", choices=("raw_model","model"))
a=ap.parse_args()
st=torch.load(a.last, map_location="cpu", weights_only=False)
args=dict(st["args"]); name=args["name"]; fold=int(args["fold"]); size=int(args["size"])
print(f"[raw_oof] {a.last}: epoch={st.get('epoch')} name={name} fold={fold} size={size} source={a.source}")
sd=st[a.source]
cfg=config_from_dict_v2(st["cfg"]); model=build_model_v2(cfg, verbose=False); model.load_state_dict(sd); model.eval()
fp=compute_fingerprint_v2(model, "cpu", int(args.get("k_eval", 64)), size)
new={k: st[k] for k in ("cfg","args","mean","std","targets","epoch","best_auc") if k in st}
new["model"]=sd; new["fingerprint"]=fp; new["args"]=dict(args, model_source=a.source.replace("_model",""))
out=a.out or f"models/{name}_fold{fold}.pt"
if a.limit: out=out.replace(".pt", "_rawsmoke.pt")
torch.save(new, out); print(f"[raw_oof] wrote {out} ({os.path.getsize(out)/1e6:.0f} MB)")
labels=pd.read_parquet(a.labels); ids=[s for s in labels.index[labels.fold==fold]]
if a.limit: ids=ids[:a.limit]
members=load_members_v2([out], device=a.device)
pred=predict_studies_v2(members, a.cache_root, split="train", batch=a.batch, workers=a.workers, device=a.device, study_ids=ids,
                        side_parquet="data/study_side.parquet", order_parquet="data/series_order.parquet", dicom_root="data/raw")
if "StudyInstanceUID" not in pred.columns: pred=pred.reset_index().rename(columns={"index":"StudyInstanceUID"})
pred=pred[["StudyInstanceUID"]+T]
sv=pd.read_csv("data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")
P=pred.set_index("StudyInstanceUID"); idx=P.index.intersection(sv.index)
au=[roc_auc_score((sv.loc[idx,t]>0.5).astype(int), P.loc[idx,t]) for t in T if (sv.loc[idx,t]>0.5).nunique()>1]
tr=pd.read_csv("data/comp_csv/train.csv").set_index("StudyInstanceUID"); gold=tr.dropna(subset=T)[T]; g=P.index.intersection(gold.index)
ga=[roc_auc_score(gold.loc[g,t].astype(int), P.loc[g,t]) for t in T if gold.loc[g,t].nunique()>1] if len(g)>1 else []
print(f"[raw_oof] n={len(idx)} steven macro={np.mean(au):.4f} | gold-fold n={len(g)} macro={np.mean(ga) if ga else float('nan'):.4f}")
if not a.limit:
    op=f"data/oof/{name}_fold{fold}.parquet"; pred.to_parquet(op, index=False); print(f"[raw_oof] OOF -> {op} ({len(pred)} rows)")
print("[raw_oof] DONE")
