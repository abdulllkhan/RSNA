"""Fit the plane-arm (or hr-arm) weight for the multi-arm rank ensemble on stitched OOF.
Usage: .venv/bin/python dev/fit_arm_weights.py effv2s_k24 effv2s_plane_k8x3_384 [plane]
Scores rank-average(base, w*arm) on (a) OOF vs binarized steven_v4 (n=4407) and (b) the 58 gold,
for w in a grid; prints the table and the recommended RSNA_ARM_WEIGHTS string."""
import glob, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
base_name, arm_name = sys.argv[1], sys.argv[2]; arm_label = sys.argv[3] if len(sys.argv)>3 else "plane"
def stitch(name):
    fs=sorted(glob.glob(f"data/oof/{name}_fold[0-9].parquet")); d=pd.concat([pd.read_parquet(f) for f in fs])
    if "StudyInstanceUID" in d.columns: d=d.set_index("StudyInstanceUID")
    d=d[~d.index.duplicated()]; print(f"{name}: {len(fs)} folds, {len(d)} studies"); return d[T]
b=stitch(base_name); a=stitch(arm_name); idx=b.index.intersection(a.index); b=b.loc[idx]; a=a.loc[idx]
sv=pd.read_csv("data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID").loc[idx]
tr=pd.read_csv("data/comp_csv/train.csv"); gold=tr.dropna(subset=T).set_index("StudyInstanceUID")[T]; gidx=gold.index.intersection(idx)
def macro(pred, truth, cols=T):
    return np.mean([roc_auc_score(truth[c], pred[c]) for c in cols if truth[c].nunique()>1])
rb=b.rank(pct=True); ra=a.rank(pct=True)
print(f"\n{'w_'+arm_label:>8} | OOF vs steven (n={len(idx)}) | gold (n={len(gidx)})")
best=None
for w in [0.0,0.5,1.0,1.5,2.0,3.0,5.0,1e9]:
    ens=(rb + w*ra)/(1+w) if w<1e8 else ra
    s=macro(ens,(sv>0.5).astype(int)); g=macro(ens.loc[gidx],gold.loc[gidx])
    tag = "base only" if w==0 else ("arm only" if w>=1e8 else "")
    print(f"{w:>8} | {s:.4f} | {g:.4f}  {tag}")
    if w<1e8 and (best is None or s>best[1]): best=(w,s,g)
print(f"\nRECOMMENDED (max steven-OOF): RSNA_ARM_WEIGHTS=\"base=1,{arm_label}={best[0]}\"  (OOF {best[1]:.4f}, gold {best[2]:.4f})")
