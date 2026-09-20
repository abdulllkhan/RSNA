"""Stitched (all-fold) 3-arm fit: rank-avg(base_tta3=1, hr_tta3=3, v2=w) vs steven / consensus / 58 gold.
Usage: .venv/bin/python dev/fit_v2_stitched.py v2_effv2s384_lr [more v2 names -> averaged as one arm]
Prints the grid and `RECOMMENDED_W=<int>` (max steven-OOF, ties -> smaller w)."""
import sys, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
def stitch(name):
    fs=sorted(glob.glob(f"data/oof/{name}_fold[0-9].parquet")); d=pd.concat([pd.read_parquet(f) for f in fs])
    if "StudyInstanceUID" in d.columns: d=d.set_index("StudyInstanceUID")
    d=d[~d.index.duplicated()]; print(f"{name}: {len(fs)} folds, {len(d)} studies"); return d[T]
names=sys.argv[1:]; b=stitch("effv2s_k24_tta3"); h=stitch("effv2s_k24_384_tta3"); V=[stitch(n) for n in names]
idx=b.index.intersection(h.index)
for v in V: idx=idx.intersection(v.index)
print("common studies", len(idx))
sv=pd.read_csv("data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID").loc[idx]
cons=pd.read_parquet("data/labels/consensus_v1_nogold.parquet").loc[idx]
tr=pd.read_csv("data/comp_csv/train.csv").set_index("StudyInstanceUID"); gold=tr.dropna(subset=T)[T]; g=idx.intersection(gold.index)
ys=(sv>0.5).astype(int); yc=(cons[T]>0.5).astype(int); yg=gold.loc[g].astype(int)
def macro(P,Y): return np.mean([roc_auc_score(Y[c],P[c]) for c in T if Y[c].nunique()>1])
rb=b.loc[idx].rank(pct=True); rh=h.loc[idx].rank(pct=True); rv=sum(v.loc[idx].rank(pct=True) for v in V)/len(V)
print(f"v2 arm alone: steven {macro(rv,ys):.4f} consensus {macro(rv,yc):.4f} gold58 {macro(rv.loc[g],yg):.4f} (n_gold={len(g)})")
best=None
for w in [0,1,2,3,4,5,6,8,10]:
    E=(rb+3*rh+w*rv)/(4+w); s_=macro(E,ys); print(f"  w={w:<3} steven {s_:.4f}  consensus {macro(E,yc):.4f}  gold58 {macro(E.loc[g],yg):.4f}")
    if best is None or s_>best[1]+1e-6: best=(w,s_)
print(f"RECOMMENDED_W={best[0]}")
