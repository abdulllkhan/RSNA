"""Stitched (all-fold) fit WITHOUT the 224 base arm: rank-avg(hr_tta3=3, v2=w) where the v2 arm is the equal
average of the given member names. Compares several candidate v2 compositions and prints RECOMMENDED_CFG / _W.
Usage: .venv/bin/python dev/fit_v2_stitched2.py "lr+c110" "lr+c110_distill" "lr+c110+c110_distill"
(short names map to v2_effv2s384_<name>; 'lr' = the 140 mm arm). Scores: steven (4,407) / consensus / 58 gold."""
import sys, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
def stitch(name):
    fs=sorted(glob.glob(f"data/oof/{name}_fold[0-9].parquet")); d=pd.concat([pd.read_parquet(f) for f in fs])
    if "StudyInstanceUID" in d.columns: d=d.set_index("StudyInstanceUID")
    d=d[~d.index.duplicated()]; print(f"{name}: {len(fs)} folds, {len(d)} studies"); assert len(fs)==5, f"{name}: need 5 folds"; return d[T]
def full(n): return n if n.startswith("v2_") or n.startswith("effv2s") else f"v2_effv2s384_{n}"
cfgs=sys.argv[1:]; names=sorted({full(n) for c in cfgs for n in c.split("+")})
A={n:stitch(n) for n in names}; h=stitch("effv2s_k24_384_tta3")
idx=h.index
for d in A.values(): idx=idx.intersection(d.index)
print("common studies", len(idx))
sv=pd.read_csv("data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID").loc[idx]
cons=pd.read_parquet("data/labels/consensus_v1_nogold.parquet").loc[idx]
tr=pd.read_csv("data/comp_csv/train.csv").set_index("StudyInstanceUID"); gold=tr.dropna(subset=T)[T]; g=idx.intersection(gold.index)
ys=(sv[T]>0.5).astype(int); yc=(cons[T]>0.5).astype(int); yg=gold.loc[g].astype(int)
def macro(P,Y): return np.mean([roc_auc_score(Y[c],P[c]) for c in T if Y[c].nunique()>1])
R={n:d.loc[idx].rank(pct=True) for n,d in A.items()}; rh=h.loc[idx].rank(pct=True)
best=None
for c in cfgs:
    members=[full(n) for n in c.split("+")]; rv=sum(R[m] for m in members)/len(members)
    print(f"[{c}] v2 arm alone: steven {macro(rv,ys):.4f} consensus {macro(rv,yc):.4f} gold58 {macro(rv.loc[g],yg):.4f}")
    cbest=None
    for w in [4,6,8,10,12,15]:
        E=(3*rh+w*rv)/(3+w); s_=macro(E,ys); print(f"  w={w:<3} steven {s_:.4f}  consensus {macro(E,yc):.4f}  gold58 {macro(E.loc[g],yg):.4f}")
        if cbest is None or s_>cbest[1]+1e-6: cbest=(w,s_)
    print(f"  best for [{c}]: w={cbest[0]} steven {cbest[1]:.4f}")
    if best is None or cbest[1]>best[2]+1e-6: best=(c,cbest[0],cbest[1])
print(f"RECOMMENDED_CFG={best[0]}\nRECOMMENDED_W={best[1]}\nRECOMMENDED_STEVEN={best[2]:.4f}")
