"""Fold-0 ensemble fit for a v2 member: rank-avg(base_tta3=1, hr_tta3=3, v2=w) on the fold-0 studies.
Usage: .venv/bin/python dev/fit_v2_fold0.py v2_effv2s384_lr [more v2 names...]  (each name -> data/oof/<name>_fold0.parquet)
Prints per-member solo numbers, per-column deltas vs HR fold-0, and the ensemble grid (steven / consensus / gold-in-fold)."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
def load(n):
    d=pd.read_parquet(f"data/oof/{n}_fold0.parquet")
    if "StudyInstanceUID" in d.columns: d=d.set_index("StudyInstanceUID")
    return d[T]
names=sys.argv[1:]; V={n: load(n) for n in names}
hr=load("effv2s_k24_384"); hr3=load("effv2s_k24_384_tta3"); b3=load("effv2s_k24_tta3")
idx=hr.index
for n in names: idx=idx.intersection(V[n].index)
sv=pd.read_csv("data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID").loc[idx]
cons=pd.read_parquet("data/labels/consensus_v1_nogold.parquet").loc[idx]
tr=pd.read_csv("data/comp_csv/train.csv").set_index("StudyInstanceUID"); gold=tr.dropna(subset=T)[T]; g=idx.intersection(gold.index)
ys=(sv>0.5).astype(int); yc=(cons[T]>0.5).astype(int); yg=gold.loc[g].astype(int)
def macro(P,Y): return np.mean([roc_auc_score(Y[c],P[c]) for c in T if Y[c].nunique()>1])
def percol(P,Y): return {c: roc_auc_score(Y[c],P[c]) for c in T if Y[c].nunique()>1}
print(f"n={len(idx)} gold-in-fold={len(g)}\n{'member':<26} steven  consensus gold")
for n,P in [("hr 384 f0",hr),("hr 384 f0 tta3",hr3)]+[(n,V[n]) for n in names]:
    print(f"{n:<26} {macro(P.loc[idx],ys):.4f}  {macro(P.loc[idx],yc):.4f}   {macro(P.loc[g],yg):.4f}")
b=percol(hr.loc[idx],ys)
for n in names:
    a=percol(V[n].loc[idx],ys); print(f"{n} - hr per column (steven):", {c: round(a[c]-b[c],3) for c in a})
rb=b3.loc[idx].rank(pct=True); rh=hr3.loc[idx].rank(pct=True)
print("\nensemble base_tta3=1, hr_tta3=3, + each v2 member at w:")
for n in names:
    rv=V[n].loc[idx].rank(pct=True)
    for w in [0,1,2,3,4,6,8]:
        E=(rb+3*rh+w*rv)/(4+w); print(f"  {n:<26} w={w}: steven {macro(E,ys):.4f}  consensus {macro(E,yc):.4f}  gold {macro(E.loc[g],yg):.4f}")
if len(names)>1:
    R=sum(V[n].loc[idx].rank(pct=True) for n in names)/len(names)
    for w in [2,4,6]:
        E=(rb+3*rh+w*R)/(4+w); print(f"  all-v2-mean w={w}: steven {macro(E,ys):.4f}  consensus {macro(E,yc):.4f}  gold {macro(E.loc[g],yg):.4f}")
