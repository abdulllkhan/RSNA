"""Fold-0 test for a candidate arm X against the OLD-CONTRACT hr arm: baseline hr=3 + lr=5 + c110=5 (rank-avg, steven
labels) vs X replacing hr (X=3) vs X added as a 4th arm (X=3). Usage: python dev/fold0_hr_swap.py data/oof/<X>_fold0.parquet
Prints REPLACE_DELTA=<f> and ADD_DELTA=<f> (steven macro deltas vs baseline) for the queue gate."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
def rd(p):
    d=pd.read_parquet(p); d=d.set_index("StudyInstanceUID") if "StudyInstanceUID" in d.columns else d; d.index=d.index.astype(str); return d[T]
X=rd(sys.argv[1]); hr=rd("data/oof/effv2s_k24_384_tta3_fold0.parquet"); lr=rd("data/oof/v2_effv2s384_lr_fold0.parquet"); c110=rd("data/oof/v2_effv2s384_c110_fold0.parquet")
idx=X.index.intersection(hr.index).intersection(lr.index).intersection(c110.index)
sv=pd.read_csv("data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID"); sv.index=sv.index.astype(str); ys=(sv.loc[idx,T]>0.5).astype(int)
tr=pd.read_csv("data/comp_csv/train.csv").set_index("StudyInstanceUID"); tr.index=tr.index.astype(str); gold=tr.dropna(subset=T)[T]; g=idx.intersection(gold.index); yg=gold.loc[g].astype(int)
R={k:v.loc[idx].rank(pct=True) for k,v in dict(X=X,hr=hr,lr=lr,c110=c110).items()}
def macro(P,Y): return float(np.mean([roc_auc_score(Y[c],P[c]) for c in T if Y[c].nunique()>1]))
def ens(parts):
    num=sum(w*R[k] for k,w in parts); den=sum(w for _,w in parts); return num/den
base=ens([("hr",3),("lr",5),("c110",5)]); rep=ens([("X",3),("lr",5),("c110",5)]); add=ens([("hr",3),("X",3),("lr",5),("c110",5)])
print(f"n={len(idx)} gold={len(g)}")
for name,E in [("X solo",R["X"]),("hr solo",R["hr"]),("baseline hr+lr+c110",base),("X replaces hr",rep),("X added (4-arm)",add)]:
    print(f"  {name:22s} steven {macro(E,ys):.4f} gold {macro(E.loc[g],yg) if len(g)>1 else float('nan'):.4f}")
print(f"REPLACE_DELTA={macro(rep,ys)-macro(base,ys):.4f}\nADD_DELTA={macro(add,ys)-macro(base,ys):.4f}")
