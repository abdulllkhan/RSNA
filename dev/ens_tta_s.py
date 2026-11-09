"""Ensemble-level slot-phase TTA check: hr_tta3=3 + v2 (equal average of the given arms) at v2 TTA S=1/3/5, using the
per-variant frames dev/tta_eval_v2.py caches under data/oof/tta_v2/<arm>_v<k>_fold<f>.parquet (variant rank-average per
fold, folds stitched). Usage: .venv/bin/python dev/ens_tta_s.py v2_effv2s384_lr v2_effv2s384_c110 [--w 10]"""
import sys, glob, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
w=float(sys.argv[sys.argv.index("--w")+1]) if "--w" in sys.argv else 10.0; _skip={sys.argv.index("--w")+1} if "--w" in sys.argv else set(); arms=[a for i,a in enumerate(sys.argv[1:],1) if not a.startswith("--") and i not in _skip]
def rd(p):
    d=pd.read_parquet(p); d=d.set_index("StudyInstanceUID") if "StudyInstanceUID" in d.columns else d; d.index=d.index.astype(str); return d[T]
def stitch_tta(arm,S):
    parts=[]
    for f in range(5):
        fr=[rd(f"data/oof/tta_v2/{arm}_v{k}_fold{f}.parquet") for k in range(S)]
        r=sum(x.rank(pct=True) for x in fr)/S; parts.append(r)
    d=pd.concat(parts); return d[~d.index.duplicated()]
def stitch(name):
    fs=sorted(glob.glob(f"data/oof/{name}_fold[0-9].parquet")); d=pd.concat([rd(f) for f in fs]); return d[~d.index.duplicated()]
hr=stitch("effv2s_k24_384_tta3")
sv=pd.read_csv("data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID"); sv.index=sv.index.astype(str)
tr=pd.read_csv("data/comp_csv/train.csv").set_index("StudyInstanceUID"); tr.index=tr.index.astype(str); gold=tr.dropna(subset=T)[T]
def macro(P,Y): return np.mean([roc_auc_score(Y[c],P[c]) for c in T if Y[c].nunique()>1])
for S in (1,3,5):
    try: V=[stitch_tta(a,S) for a in arms]
    except FileNotFoundError as e: print(f"S={S}: frames missing ({e})"); continue
    idx=hr.index
    for v in V: idx=idx.intersection(v.index)
    rv=sum(v.loc[idx] for v in V)/len(V); rh=hr.loc[idx].rank(pct=True); E=(3*rh+w*rv)/(3+w)
    ys=(sv.loc[idx,T]>0.5).astype(int); g=idx.intersection(gold.index); yg=gold.loc[g].astype(int)
    print(f"S={S}: v2 arm alone steven {macro(rv,ys):.4f} gold58 {macro(rv.loc[g],yg):.4f} | ensemble (hr=3, v2={w:g}) steven {macro(E,ys):.4f} gold58 {macro(E.loc[g],yg):.4f}  (n={len(idx)})")
