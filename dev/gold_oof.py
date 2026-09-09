"""Honest validation: stitch per-fold OOF preds and score (a) vs the 58 GOLD studies (the only
true labels) and (b) vs binarized steven_v4 over all studies (self-consistency proxy)."""
import glob, sys, numpy as np, pandas as pd
NAME = sys.argv[1] if len(sys.argv) > 1 else "effv2s_k24"   # e.g. effv2s_k24_384
from sklearn.metrics import roc_auc_score
T = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA",
     "Effusion","Synovitis","Baker's","Contusion","Fracture"]
tr = pd.read_csv("data/comp_csv/train.csv")
gold = tr.dropna(subset=T)[["StudyInstanceUID"]+T].set_index("StudyInstanceUID")
oofs = []
print(f"== scoring OOF for name={NAME} ==")
for f in sorted(glob.glob(f"data/oof/{NAME}_fold[0-9].parquet")):
    d = pd.read_parquet(f)
    if "StudyInstanceUID" not in d.columns: d = d.reset_index()
    idc = [c for c in d.columns if "Study" in c][0]; d = d.set_index(idc)
    pc = [c for c in T if c in d.columns]
    print(f"{f}: shape={d.shape} pred_cols={len(pc)} cols={list(d.columns)[:6]}...")
    oofs.append(d[pc])
oof = pd.concat(oofs); oof = oof[~oof.index.duplicated()]
print("OOF studies:", len(oof))
g = gold.join(oof, rsuffix="_p", how="inner")
aucs = {t: roc_auc_score(g[t], g[t+"_p"]) for t in T if g[t].nunique() > 1}
print(f"\nGOLD macro-AUC (n={len(g)} gold covered): {np.mean(list(aucs.values())):.4f}")
for t, a in sorted(aucs.items(), key=lambda x: x[1]): print(f"  {t:16s} {a:.3f}")
sv = pd.read_csv("data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")
j = sv.join(oof, rsuffix="_p", how="inner")
a2 = {t: roc_auc_score((j[t] > 0.5).astype(int), j[t+"_p"]) for t in T}
print(f"\nOOF macro-AUC vs binarized steven_v4 (n={len(j)}): {np.mean(list(a2.values())):.4f}")
