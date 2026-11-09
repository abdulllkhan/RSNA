"""Task 3: the public reader's honest OOF.

Each study of reader-fold k in {0, 1, 2} is predicted ONLY by checkpoint cnxt_v0_fold<k> (the one that did not train
on it); the 58 gold studies (never trained on by any fold) by the mean of the three checkpoints' probabilities
(as the author's oof.py does). Studies of reader folds 3 and 4 are trained on by all three public checkpoints and are
left out. Writes a NEW file (never touches existing OOF files):
  primary  (released infer.py, k=12):         data/oof/pubreader_gpjw_oof.parquet
  secondary (0.954 notebook, k=16+fallback):  outputs/pub_reader/pubreader_gpjw_k16fb_oof.parquet
plus a provenance table outputs/pub_reader/pubreader_gpjw_oof_provenance.csv (study -> reader fold / source).

Usage: PYTHONPATH=$PWD .venv/bin/python -P dev/pub_reader/build_oof.py [--variant k12]
"""
import argparse
from pathlib import Path

import pandas as pd

from dev.pub_reader.common import ROOT, T, gold58, reader_folds, reader_pred

OUT = {"k12": ROOT / "data/oof/pubreader_gpjw_oof.parquet",
       "k16fb": ROOT / "outputs/pub_reader/pubreader_gpjw_k16fb_oof.parquet"}


def build(variant):
    P = {k: reader_pred(variant, k) for k in (0, 1, 2)}
    folds = reader_folds("stable")
    g = gold58()
    assert set(folds.index[folds == -1]) == set(g.index), "gold set != reader fold -1"
    parts, prov = [], []
    for k in (0, 1, 2):
        ids = folds.index[folds == k]
        parts.append(P[k].loc[ids])
        prov.append(pd.DataFrame({"StudyInstanceUID": ids, "reader_fold": k, "source": f"cnxt_v0_fold{k}"}))
    parts.append(sum(P[k].loc[g.index] for k in (0, 1, 2)) / 3)
    prov.append(pd.DataFrame({"StudyInstanceUID": g.index, "reader_fold": -1, "source": "mean(fold0,fold1,fold2)"}))
    oof = pd.concat(parts)[T].astype("float32")
    oof.index.name = "StudyInstanceUID"
    assert oof.index.is_unique and len(oof) == 2610 + 58 and oof.notna().all().all()
    return oof, pd.concat(prov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="k12")
    ap.add_argument("--out", default=None, help="override the output path (tests only)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT[a.variant]
    if out.exists():
        raise SystemExit(f"{out} exists - refusing to overwrite (delete it deliberately first)")
    oof, prov = build(a.variant)
    oof.to_parquet(out)
    if a.variant == "k12" and not a.out:
        prov.to_csv(ROOT / "outputs/pub_reader/pubreader_gpjw_oof_provenance.csv", index=False)
    print(f"wrote {out}: {len(oof)} studies ({(prov.reader_fold >= 0).sum()} held-out + "
          f"{(prov.reader_fold == -1).sum()} gold), columns {list(oof.columns)}")


if __name__ == "__main__":
    main()
