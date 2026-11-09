"""Lay out training studies as a fake competition TEST mount (the layout the notebook reads on Kaggle):
  <out>/test.csv            StudyInstanceUID only (as the real test.csv)
  <out>/test_series.csv     the studies' rows of train_series.csv, same 5 columns as the real test_series.csv
                            (StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression, Anatomical_Plane)
  <out>/test_series/<study> symlink -> data/raw/train_series/<study>
Options: --ids FILE (one study per line) or --studies N --seed S (random, fixed seed);
         --reader-invalid K: the LAST K studies get Anatomical_Plane="Oblique" on every series row -- the reader cannot
         place them (0 valid slots -> invalid for its guard) while the other arms still read them (they map an unknown
         plane to Sagittal). A test-only trigger for the reader's per-row fallback / guard.
"""
import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--studies", type=int, default=30)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--reader-invalid", type=int, default=0)
    a = ap.parse_args()
    se = pd.read_csv(ROOT / "data/comp_csv/train_series.csv")
    if a.ids:
        ids = [l.strip() for l in open(a.ids) if l.strip()]
    else:
        ids = pd.Series(se.StudyInstanceUID.drop_duplicates().tolist()).sample(a.studies, random_state=a.seed).tolist()
    out = Path(a.out)
    (out / "test_series").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(out / "test.csv", index=False)
    sub = se[se.StudyInstanceUID.isin(set(ids))].copy()
    bad = set(ids[len(ids) - a.reader_invalid:]) if a.reader_invalid else set()
    sub.loc[sub.StudyInstanceUID.isin(bad), "Anatomical_Plane"] = "Oblique"
    sub.to_csv(out / "test_series.csv", index=False)
    for st in ids:
        link = out / "test_series" / st
        if not link.exists():
            os.symlink(ROOT / "data/raw/train_series" / st, link)
    print(f"fake mount {out}: {len(ids)} studies, {len(sub)} series rows, reader-invalid {len(bad)}")
    (out / "ids.txt").write_text("\n".join(ids) + "\n")


if __name__ == "__main__":
    main()
