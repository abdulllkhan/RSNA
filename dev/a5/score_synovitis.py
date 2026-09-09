#!/usr/bin/env python
"""A5 step 4 — did the LLM relabel actually improve the Synovitis column?

Measures, for the Synovitis column ONLY:
  1. gold anchor (the 58 studies with real labels in train.csv):
     ROC-AUC for steven_v4 alone, the LLM alone, the report keyword baseline, and
     rank-blends at w = 0.25 / 0.5 / 0.75, plus a paired bootstrap CI on the delta.
  2. corpus-wide agreement between the LLM and steven_v4 over all 4,407 studies
     (Spearman/Pearson, hard agreement @0.5, positive rates, confusion).
  3. the biggest disagreements, with the LLM's quoted evidence, so the direction of
     the disagreement can be eyeballed rather than trusted.

VALIDATION RULE (non-negotiable, see reports/label_foundation.md §3):
truth is ONLY `train.csv`'s 58 gold cells.  `yunus_v5` and the merged `__soft`
column leak those gold labels — this script refuses to load them as truth.

    .venv/bin/python dev/a5/score_synovitis.py --llm-csv data/a5/synovitis_llm.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[2]
TRAIN_CSV = REPO / "data" / "comp_csv" / "train.csv"
STEVEN_V4 = (
    REPO / "data" / "labels" / "stevenleehans_rsna-knee-llm-report-labels"
    / "llm_labels_v4_blend.csv"
)
REPORTS_PARQUET = REPO / "data" / "a5" / "reports.parquet"
COL = "Synovitis"

# any path matching these is a gold-leaking source and may never be truth
BANNED = ("yunus", "merged_labels", "__soft")


def _guard(path: Path, role: str) -> None:
    low = str(path).lower()
    for b in BANNED:
        if b in low:
            raise SystemExit(
                f"REFUSED: '{path}' looks like a gold-leaking source ({b}) and cannot "
                f"be used as {role}. Truth is train.csv's 58 gold cells only "
                f"(reports/label_foundation.md §3)."
            )


def pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2 or len(y) < 4:
        return float("nan")
    return float(roc_auc_score(y, p))


def boot_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int, seed: int = 0):
    """Paired bootstrap on AUC(b) - AUC(a).  n=58 is small; this says how small."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    deltas = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        try:
            deltas.append(roc_auc_score(y[s], b[s]) - roc_auc_score(y[s], a[s]))
        except ValueError:
            continue
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    d = np.asarray(deltas)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm-csv", default=str(REPO / "data" / "a5" / "synovitis_llm.csv"))
    ap.add_argument("--steven", default=str(STEVEN_V4))
    ap.add_argument("--reports", default=str(REPORTS_PARQUET))
    ap.add_argument("--weights", default="0.25,0.5,0.75",
                    help="LLM weights for the rank blend")
    ap.add_argument("--top-disagree", type=int, default=15)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--write-blend", default=None,
                    help="optional: write the chosen blend as a Synovitis label CSV")
    ap.add_argument("--blend-w", type=float, default=0.5,
                    help="LLM weight used by --write-blend")
    args = ap.parse_args()

    llm_path = Path(args.llm_csv)
    steven_path = Path(args.steven)
    _guard(steven_path, "the steven_v4 reference")
    if not llm_path.exists():
        raise SystemExit(f"missing {llm_path} — run dev/a5/run_relabel.py first")

    train = pd.read_csv(TRAIN_CSV, usecols=["StudyInstanceUID", "Report", COL])
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    train = train.rename(columns={COL: "gold"})

    steven = pd.read_csv(steven_path)[["StudyInstanceUID", COL]]
    steven["StudyInstanceUID"] = steven["StudyInstanceUID"].astype(str)
    steven = steven.rename(columns={COL: "steven_v4"})

    llm = pd.read_csv(llm_path)
    llm["StudyInstanceUID"] = llm["StudyInstanceUID"].astype(str)
    keep = ["StudyInstanceUID", "synovitis_llm"] + [
        c for c in ("evidence", "llm_language") if c in llm.columns
    ]
    llm = llm[keep].drop_duplicates("StudyInstanceUID", keep="last")
    for c in ("evidence", "llm_language"):
        if c in llm.columns:
            llm[c] = llm[c].fillna("").astype(str)

    df = train.merge(steven, on="StudyInstanceUID", how="left").merge(
        llm, on="StudyInstanceUID", how="left"
    )
    if Path(args.reports).exists():
        rp = pd.read_parquet(args.reports)[["StudyInstanceUID", "lang", "kw_syn"]]
        rp["StudyInstanceUID"] = rp["StudyInstanceUID"].astype(str)
        df = df.merge(rp, on="StudyInstanceUID", how="left")
    else:
        df["lang"] = "?"
        df["kw_syn"] = False

    n_all = len(df)
    cov = df["synovitis_llm"].notna()
    print(f"\n=== A5 Synovitis scoring ===")
    print(f"corpus {n_all} studies | LLM coverage {int(cov.sum())} "
          f"({cov.mean():.1%}) | steven_v4 coverage {int(df.steven_v4.notna().sum())}")
    if cov.sum() < n_all:
        print(f"[note] {n_all - int(cov.sum())} studies have no LLM label yet "
              f"(partial/smoke run) — corpus stats below use the covered subset.")

    # ---------------------------------------------------------- ranks + blends
    sub = df[cov].copy()
    sub["r_steven"] = pct_rank(sub["steven_v4"])
    sub["r_llm"] = pct_rank(sub["synovitis_llm"])
    weights = [float(w) for w in args.weights.split(",") if w.strip()]
    for w in weights:
        sub[f"blend_{w:g}"] = w * sub["r_llm"] + (1 - w) * sub["r_steven"]

    # ------------------------------------------------------------ gold anchor
    gold = sub[sub["gold"].notna()].copy()
    print(f"\n--- GOLD ANCHOR: {len(gold)} of the 58 gold studies covered "
          f"({int(gold.gold.sum())} positive / {int((gold.gold == 0).sum())} negative) ---")
    if len(gold) < 10 or gold["gold"].nunique() < 2:
        print("  too few covered gold rows for a meaningful AUC (smoke run?) — "
              "run the full corpus, or use --order gold-first for a quick read.")
    else:
        y = gold["gold"].to_numpy(float)
        cands: list[tuple[str, np.ndarray]] = [
            ("steven_v4 (baseline)", gold["steven_v4"].to_numpy(float)),
            ("LLM alone", gold["synovitis_llm"].to_numpy(float)),
            ("keyword regex only", gold["kw_syn"].fillna(False).astype(float).to_numpy()),
        ]
        for w in weights:
            cands.append((f"rank-blend w_llm={w:g}", gold[f"blend_{w:g}"].to_numpy(float)))

        base = cands[0][1]
        base_auc = safe_auc(y, base)
        print(f"\n  {'model':<26} {'AUC':>7}   {'delta vs steven_v4':>20}")
        print("  " + "-" * 58)
        for name, p in cands:
            a = safe_auc(y, p)
            if name.startswith("steven_v4"):
                print(f"  {name:<26} {a:>7.4f}   {'—':>20}")
                continue
            m, lo, hi = boot_delta(y, base, p, args.bootstrap)
            print(f"  {name:<26} {a:>7.4f}   {a - base_auc:+.4f}  "
                  f"[boot 95% {lo:+.3f},{hi:+.3f}]")
        print(f"\n  n={len(gold)} -> AUC standard error is roughly +/-0.07; only a "
              f"delta whose bootstrap CI clears 0 is real.")

        # per-language gold read (tiny cells, printed as a hint only)
        if "lang" in gold.columns:
            print("\n  gold AUC by language (n<10 cells are noise, shown for direction):")
            for lg, g in gold.groupby("lang"):
                if len(g) < 4 or g["gold"].nunique() < 2:
                    continue
                print(f"    {lg:<4} n={len(g):<3} steven {safe_auc(g.gold, g.steven_v4):.3f}  "
                      f"llm {safe_auc(g.gold, g.synovitis_llm):.3f}")

    # ------------------------------------------------- corpus-wide agreement
    print(f"\n--- CORPUS AGREEMENT (LLM vs steven_v4, n={len(sub)}) ---")
    s, l = sub["steven_v4"], sub["synovitis_llm"]
    print(f"  positive rate @0.5   steven_v4 {(s > 0.5).mean():.4f}   "
          f"LLM {(l > 0.5).mean():.4f}")
    print(f"  mean probability     steven_v4 {s.mean():.4f}   LLM {l.mean():.4f}")
    print(f"  pearson  {s.corr(l):.4f}    spearman {s.corr(l, method='spearman'):.4f}")
    hs, hl = (s > 0.5), (l > 0.5)
    print(f"  hard agreement @0.5  {(hs == hl).mean():.4f}")
    print(f"    both pos {int((hs & hl).sum()):>5}   both neg {int((~hs & ~hl).sum()):>5}")
    print(f"    LLM pos / steven neg {int((hl & ~hs).sum()):>5}   "
          f"(the LLM finds synovitis steven missed)")
    print(f"    LLM neg / steven pos {int((~hl & hs).sum()):>5}   "
          f"(the LLM rejects a steven positive)")
    if "kw_syn" in sub.columns:
        k = sub["kw_syn"].fillna(False)
        print(f"  vs the plain keyword: LLM pos-rate | kw hit {l[k].gt(0.5).mean():.3f} | "
              f"kw miss {l[~k].gt(0.5).mean():.3f}")
    if "evidence" in sub.columns:
        ev = sub["evidence"].fillna("")
        print(f"  LLM returned a quoted span on {(ev.str.len() > 0).mean():.1%} of studies")

    # ------------------------------------------------------- top disagreements
    if args.top_disagree:
        sub["disagree"] = (sub["r_llm"] - sub["r_steven"]).abs()
        top = sub.nlargest(args.top_disagree, "disagree")
        print(f"\n--- TOP {len(top)} DISAGREEMENTS (by rank gap) ---")
        for _, r in top.iterrows():
            tag = "GOLD=" + ("1" if r.gold == 1 else "0") if pd.notna(r.gold) else "     "
            ev_raw = r.get("evidence")
            ev = ("" if pd.isna(ev_raw) else str(ev_raw))[:70].replace("\n", " ")
            rep = " ".join(str(r.Report).split())[:90]
            print(
                f"  {r.StudyInstanceUID[-12:]}  {tag}  steven {r.steven_v4:.2f} -> "
                f"llm {r.synovitis_llm:.2f}  [{r.get('lang', '?')}]"
            )
            print(f"      evidence: {ev!r}")
            print(f"      report  : {rep}...")

    # --------------------------------------------------------- optional blend
    if args.write_blend:
        w = args.blend_w
        out = df[["StudyInstanceUID"]].copy()
        full = df.copy()
        full["r_steven"] = pct_rank(full["steven_v4"])
        full["r_llm"] = pct_rank(full["synovitis_llm"])
        # studies the LLM never saw fall back to steven's rank
        blend = np.where(
            full["r_llm"].notna(),
            w * full["r_llm"].fillna(0) + (1 - w) * full["r_steven"],
            full["r_steven"],
        )
        out[COL] = blend
        dest = Path(args.write_blend)
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dest, index=False)
        print(f"\nwrote blended Synovitis column (w_llm={w:g}) -> {dest}")
        print("  NOTE: this is a percentile rank in [0,1], which is what a rank-based "
              "AUC target wants; rescale if the trainer needs a probability.")


if __name__ == "__main__":
    main()
