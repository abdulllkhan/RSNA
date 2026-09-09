#!/usr/bin/env python
"""A5 step 1 — extract the 4,407 free-text reports into a compact working table.

Writes `data/a5/reports.parquet` with:
    StudyInstanceUID, Report, lang, n_chars, n_words,
    kw_syn, kw_eff, kw_neg_syn, is_gold, gold_synovitis, steven_v4_synovitis

and prints
  * the language distribution,
  * the Synovitis-keyword hit-rate per language (regex over synovit*/sinovit*/
    synovial*/Synovialitis/sinovyal ... plus the effusion decoys),
  * how the keyword alone lines up with the 58 gold labels and with steven_v4.

No network, no GPU, read-only on the competition CSVs.

Usage:
    .venv/bin/python dev/a5/extract_reports.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from langid import guess_lang, normalize_for_match  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TRAIN_CSV = REPO / "data" / "comp_csv" / "train.csv"
STEVEN_V4 = (
    REPO
    / "data"
    / "labels"
    / "stevenleehans_rsna-knee-llm-report-labels"
    / "llm_labels_v4_blend.csv"
)
OUT_PARQUET = REPO / "data" / "a5" / "reports.parquet"

# ---------------------------------------------------------------- regexes
# Accent-stripped, lowercased text is matched, so `sinovítis` -> `sinovitis`.
# Covers: synovitis/sinovitis (es,pt,it,hr), synovitis (en,nl), Synovialitis/
# Synovitis (de), sinovit/sinovyal (tr), synoviaal/synoviale (nl),
# synovial thickening / proliferation / hypertrophy (en) and its translations,
# plus the non-latin scripts: синовит (bg/ru), υμενίτιδα / αρθρικ. υμέν* (el),
# błona maziowa / maziówka (pl).
RE_SYN = re.compile(
    r"sinovi|synovi|sinovy|synovy|pannus|"          # latin scripts
    r"синовит|синовиал|синовиальн|"                   # bg / ru
    r"υμενιτ|υμενίτ|υμενα|υμένα|υμενο|υμένο|"          # el (arthrikos ymenas)
    r"maziow|maziówk",                                 # pl
    re.I,
)

# Effusion decoys — a report saying only "effusion" must NOT count as synovitis.
RE_EFF = re.compile(
    r"effusion|derrame|erguss|ergu[sß]|vocht|hydrops|epanchement|efuzyon|"
    r"efüzyon|versamento|joint fluid|liquido articular|gelenkerguss|"
    r"sivi kolek|sıvı|izljev|izliv|wysiek|wysięk|plyn w stawie|"
    r"излив|выпот|συλλογη|συλλογή",
    re.I,
)

# Crude negation window: a negation cue within ~40 chars before a synovial term.
RE_NEG_SYN = re.compile(
    r"(?:\bno\b|\bnot\b|\bsin\b|\bgeen\b|\bkein\w*\b|\bnicht\b|\bausschluss\b|"
    r"\bnegativ\w*\b|\bpas de\b|\bsans\b|\bsenza\b|\bnon\b|\bausencia\b|"
    r"\bsem\b|\byok\b|\bizlenmemis\w*\b|\bwithout\b|\bfree of\b|\bnegative for\b)"
    r"[^.;\n]{0,40}?(?:sinovi|synovi|sinovy|synovy)",
    re.I,
)


def build(limit: int | None = None) -> pd.DataFrame:
    if not TRAIN_CSV.exists():
        raise SystemExit(f"missing {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)
    if limit:
        df = df.head(limit).copy()

    out = pd.DataFrame(
        {
            "StudyInstanceUID": df["StudyInstanceUID"].astype(str),
            "Report": df["Report"].fillna("").astype(str),
        }
    )
    out["lang"] = out["Report"].map(guess_lang)
    out["n_chars"] = out["Report"].str.len().astype("int32")
    out["n_words"] = out["Report"].str.split().map(len).astype("int32")

    norm = out["Report"].map(normalize_for_match)
    out["kw_syn"] = norm.str.contains(RE_SYN)
    out["kw_eff"] = norm.str.contains(RE_EFF)
    out["kw_neg_syn"] = norm.str.contains(RE_NEG_SYN)

    # gold (58 rows) — validation anchor only, never a model input
    out["is_gold"] = df["Synovitis"].notna().to_numpy()
    out["gold_synovitis"] = df["Synovitis"].to_numpy()

    if STEVEN_V4.exists():
        sv = pd.read_csv(STEVEN_V4)[["StudyInstanceUID", "Synovitis"]]
        sv = sv.rename(columns={"Synovitis": "steven_v4_synovitis"})
        sv["StudyInstanceUID"] = sv["StudyInstanceUID"].astype(str)
        out = out.merge(sv, on="StudyInstanceUID", how="left")
    else:
        out["steven_v4_synovitis"] = pd.NA
    return out


def report(out: pd.DataFrame) -> None:
    n = len(out)
    print(f"\n=== A5 report corpus: {n} studies ===")
    print(
        f"report length chars: mean {out.n_chars.mean():.0f}  "
        f"median {out.n_chars.median():.0f}  p90 {out.n_chars.quantile(0.9):.0f}  "
        f"max {out.n_chars.max()}"
    )
    print(
        f"report length words: mean {out.n_words.mean():.0f}  "
        f"median {out.n_words.median():.0f}  p90 {out.n_words.quantile(0.9):.0f}"
    )

    print("\n--- language distribution (heuristic guess) ---")
    lang = out.groupby("lang", dropna=False)
    tbl = pd.DataFrame(
        {
            "n": lang.size(),
            "share": lang.size() / n,
            "mean_chars": lang["n_chars"].mean().round(0),
            "kw_syn_rate": lang["kw_syn"].mean(),
            "kw_eff_rate": lang["kw_eff"].mean(),
            "kw_neg_syn_rate": lang["kw_neg_syn"].mean(),
            "steven_v4_mean": lang["steven_v4_synovitis"].mean(),
        }
    ).sort_values("n", ascending=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(tbl.to_string())

    print("\n--- Synovitis keyword hit-rate, whole corpus ---")
    print(f"  kw_syn      : {out.kw_syn.mean():.4f}  ({int(out.kw_syn.sum())} studies)")
    print(f"  kw_eff      : {out.kw_eff.mean():.4f}  ({int(out.kw_eff.sum())} studies)")
    print(
        f"  kw_neg_syn  : {out.kw_neg_syn.mean():.4f}  "
        f"({int(out.kw_neg_syn.sum())} studies) <- negated mentions"
    )
    eff_only = out.kw_eff & ~out.kw_syn
    print(
        f"  effusion-but-no-synovial-term: {eff_only.mean():.4f} "
        f"({int(eff_only.sum())}) <- the main confusable class"
    )

    if out["steven_v4_synovitis"].notna().any():
        print("\n--- steven_v4 Synovitis vs the keyword ---")
        s = out["steven_v4_synovitis"]
        print(f"  mean steven_v4 | kw_syn=True : {s[out.kw_syn].mean():.3f}")
        print(f"  mean steven_v4 | kw_syn=False: {s[~out.kw_syn].mean():.3f}")
        print(f"  steven_v4 > 0.5 rate         : {(s > 0.5).mean():.4f}")
        print(
            f"  steven_v4 > 0.5 AND no synovial term: "
            f"{int(((s > 0.5) & ~out.kw_syn).sum())} studies"
        )
        print(
            f"  steven_v4 <= 0.5 AND synovial term  : "
            f"{int(((s <= 0.5) & out.kw_syn).sum())} studies"
        )

    g = out[out.is_gold]
    if len(g):
        print(f"\n--- the {len(g)} gold studies (validation anchor) ---")
        print(f"  gold Synovitis positives: {int(g.gold_synovitis.sum())}/{len(g)}")
        ct = pd.crosstab(g.gold_synovitis, g.kw_syn)
        print("  crosstab gold x kw_syn:")
        print(ct.to_string().replace("\n", "\n    "))
        tp = int(((g.gold_synovitis == 1) & g.kw_syn).sum())
        fp = int(((g.gold_synovitis == 0) & g.kw_syn).sum())
        fn = int(((g.gold_synovitis == 1) & ~g.kw_syn).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        print(f"  keyword-alone precision {prec:.3f}  recall {rec:.3f}")
        print(
            "  -> the reports are often SILENT on the synovium; the LLM read has to "
            "\n     harvest the implied/indirect statements the plain regex misses."
        )
        print("\n  gold language mix:")
        print("    " + g.lang.value_counts().to_string().replace("\n", "\n    "))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_PARQUET))
    ap.add_argument("--limit", type=int, default=None, help="debug: first N rows only")
    args = ap.parse_args()

    out = build(args.limit)
    report(out)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    print(f"\nwrote {dest}  ({len(out)} rows, {dest.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
