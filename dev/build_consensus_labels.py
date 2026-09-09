#!/usr/bin/env python
"""Build the three-teacher soft-label consensus (public recipe, prvsiyan cell 50).

Teachers (the three SHA-pinned files every 0.93+ public notebook shares --
see reports/public_solutions.md 2(b)):

    pilkwang   data/labels/pilkwang_rsna-knee-llm-labels/report_labels_v2.csv
    steven     data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v2.csv
    lixin73    data/labels/lixin73_rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv

`yunus` is NEVER loaded: it reproduces the 58 gold labels exactly
(reports/label_foundation.md 3) and would poison both target and ruler.

Recipe, verbatim from the survey:

    consensus    = np.nanmean(cube, axis=0)                       # cube = 3 teachers x studies x 12
    disagreement = np.nanmean(np.abs(cube - consensus[None]), axis=0)
    agreement    = np.clip(1.0 - 2.0*disagreement, 0.0, 1.0)
    certainty    = np.clip(2.0*np.abs(consensus - 0.5), 0.0, 1.0)
    w = 0.15 + 0.85*(0.65*agreement + 0.35*certainty)
    y[gold] = train.loc[gold, TARGETS];  w[gold] = 3.0

Outputs
    data/labels/consensus_v1.parquet         gold rows overwritten with the gold label, w = --gold-weight
    data/labels/consensus_v1_nogold.parquet  gold rows keep the consensus y/w (ablation)

Run:  nice -n 19 .venv/bin/python dev/build_consensus_labels.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

UID = "StudyInstanceUID"

# --- teacher / candidate sources -------------------------------------------------
PILKWANG = ROOT / "data/labels/pilkwang_rsna-knee-llm-labels/report_labels_v2.csv"
STEVEN_V2 = ROOT / "data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v2.csv"
STEVEN_V4 = ROOT / "data/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
LIXIN_HARD = ROOT / "data/labels/lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv"
LIXIN_REPORT = ROOT / "data/labels/lixin73_rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv"

TRAIN_CSV = ROOT / "data/comp_csv/train.csv"
FOLDS_CSV = ROOT / "data/folds.csv"

# leak thresholds
LEAK_AUC = 0.99          # macro gold AUC above this == leaking
LEAK_EXACT = 0.99        # fraction of gold cells reproduced exactly == leaking


# ---------------------------------------------------------------------------------
def load_targets(path: Path, index: pd.Index, cols=None) -> pd.DataFrame:
    """Read a label csv, reindex onto the canonical 4,407 study order."""
    df = pd.read_csv(path)
    df = df.set_index(UID)
    take = list(cols) if cols is not None else TARGETS
    out = df[take].reindex(index).astype("float64")
    out.columns = TARGETS
    return out


def temper(y: pd.DataFrame, conf: pd.DataFrame) -> pd.DataFrame:
    """0.5 + (y-0.5)*conf -- pull a hard 0/1 label toward 0.5 by its confidence."""
    return 0.5 + (y - 0.5) * conf.to_numpy()


def per_col_auc(pred: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """AUC per target on the gold rows only; None where a column is single-class."""
    out = {}
    for t in TARGETS:
        yt = gold[t].to_numpy()
        yp = pred.loc[gold.index, t].to_numpy()
        ok = ~np.isnan(yt) & ~np.isnan(yp)
        if ok.sum() < 2 or len(np.unique(yt[ok])) < 2:
            out[t] = None
        else:
            out[t] = float(roc_auc_score(yt[ok], yp[ok]))
    return out


def macro(d: dict) -> float:
    v = [x for x in d.values() if x is not None]
    return float(np.mean(v)) if v else float("nan")


def exact_gold_match(pred: pd.DataFrame, gold: pd.DataFrame) -> tuple[int, int, float]:
    """How many gold cells the source reproduces bit-for-bit (0.0 / 1.0)."""
    p = pred.loc[gold.index, TARGETS].to_numpy()
    g = gold[TARGETS].to_numpy()
    ok = ~np.isnan(p) & ~np.isnan(g)
    hit = int((np.isclose(p[ok], g[ok], atol=1e-9)).sum())
    return hit, int(ok.sum()), hit / max(int(ok.sum()), 1)


# ---------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-weight", type=float, default=3.0,
                    help="per-sample weight given to the 58 gold rows (public recipe: 3.0)")
    ap.add_argument("--out", type=Path, default=ROOT / "data/labels/consensus_v1.parquet")
    ap.add_argument("--out-nogold", type=Path,
                    default=ROOT / "data/labels/consensus_v1_nogold.parquet")
    ap.add_argument("--stats-json", type=Path, default=None,
                    help="optional dump of every number in the report")
    args = ap.parse_args()

    # ---- canonical index + gold ------------------------------------------------
    train = pd.read_csv(TRAIN_CSV).set_index(UID)
    index = train.index
    assert index.is_unique and len(index) == 4407, f"unexpected train index {len(index)}"
    gold_mask = train[TARGETS].notna().all(axis=1)
    gold = train.loc[gold_mask, TARGETS].astype("float64")
    print(f"[gold] {len(gold)} studies, pos/col: "
          f"{ {t: int(gold[t].sum()) for t in TARGETS} }")

    folds = pd.read_csv(FOLDS_CSV).set_index(UID)["fold"].reindex(index)
    assert folds.notna().all(), "folds.csv does not cover every study"

    # ---- load every candidate --------------------------------------------------
    pilk = load_targets(PILKWANG, index)
    pilk_conf = load_targets(PILKWANG, index, cols=[f"{t}__conf" for t in TARGETS])
    st_v2 = load_targets(STEVEN_V2, index)
    st_v4 = load_targets(STEVEN_V4, index)
    lix_hard = load_targets(LIXIN_HARD, index)
    lix_rep = load_targets(LIXIN_REPORT, index)
    lix_conf = load_targets(LIXIN_REPORT, index, cols=[f"{t}__conf" for t in TARGETS])
    lix_temp = temper(lix_rep, lix_conf)

    # what kind of values does each file hold?
    def kind(df: pd.DataFrame) -> str:
        v = df.to_numpy()
        v = v[~np.isnan(v)]
        n_uniq = len(np.unique(v))
        frac_bin = float(np.isin(v, [0.0, 1.0]).mean())
        tag = "HARD 0/1" if frac_bin == 1.0 else ("NEAR-HARD" if frac_bin > 0.5 else "SOFT prob")
        return (f"{tag} (n_uniq={n_uniq}, frac_exactly_0or1={frac_bin:.3f}, "
                f"min={v.min():.3f}, max={v.max():.3f})")

    value_kinds = {
        "pilkwang_v2": kind(pilk),
        "steven_v2": kind(st_v2),
        "steven_v4_blend": kind(st_v4),
        "lixin73_hard": kind(lix_hard),
        "lixin73_report": kind(lix_rep),
        "lixin73_report_conftempered": kind(lix_temp),
    }
    print("\n[value kinds]")
    for k, v in value_kinds.items():
        print(f"  {k:32s} {v}")
    same = np.array_equal(np.nan_to_num(lix_hard.to_numpy(), nan=-1),
                          np.nan_to_num(lix_rep.to_numpy(), nan=-1))
    print(f"  lixin73 hard == lixin73 report labels: {same}")
    cf = lix_conf.to_numpy()[~np.isnan(lix_conf.to_numpy())]
    lix_conf_desc = dict(mean=float(cf.mean()), p10=float(np.percentile(cf, 10)),
                         p50=float(np.percentile(cf, 50)), p90=float(np.percentile(cf, 90)),
                         min=float(cf.min()), max=float(cf.max()))
    print(f"  lixin73 __conf: {lix_conf_desc}")
    pk = pilk_conf.to_numpy()[~np.isnan(pilk_conf.to_numpy())]
    print(f"  pilkwang __conf: mean={pk.mean():.3f} p50={np.percentile(pk, 50):.3f}")

    # ---- 1. LEAK CHECK (every candidate, before anything else) -----------------
    candidates = {
        "pilkwang_v2": pilk,
        "steven_v2": st_v2,
        "steven_v4_blend": st_v4,
        "lixin73_hard": lix_hard,
        "lixin73_report": lix_rep,
    }
    leak = {}
    print("\n[leak check] vs the 58 gold")
    for name, df in candidates.items():
        aucs = per_col_auc(df, gold)
        m = macro(aucs)
        hit, tot, frac = exact_gold_match(df, gold)
        n_cov = int(df.loc[gold.index].notna().all(axis=1).sum())
        verdict = "LEAK" if (m > LEAK_AUC or frac > LEAK_EXACT) else "clean"
        leak[name] = dict(macro_auc=m, per_col=aucs, exact_hits=hit, exact_cells=tot,
                          exact_frac=frac, gold_rows_covered=n_cov, verdict=verdict)
        print(f"  {name:18s} macro-AUC={m:.4f}  exact-gold-cells={hit}/{tot} "
              f"({frac:.3f})  gold rows covered={n_cov}/58  -> {verdict}")
    leakers = [k for k, v in leak.items() if v["verdict"] == "LEAK"]
    if leakers:
        print("\n" + "!" * 78)
        print(f"!! GOLD LEAK DETECTED in: {leakers} -- EXCLUDED from the consensus")
        print("!" * 78 + "\n")
    else:
        print("  -> no leaks: every candidate is a genuine report-derived labeler.\n")

    for name in ("pilkwang_v2", "steven_v2", "lixin73_report"):
        if name in leakers:
            raise SystemExit(f"teacher {name} leaks the gold; refusing to build a consensus on it")

    # ---- 2. consensus, exactly the public recipe -------------------------------
    teachers = ["pilkwang_v2", "steven_v2", "lixin73_report_conftempered"]
    cube = np.stack([pilk.to_numpy(), st_v2.to_numpy(), lix_temp.to_numpy()], axis=0)
    print(f"[cube] {cube.shape} = teachers x studies x targets; "
          f"NaN cells = {int(np.isnan(cube).sum())} "
          f"(pilkwang is missing 1 study -> {len(TARGETS)} cells)")

    consensus = np.nanmean(cube, axis=0)
    disagreement = np.nanmean(np.abs(cube - consensus[None]), axis=0)
    agreement = np.clip(1.0 - 2.0 * disagreement, 0.0, 1.0)
    certainty = np.clip(2.0 * np.abs(consensus - 0.5), 0.0, 1.0)
    w = 0.15 + 0.85 * (0.65 * agreement + 0.35 * certainty)

    cons_df = pd.DataFrame(consensus, index=index, columns=TARGETS)
    w_df = pd.DataFrame(w, index=index, columns=TARGETS)

    # a literal-upstream variant (raw hard lixin73) for comparison only
    cube_hard = np.stack([pilk.to_numpy(), st_v2.to_numpy(), lix_hard.to_numpy()], axis=0)
    cons_hard = pd.DataFrame(np.nanmean(cube_hard, axis=0), index=index, columns=TARGETS)

    # ---- 3. quality of the consensus vs gold (NO gold overwrite) ---------------
    cons_auc = per_col_auc(cons_df, gold)
    cons_hard_auc = per_col_auc(cons_hard, gold)
    st_v4_auc = leak["steven_v4_blend"]["per_col"]
    print("[gold AUC] consensus(conf-tempered)={:.4f}  consensus(raw-hard-lixin)={:.4f}  "
          "steven_v4_blend={:.4f}".format(macro(cons_auc), macro(cons_hard_auc), macro(st_v4_auc)))

    # ---- 4. diagnostics --------------------------------------------------------
    pair_absdiff = {}
    named = {"pilkwang_v2": pilk, "steven_v2": st_v2,
             "lixin73_report_conftempered": pd.DataFrame(lix_temp, index=index, columns=TARGETS)}
    for a, b in itertools.combinations(named, 2):
        d = (named[a] - named[b]).abs()
        pair_absdiff[f"{a} vs {b}"] = {t: float(d[t].mean(skipna=True)) for t in TARGETS}
        pair_absdiff[f"{a} vs {b}"]["__macro"] = float(np.mean(
            [pair_absdiff[f"{a} vs {b}"][t] for t in TARGETS]))

    w_stats = {t: dict(mean=float(w_df[t].mean()),
                       p10=float(np.percentile(w_df[t], 10)),
                       p50=float(np.percentile(w_df[t], 50)),
                       p90=float(np.percentile(w_df[t], 90))) for t in TARGETS}

    pos_rate = {t: dict(consensus=float((cons_df[t] > 0.5).mean()),
                        steven_v4=float((st_v4[t] > 0.5).mean()),
                        pilkwang=float((pilk[t] > 0.5).mean()),
                        steven_v2=float((st_v2[t] > 0.5).mean()),
                        lixin73=float((lix_hard[t] > 0.5).mean())) for t in TARGETS}

    # cells where all three teachers sit inside a 0.1 band
    spread = np.nanmax(cube, axis=0) - np.nanmin(cube, axis=0)
    tight = spread <= 0.1
    tight_per_col = {t: float(tight[:, i].mean()) for i, t in enumerate(TARGETS)}
    n_tight, n_cells = int(tight.sum()), int(tight.size)
    print(f"[tight cells] all 3 teachers within 0.1: {n_tight}/{n_cells} "
          f"({n_tight / n_cells:.3f})")

    # ---- 5. write the parquets -------------------------------------------------
    y_out = cons_df.copy()
    w_out = w_df.copy()
    y_out.loc[gold.index, TARGETS] = gold[TARGETS].to_numpy()
    w_out.loc[gold.index, TARGETS] = args.gold_weight

    def assemble(y: pd.DataFrame, ww: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=index)
        out.index.name = UID
        for t in TARGETS:
            out[t] = y[t].astype("float32")
        for t in TARGETS:
            out[f"{t}__w"] = ww[t].astype("float32")
        out["is_gold"] = gold_mask.reindex(index).fillna(False).astype(bool)
        out["fold"] = folds.astype("int8")
        return out

    meta_common = {
        "recipe": "prvsiyan 3-teacher consensus (reports/public_solutions.md 2b)",
        "teachers": json.dumps(teachers),
        "teacher_files": json.dumps({
            "pilkwang_v2": str(PILKWANG.relative_to(ROOT)),
            "steven_v2": str(STEVEN_V2.relative_to(ROOT)),
            "lixin73_report_conftempered": str(LIXIN_REPORT.relative_to(ROOT)),
        }),
        "lixin73_transform": "0.5 + (y-0.5)*conf   (file holds hard 0/1 + __conf)",
        "weight_formula": "w = 0.15 + 0.85*(0.65*agreement + 0.35*certainty)",
        "yunus_excluded": "true (leaks the 58 gold; see reports/label_foundation.md)",
        "leak_check": json.dumps({k: {"macro_auc": round(v["macro_auc"], 4),
                                      "exact_frac": round(v["exact_frac"], 4),
                                      "verdict": v["verdict"]} for k, v in leak.items()}),
        "consensus_gold_macro_auc_no_overwrite": f"{macro(cons_auc):.4f}",
        "folds_source": str(FOLDS_CSV.relative_to(ROOT)),
        "n_gold": str(len(gold)),
    }

    for path, y, ww, gw in ((args.out, y_out, w_out, args.gold_weight),
                            (args.out_nogold, cons_df, w_df, None)):
        df = assemble(y, ww)
        tbl = pa.Table.from_pandas(df, preserve_index=True)
        meta = dict(meta_common)
        meta["gold_weight"] = "none (gold rows keep the consensus)" if gw is None else str(gw)
        meta["gold_overwrite"] = "false" if gw is None else "true"
        md = {**{k.encode(): v.encode() for k, v in meta.items()},
              **(tbl.schema.metadata or {})}
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl.replace_schema_metadata(md), path, compression="zstd")
        print(f"[write] {path}  {df.shape}  gold_weight={meta['gold_weight']}")

    stats = dict(value_kinds=value_kinds, lixin73_hard_equals_report=bool(same),
                 lixin73_conf=lix_conf_desc, leak=leak, teachers=teachers,
                 consensus_gold_auc=cons_auc, consensus_gold_macro=macro(cons_auc),
                 consensus_hardlixin_gold_macro=macro(cons_hard_auc),
                 steven_v4_gold_auc=st_v4_auc, steven_v4_gold_macro=macro(st_v4_auc),
                 pair_absdiff=pair_absdiff, weight_stats=w_stats, pos_rate=pos_rate,
                 tight_cells=dict(n=n_tight, total=n_cells, frac=n_tight / n_cells,
                                  per_col=tight_per_col),
                 gold_weight=args.gold_weight,
                 gold_pos={t: int(gold[t].sum()) for t in TARGETS})
    if args.stats_json:
        args.stats_json.write_text(json.dumps(stats, indent=1))
        print(f"[write] {args.stats_json}")
    print("\n===STATS_JSON===")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
