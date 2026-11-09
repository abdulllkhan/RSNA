"""Regenerate the figures of paper.tex (vector PDF) into this directory.

    PYTHONPATH=<repo root> .venv/bin/python -P figures/make_figures.py

Inputs. Figure 3 recomputes its local (OOF) deltas from LOCAL, UNTRACKED data that is not in
this repository (the competition data cannot be redistributed and the OOF files are generated):
    data/oof/*.parquet      stitched 5-fold out-of-fold predictions, one file per member and fold
    data/labels/...         the public stevenleehans LLM report labels (llm_labels_v4_blend.csv)
    data/comp_csv/train.csv competition training table (the 58 radiologist-labelled studies)
The leaderboard scores in Figures 2 and 3 are the 15 submissions as listed by
`kaggle competitions submissions rsna-knee-abnormality-detection` after the competition closed.
Figure 1 is a diagram and needs no data.

Outputs: pipeline.pdf (Fig. 1), submissions.pdf (Fig. 2), transfer.pdf (Fig. 3). The numbers behind
Figure 3 are also printed, together with the gold-58 deltas quoted in the text.
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OOF = os.path.join(ROOT, "data", "oof")
STEVEN_V4 = os.path.join(ROOT, "data", "labels", "stevenleehans_rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
TRAIN_CSV = os.path.join(ROOT, "data", "comp_csv", "train.csv")
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]

# All 15 submissions in time order: (paper label, Kaggle ref, date UTC, public, private).
SUBS = [("1", 56082798, "09-07", 0.877, 0.857), ("2", 56084729, "09-07", 0.893, 0.877),
        ("3", 56089619, "09-08", 0.901, 0.882), ("4", 56106809, "09-08", 0.905, 0.889),
        ("5", 56107848, "09-08", 0.905, 0.889), ("6", 56108377, "09-08", 0.913, 0.896),
        ("7", 56121468, "09-09", 0.928, 0.913), ("8a", 56129012, "09-09", 0.931, 0.916),
        ("8", 56140141, "09-10", 0.935, 0.921), ("9", 56176442, "09-12", 0.938, 0.924),
        ("10", 56387291, "09-20", 0.939, 0.924), ("10b", 56388931, "09-20", 0.940, 0.925),
        ("11", 56434931, "09-21", 0.940, 0.925), ("12", 56468196, "09-22", 0.937, 0.923),
        ("13", 57104154, "10-12", 0.941, 0.927)]
TOP_PUBLIC, TOP_PRIVATE = 0.970, 0.960
FINALS = {"13": "A", "10b": "B"}
BOARD = {s[0]: (s[3], s[4]) for s in SUBS}

# Colour-blind-safe pair (validated: CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on white) + neutral inks.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FILL_DATA, FILL_MODEL, EDGE_MODEL = "#f0efec", "#dbe9fb", "#86b6ef"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "mathtext.cal": "sans",
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
})
COL_W, TEXT_W = 3.55, 7.30  # inches: one column and the full text width of paper.tex


# ----------------------------------------------------------------------------- data
def _read(path):
    d = pd.read_parquet(path)
    if "StudyInstanceUID" in d.columns:
        d = d.set_index("StudyInstanceUID")
    d.index = d.index.astype(str)
    return d[T].astype(float)


def stitched(name):
    """All five held-out folds of one member, concatenated (each study predicted by the model that did not train on it)."""
    files = sorted(glob.glob(os.path.join(OOF, f"{name}_fold[0-9].parquet")))
    if len(files) != 5:
        raise SystemExit(f"{name}: expected 5 fold files in {OOF}, found {len(files)}")
    d = pd.concat([_read(f) for f in files])
    return d[~d.index.duplicated()]


def macro(P, Y):
    return float(np.mean([roc_auc_score(Y[c], P[c]) for c in T if Y[c].nunique() > 1]))


def labels():
    sv = pd.read_csv(STEVEN_V4).set_index("StudyInstanceUID")
    sv.index = sv.index.astype(str)
    llm = (sv[T] > 0.5).astype(int)                    # the OOF ruler: binarised LLM report labels
    tr = pd.read_csv(TRAIN_CSV).set_index("StudyInstanceUID")
    tr.index = tr.index.astype(str)
    gold = tr.dropna(subset=T)[T].astype(int)          # the 58 radiologist-labelled studies
    return llm, gold


def blend(parts, idx):
    """Per-column percentile-rank average, ranks taken within idx: parts = [(frame, weight), ...]."""
    tot = sum(w for _, w in parts)
    return sum(w * f.loc[idx].rank(pct=True) for f, w in parts) / tot


def local_deltas():
    """Local (OOF) gain of each single change in Fig. 3, recomputed from the OOF files."""
    llm, gold = labels()
    m = {n: stitched(n) for n in ["effv2s_k24", "effv2s_k24_384", "effv2s_k24_tta3", "effv2s_k24_384_tta3",
                                  "v2_effv2s384_lr", "v2_effv2s384_c110", "v2_effv2s384_lr_distill",
                                  "v2_effv2s384_c110_distill"]}
    idx = m["effv2s_k24"].index
    for d in m.values():
        idx = idx.intersection(d.index)
    g = idx.intersection(gold.index)

    def score(parts, ix=idx):
        e = blend(parts, ix)
        gi = ix.intersection(gold.index)
        return macro(e, llm.loc[ix]), macro(e.loc[gi], gold.loc[gi])

    sub2 = score([(m["effv2s_k24"], 1)])
    sub4 = score([(m["effv2s_k24"], 1), (m["effv2s_k24_384"], 3)])
    sub6 = score([(m["effv2s_k24_tta3"], 1), (m["effv2s_k24_384_tta3"], 3)])
    hr = (m["effv2s_k24_384_tta3"], 3)

    def sub_v2(a, b, w):
        """hr at weight 3 plus a v2 arm of weight w that is an equal mix of two 5-fold members."""
        return score([hr, (m[a], w / 2), (m[b], w / 2)])

    sub10b = sub_v2("v2_effv2s384_lr", "v2_effv2s384_c110", 10)
    sub11 = score([hr] + [(m[n], 15 / 3) for n in ["v2_effv2s384_lr", "v2_effv2s384_c110", "v2_effv2s384_c110_distill"]])
    sub12 = sub_v2("v2_effv2s384_lr_distill", "v2_effv2s384_c110_distill", 15)

    # one v2 fold-0 member added to the fold-0 ensemble (sub #7 vs #6): fold 0 only, 882 studies
    f0 = {n: _read(os.path.join(OOF, f"{n}_fold0.parquet")) for n in ["effv2s_k24_tta3", "effv2s_k24_384_tta3", "v2_effv2s384_lr"]}
    i0 = f0["v2_effv2s384_lr"].index.intersection(f0["effv2s_k24_tta3"].index)
    base0 = [(f0["effv2s_k24_tta3"], 1), (f0["effv2s_k24_384_tta3"], 3)]
    sub6_f0 = score(base0, i0)
    sub7_f0 = score(base0 + [(f0["v2_effv2s384_lr"], 5)], i0)

    # public reader at weight 6 (sub #13 vs #10b), on the studies with an honest reader prediction
    rd = _read(os.path.join(OOF, "pubreader_gpjw_oof.parquet"))
    ov = idx.intersection(rd.index)
    v2s = [(m["v2_effv2s384_lr"], 5), (m["v2_effv2s384_c110"], 5)]
    r10b = score([hr] + v2s, ov)
    r13 = score([hr] + v2s + [(rd, 6)], ov)

    rows = [  # label, (before, after) local scores, board before, board after, basis
        ("+384 px arm (#2→#4)", sub2, sub4, "2", "4", "stitched OOF, 4,407 studies"),
        ("+TTA $S$=3 (#5→#6)", sub4, sub6, "5", "6", "stitched OOF, 4,407 studies"),
        ("+v2 slots, one fold (#6→#7)", sub6_f0, sub7_f0, "6", "7", f"fold-0 ensemble, {len(i0)} studies"),
        ("+distilled arm (#10b→#11)", sub10b, sub11, "10b", "11", "stitched OOF, 4,407 studies"),
        ("distilled arms only (#10b→#12)", sub10b, sub12, "10b", "12", "stitched OOF, 4,407 studies"),
        ("+public reader (#10b→#13)", r10b, r13, "10b", "13", f"reader overlap, {len(ov)} studies"),
    ]
    print(f"[data] {len(idx)} studies, {len(g)} gold; reader overlap {len(ov)}")
    out = []
    for lab, (b_oof, b_gold), (a_oof, a_gold), sb, sa, basis in rows:
        d_pub = round(BOARD[sa][0] - BOARD[sb][0], 3)
        d_prv = round(BOARD[sa][1] - BOARD[sb][1], 3)
        d_oof = a_oof - b_oof
        print(f"  {lab:38s} OOF {b_oof:.4f}->{a_oof:.4f} ({d_oof:+.4f})  gold {b_gold:.4f}->{a_gold:.4f} "
              f"({a_gold - b_gold:+.4f})  public {d_pub:+.3f}  private {d_prv:+.3f}  [{basis}]")
        out.append((lab, d_oof, d_pub, d_prv))
    print(f"  sub #10b stitched: OOF {sub10b[0]:.4f} gold {sub10b[1]:.4f}")
    return out


# ----------------------------------------------------------------------------- Fig. 1
def _box(ax, x, y, w, h, text, fill=FILL_DATA, edge=AXIS, size=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04", fc=fill, ec=edge, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.12)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6.5, lw=0.8, color=INK2,
                                 shrinkA=0, shrinkB=0))


def fig_pipeline(path):
    H = 1.12
    fig = plt.figure(figsize=(TEXT_W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, TEXT_W)
    ax.set_ylim(0, H)
    ax.axis("off")
    gap, h = 0.14, 0.385

    # (a) one v2 model, per study
    ax.text(0.0, 1.065, "(a) v2 arm: one model, per study", fontsize=7.2, color=INK2, va="center", style="italic")
    y = 0.615
    row = [(0.84, "DICOM study\n~5 series\n(sag, cor, ax)", FILL_DATA, AXIS),
           (1.10, "slices sorted by\nImagePositionPatient;\nleft knees mirrored", FILL_DATA, AXIS),
           (1.00, "5 plane \u00d7 fluid\nslots, 64 slices\nover 2\u201398%", FILL_DATA, AXIS),
           (1.06, "140 or 110 mm crop,\n1\u201399% window\nper series, 384 px", FILL_DATA, AXIS),
           (1.08, "EfficientNetV2-S\nper slice + slot and\nposition embedding", FILL_MODEL, EDGE_MODEL),
           (0.94, "12 per-finding\nattention pools\n(+ slot prior)", FILL_MODEL, EDGE_MODEL)]
    x = 0.0
    for w, txt, fc, ec in row:
        _box(ax, x, y, w, h, txt, fc, ec)
        _arrow(ax, x + w + 0.01, y + h / 2, x + w + gap - 0.01, y + h / 2)
        x += w + gap
    _box(ax, x, y, TEXT_W - x, h, "12\nlogits", FILL_MODEL, EDGE_MODEL)

    # (b) the submitted ensemble
    ax.text(0.0, 0.48, "(b) final ensemble (#13): per-column rank average of three arms", fontsize=7.2,
            color=INK2, va="center", style="italic")
    y = 0.02
    arms = [(1.76, "hr arm: 5 folds, weight 3\nfull frame, 384 px, 24 slices,\none attention pool, TTA $S$ = 3",
             FILL_DATA, AXIS),
            (1.76, "v2 arm: 10 models, weight 10\n5 folds \u00d7 {140, 110} mm,\nTTA $S$ = 3 (panel a)", FILL_MODEL,
             EDGE_MODEL),
            (1.76, "public 2.5D ConvNeXt-T\nreader: 3 folds, weight 6,\nno TTA", FILL_DATA, AXIS)]
    plus_w = 0.14
    x = 0.0
    for i, (w, txt, fc, ec) in enumerate(arms):
        _box(ax, x, y, w, h, txt, fc, ec)
        x += w
        if i < len(arms) - 1:
            ax.text(x + plus_w / 2, y + h / 2, "+", ha="center", va="center", fontsize=9, color=INK2)
            x += plus_w
    _arrow(ax, x + 0.01, y + h / 2, x + gap - 0.01, y + h / 2)
    x += gap
    w_comb = 0.78
    _box(ax, x, y, w_comb, h, "percentile\nrank average\nper column")
    _arrow(ax, x + w_comb + 0.01, y + h / 2, x + w_comb + gap - 0.01, y + h / 2)
    x += w_comb + gap
    _box(ax, x, y, TEXT_W - x, h, "submission\n(12 columns)")
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 2
def fig_submissions(path):
    fig, ax = plt.subplots(figsize=(COL_W, 1.48))
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.24, top=0.985)
    x = np.arange(1, len(SUBS) + 1)
    pub = np.array([s[3] for s in SUBS])
    prv = np.array([s[4] for s in SUBS])
    ax.set_ylim(0.846, 0.980)
    ax.set_xlim(0.4, len(SUBS) + 0.6)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for yv, txt, c, dy, va in [(TOP_PUBLIC, "top public score 0.970", BLUE, 0.0012, "bottom"),
                               (TOP_PRIVATE, "top private score 0.960", ORANGE, -0.0012, "top")]:
        ax.axhline(yv, color=c, lw=0.8, ls=(0, (3, 2)), alpha=0.75, zorder=1)
        ax.text(0.55, yv + dy, txt, fontsize=7, color=INK2, va=va, ha="left")
    ax.plot(x, pub, color=BLUE, lw=1.1, marker="o", ms=3.4, mec="white", mew=0.6, label="public", zorder=3)
    ax.plot(x, prv, color=ORANGE, lw=1.1, marker="s", ms=3.2, mec="white", mew=0.6, label="private", zorder=3)
    for i, s in enumerate(SUBS):
        if s[0] in FINALS:
            ax.plot([x[i], x[i]], [prv[i], pub[i]], color=INK2, lw=0.6, zorder=2)
            ax.text(x[i], pub[i] + 0.0042, FINALS[s[0]], ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                    color=INK)
    notes = {2: ("5 folds", 0.25), 4: ("384 px", 0.0), 6: ("TTA", 0.0), 7: ("slots", 0.0), 10: ("110 mm", 0.0),
             14: ("distilled", 0.0)}
    for pos, (txt, dx) in notes.items():
        ax.text(pos + dx, prv[pos - 1] - 0.0105, txt, ha="center", va="top", fontsize=7, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in SUBS])
    ax.tick_params(axis="x", labelsize=7, pad=1.5)
    ax.set_xlabel("submission, in time order (7 Sep \u2013 12 Oct 2026)", labelpad=1.5)
    ax.set_ylabel("macro ROC AUC", labelpad=2)
    ax.set_yticks([0.86, 0.88, 0.90, 0.92, 0.94, 0.96])
    ax.legend(loc="lower right", frameon=False, handlelength=1.6, borderaxespad=0.15, labelspacing=0.25)
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 3
def fig_transfer(path, rows):
    fig, ax = plt.subplots(figsize=(COL_W, 1.45))
    fig.subplots_adjust(left=0.415, right=0.975, bottom=0.245, top=0.99)
    n = len(rows)
    yc = np.arange(n)[::-1].astype(float)
    bh = 0.25
    series = [("local OOF (LLM labels)", 1, MUTED), ("public board", 2, BLUE), ("private board", 3, ORANGE)]
    for name, k, col in series:
        vals = [r[k] for r in rows]
        yy = yc + (1 - (k - 1)) * bh
        ax.barh(yy, vals, height=bh * 0.9, color=col, label=name, zorder=3, lw=0)
        for yv, v in zip(yy, vals):
            if abs(v) < 5e-4:  # a change of exactly 0.000: draw a stub at zero instead of nothing
                ax.plot([0, 0], [yv - bh * 0.42, yv + bh * 0.42], color=col, lw=2.2, solid_capstyle="butt", zorder=5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.axvline(0, color=INK2, lw=0.6, zorder=4)
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_yticks(yc)
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis="y", length=0, labelsize=7, pad=2)
    ax.tick_params(axis="x", labelsize=7, pad=1.5)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.0045, 0.025)
    ax.set_xticks([0.0, 0.005, 0.01, 0.015, 0.02, 0.025])
    ax.set_xticklabels(["0", ".005", ".010", ".015", ".020", ".025"])
    ax.set_xlabel("gain in macro ROC AUC", labelpad=1.5)
    ax.axhline(2.5, color=AXIS, lw=0.6, zorder=2)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.315), frameon=False, handlelength=1.0, borderaxespad=0.1,
              labelspacing=0.2)
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)


def main():
    rows = local_deltas()
    fig_pipeline(os.path.join(HERE, "pipeline.pdf"))
    fig_submissions(os.path.join(HERE, "submissions.pdf"))
    fig_transfer(os.path.join(HERE, "transfer.pdf"), rows)
    print("[done] wrote pipeline.pdf, submissions.pdf, transfer.pdf to", HERE)


if __name__ == "__main__":
    main()
