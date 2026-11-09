"""Generate notebook/submission_multi.ipynb — the MULTI-ARM ensemble inference notebook.

Sibling of `build_notebook.py` (which stays the working single-arm v7 generator and is not
touched). Same contract: logic lives in `src/rsna/*.py`, this script INLINES that source into
a self-contained notebook (no `src` import, no network at Kaggle rerun) — single source of
truth, zero copies of model/dataset/inference code.

What is new versus `build_notebook.py`:
  * it inlines the FULL dependency chain, not just the 224 path:
        __init__ -> decode -> order -> laterality -> dataset -> dataset_hr -> tta
                 -> dataset_25d -> dataset_ordered -> dataset_plane -> dataset_v2
                 -> model -> model_plane -> model_v2
                 -> infer -> infer_hr -> infer_25d -> infer_ordered -> infer_plane -> infer_v2
                 -> infer_tta
    (the order the arm reports specify: reports/hr_arm.md §5.2, reports/plane_arm.md §7.1,
     reports/arm_25d.md §6, reports/arm_ordered.md §6, reports/tta_tooling.md §7,
     reports/arm_v2.md §8(1));
  * the stripper handles MULTI-LINE relative imports (`from .dataset import (\n A,\n B,\n)`),
    which `dataset_hr`/`dataset_plane`/`model_plane`/`infer_hr`/`infer_plane` all use — the v7
    single-line-only stripper would emit a SyntaxError on them;
  * it injects a `__file__` shim, because `dataset_hr.ROOT` is computed from `__file__` and a
    notebook cell has none (NameError -> the whole bundle cell would die);
  * the driver AUTO-DETECTS each checkpoint's ARM from the checkpoint's own `args`/`cfg`,
    runs each arm through its own size-aware decode-once predictor, and combines the arms with
    a per-column percentile-RANK average (PLAN §9 / reports/metric_and_format.md §1: AUC only
    sees within-column order, and the arms are not probability-calibrated against each other).

Arm detection (from the checkpoint alone — nothing is inferred from the file name):
    infer_v2.is_v2_meta(args)                           -> "v2"     (infer_v2, 5 plane x fluid
                                                                     slots, laterality-canonical)
    infer_ordered.is_ordered_meta(args)                 -> "ordered"(infer_ordered, anat. slots)
    infer_25d.is_25d_meta(args)                         -> "25d"    (infer_25d, [prev,cur,next])
    plane_mode/k_per_plane in args, or k_per_plane/fusion in cfg, or a plane_weights tensor
                                                        -> "plane"  (infer_plane, 3 plane bags)
    args["size"] present and != 224                     -> "hr"     (infer_hr, size-aware bag)
    otherwise (baseline 224: args has `k`, no `size`)    -> "base"   (infer, the v7 path verbatim)
The `v2`, `ordered` and `25d` tests come FIRST because all three arms carry args["size"] like the
plain HR arm: without them a 384 checkpoint from any of them falls through to `hr` and is fed the
wrong input distribution (lexicographic slots / gray x 3 / a 24-slice bag with no
slot_id+pos+valid side inputs) — silently wrong, not merely worse (reports/arm_25d.md §6,
reports/arm_ordered.md §6, reports/arm_v2.md §8(3)). The three predicates are mutually exclusive
and all read args["mode"], which only one trainer ever writes: "v2" (train_v2), "2.5d"
(train_25d), "ordered" (train_ordered; also accepts the args["order"]+args["coverage"] pair).
A `train_hr` checkpoint that happens to be 224 lands in "base": `dataset_hr.read_slice_hr` at
224 is bit-identical to `dataset._read_slice_dicom` (reports/decode_raw.md), so that is a free
win — it joins the v7 loader instead of opening a second 224 decode pass.

NOTE: none of ordered / 2.5D / v2 needs `data/series_order.parquet` on Kaggle. All three
inference paths order slices from the DICOM headers of the mount itself (ordered/25d build a
`SliceOrderResolver` with no parquet; v2 calls `order.series_ranks_from_dicom` per chosen
series). Do NOT ship the 30.7 MB parquet — nor `data/study_side.parquet`: the v2 arm derives
laterality from the same decoded headers via `laterality.side_from_series_meta`.

Combination:
  * within an arm, members are combined FIRST by the arm's own predictor (they share
    architecture + preprocessing): probability mean by default, `RSNA_MEMBER_MODE=rank` for a
    rank mean. The base arm always uses `infer.predict_studies`, i.e. the exact v7 combiner, so
    a base-only weights dataset reproduces the v7 submission byte-for-byte;
  * across arms, `infer_plane.rank_average_frames` per-column rank-averages the finished arm
    frames with per-arm weights from `RSNA_ARM_WEIGHTS`.

Slot-sampling TTA (`RSNA_TTA`, reports/tta_tooling.md):
  * S>1 makes an arm run its whole predictor S times, once per `tta.SamplingVariant` (a phase
    shift of the eval slot grid), and per-column rank-averages the S frames BEFORE the arm frame
    enters the across-arm combiner. `variant 0 IS the baseline sampling`, so **S=1 is the literal
    code path this notebook ran before TTA existed** — byte-identical output, proven by md5.
  * `base` (via `infer_tta.predict_studies_base_tta`, wrapping the exact v7 combiner), `hr`
    (via `infer_tta.predict_studies_tta`) and `v2` (via `infer_v2.predict_studies_v2(tta=S)`,
    which shifts the SLOT PHASE in `dataset_v2.plan_slots` — reports/tta_v2.md) have TTA.
    `plane` has its own per-plane sampler; `25d`/`ordered` build their slots through samplers
    `tta.dataset_factory` has no wrapper for (and `ordered`'s coverage="center" bypasses
    `_pick_positions` entirely, so a variant would be a silent no-op). Those three arms are
    pinned to S=1 and say so.
  * Cost for base/hr is ~linear in S (decode AND forward repeat). For `v2` it is S x FORWARD
    only: the S variant bags are assembled from ONE decode per study, because a slot phase
    changes only which already-decoded slices are stacked. Do not raise S without an OOF
    measurement (reports/tta_tooling.md §6 / reports/tta_v2.md: the bar is >= +0.005 macro on
    both anchors).

Env / config (all optional; the Kaggle defaults are the fallbacks):
    RSNA_INPUT_ROOT   data mount (default: scan /kaggle/input for test.csv + test_series/)
    RSNA_WEIGHTS_DIR  weights search root (default /kaggle/input)
    RSNA_OUT          output csv (default /kaggle/working/submission.csv)
    RSNA_DEVICE       cuda|cpu (default cuda if available, with the P100-style CPU fallback)
    RSNA_WORKERS      DataLoader workers (default 2)
    RSNA_BATCH        base batch (default 8); RSNA_BATCH_{BASE,HR,25D,ORDERED,V2,PLANE} per arm
    RSNA_ARMS         comma filter, e.g. "base,hr"      (default: all of ARM_ORDER)
    RSNA_ARM_WEIGHTS  e.g. "base=1,hr=3,25d=2,ordered=1,v2=4,plane=1" (default equal; <=0 off)
    RSNA_MEMBER_MODE  prob|rank, within-arm combiner for hr/25d/ordered/v2/plane (default prob)
    RSNA_TTA          slot-sampling TTA: S samplings per arm, 1 = off (default 1).
                      Global ("3") or per arm ("base:3,hr:2"); a bare number sets the default
                      for every arm and `arm:S` overrides one. base + hr + v2 only (see below).
    RSNA_TTA_COMBINE  rank|prob, across-variant combiner (default rank)
    RSNA_BATCH_READER / RSNA_WORKERS_READER   reader arm batch (default 4 = the released infer.py) / loader
                      workers (default RSNA_WORKERS if set, else 4 = the released infer.py)
    RSNA_READER_DEADLINE_H   the reader arm aborts (and is dropped) once the driver has run this long (default 8.0)
    RSNA_DUMP_ARMS    local tests only: pickle every finished arm frame into this directory

The `reader` arm (src/rsna/reader_gpjw.py; EXPERIMENTS A18): goodpjw2008's public 2.5D ConvNeXt-T reader, its 3
checkpoints read from the attached public dataset goodpjw2008/rsna-knee-2-5d-convnext-reader. It is OPT-IN (weight
0 unless RSNA_ARM_WEIGHTS names it, so an older weights string runs exactly as before), runs LAST, is pinned to S=1,
and is inlined in its own module namespace (its names collide with the other arms'). Its frame may lack single
studies (0 valid slots / a failed forward): those rows average the other arms with renormalised weights
(combine_partial.py); its pre-registered guard drops the whole arm on > 2 % invalid studies or > 2 % series with
decode errors. Before it runs, the other arms' submission is already written to RSNA_OUT (crash fallback).

Build-time switch: RSNA_BUNDLE_PKG=<dir> reads the inlined modules from <dir> instead of src/rsna (used to build
the sub #13 bundle from git-HEAD copies of the pre-existing modules while src/rsna carries unrelated edits).

Run:  .venv/bin/python notebook/build_notebook_multi.py
"""
from __future__ import annotations

import ast
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
# Optional build-time switch: inline the modules from another package dir (default: src/rsna, unchanged).
PKG = os.path.abspath(os.environ.get("RSNA_BUNDLE_PKG") or os.path.join(ROOT, "src", "rsna"))

# Dependency order. decode.py, order.py and laterality.py are import-free (lazy pydicom);
# dataset_hr needs dataset+decode; tta needs dataset+dataset_hr; dataset_25d / dataset_ordered
# need dataset+dataset_hr+order; dataset_plane needs dataset+dataset_hr; dataset_v2 needs
# dataset+dataset_hr+order (and, lazily, laterality); model_plane needs model+dataset_plane;
# model_v2 needs dataset_v2; infer_hr / infer_25d / infer_ordered / infer_plane / infer_v2
# reuse helpers from infer.py (infer_v2 also reuses infer_hr._forward_with_guard, and calls
# decode.read_series_v2 + order.series_ranks_from_dicom + laterality.side_from_series_meta
# through lazy imports that the stripper removes -> they resolve from the flat namespace);
# infer_tta wraps infer + infer_hr, so it goes last.
# (reports/arm_25d.md §6 edit 1, reports/arm_ordered.md §6 edit 1, reports/tta_tooling.md §7.1,
#  reports/arm_v2.md §8 edit 1)
MODULES = [
    "__init__.py",
    "decode.py",           # also carries the v2 decoder: read_series_v2 / normalize_series_pct
    "order.py",            # import-free (lazy pydicom); must precede dataset_25d/_ordered/_v2
    "laterality.py",       # import-free (lazy pydicom); must precede dataset_v2 / infer_v2
    "dataset.py",
    "dataset_hr.py",
    "tta.py",              # needs dataset + dataset_hr (binds sample_study_slices.__globals__)
    "dataset_25d.py",      # needs dataset + dataset_hr + order
    "dataset_ordered.py",  # needs dataset + dataset_hr + order
    "dataset_plane.py",
    "dataset_v2.py",       # needs dataset + dataset_hr + order (+ lazy laterality)
    "model.py",
    "model_plane.py",
    "model_v2.py",         # needs dataset_v2 (K_FULL / N_SLOTS / SLOTS / slot_id_vector)
    "infer.py",
    "infer_hr.py",
    "infer_25d.py",        # reuses _forward_with_guard / check_fingerprint_hr from infer_hr
    "infer_ordered.py",    # ditto
    "infer_plane.py",
    "infer_v2.py",         # reuses _forward_with_guard from infer_hr; needs model_v2
    "infer_tta.py",        # wraps infer.predict_studies + infer_hr.predict_studies_hr
    "combine_partial.py",  # per-row fallback around infer_plane.rank_average_frames (the reader arm)
    "reader_gpjw.py",      # ISOLATED (see ISOLATED_MODULES): the public 2.5D reader, module `reader_gpjw`
]

# Modules executed in their OWN namespace instead of being flattened into the bundle. reader_gpjw.py is the
# verbatim port of a third-party reader whose module-level names (N_SLOTS=6, SIZE, MEAN, STD, PLANES, LABELS,
# load_series, make_views, ...) collide with -- and would silently rebind -- names the other arms read at call
# time (dataset_v2.N_SLOTS=5, ...). Renaming them would break the port's AST identity with the author's code, so
# the file is embedded verbatim as a string and exec'd into a fresh module object; the driver calls it as
# `reader_gpjw.<name>`.
ISOLATED_MODULES = {"reader_gpjw.py": "reader_gpjw"}

# ---------------------------------------------------------------------------------------
# Per-module identifier renames applied while inlining.
#
# The bundle is ONE FLAT NAMESPACE, so a module-level name defined by two modules silently
# resolves to whichever module was inlined last -- for every function in the bundle, because
# functions look their globals up at CALL time.
#
# `MODE_TAG` is exactly that trap: `dataset_25d.MODE_TAG == "2.5d"` and
# `dataset_ordered.MODE_TAG == "ordered"`, and `is_25d_meta` / `is_ordered_meta` both compare
# against the bare name. Inlined as-is, BOTH predicates would test against "ordered": every
# 2.5D checkpoint would fall through to hr/base, and every ORDERED checkpoint would answer True
# to `is_25d_meta` and be run through the 2.5D predictor. Renaming per module restores the
# import-time binding each module had in the package.
#
# `_warn_once` is the same trap in miniature: `decode.py` and `order.py` each define a private
# warn-once helper with its OWN dedup dict and log prefix. Flattened, the later one wins and
# decode's fallbacks would print under `[order]` and share order's dedup keys. Cosmetic rather
# than numerical, but it is the same class of bug, so it gets the same treatment.
RENAMES = {
    "dataset_25d.py": {"MODE_TAG": "MODE_TAG_25D"},
    "infer_25d.py": {"MODE_TAG": "MODE_TAG_25D"},
    "dataset_ordered.py": {"MODE_TAG": "MODE_TAG_ORDERED"},
    "infer_ordered.py": {"MODE_TAG": "MODE_TAG_ORDERED"},
    "order.py": {"_warn_once": "_warn_once_order"},
}

# Module-level names that MAY legitimately be defined more than once in the flat bundle,
# each checked to be either identical or never read on the inference path:
#   __all__               never read in the bundle
#   DEFAULT_SIZE          224 in decode / infer_hr / infer_plane
#   DICOM_EXTS            (".dcm",) in decode / dataset
#   ROOT                  same dirname(__file__)/../.. expression in order / dataset_hr
#   _autocast, _forward_with_guard, _FP32_FALLBACK
#                         textually identical in infer / infer_hr / infer_plane (pre-existing)
#   _root_for_size        identical bodies in infer_hr / infer_25d / infer_ordered / infer_plane
#   build_infer_order     identical body in infer_25d / infer_ordered
#   build_order_resolver  dataset_25d / dataset_ordered; a TRAINING helper, never called by any
#                         inference path (both inference modules use build_infer_order)
#   main                  per-module CLIs; the `if __name__ == "__main__"` guards are stripped,
#                         so nothing in the bundle ever calls one
ALLOWED_DUPLICATES = {
    "__all__", "DEFAULT_SIZE", "DICOM_EXTS", "ROOT",
    "_autocast", "_forward_with_guard", "_FP32_FALLBACK", "_root_for_size",
    "build_infer_order", "build_order_resolver", "main",
}

_FUTURE = re.compile(r"^\s*from\s+__future__\s+import\s+.*$")
# `from .x import ...`, `from . import ...`, `from .x.y import ...`
_REL_IMPORT = re.compile(r"^\s*from\s+\.[\w.]*\s+import\s+.*$|^\s*from\s+\.\s+import\s+.*$")
# `if __name__ == "__main__":` at column 0 -- see _strip_module.
_MAIN_GUARD = re.compile(r"""^if\s+__name__\s*==\s*['"]__main__['"]\s*:""")


def _strip_module(src: str) -> str:
    """Drop `from __future__`, intra-package relative imports, and `__main__` guards.

    * relative imports: the inlined bundle is one flat namespace. Unlike the v7 stripper this
      consumes MULTI-LINE parenthesised imports: leaving the continuation lines behind would be
      a SyntaxError.
    * `if __name__ == "__main__":` blocks: a Jupyter cell's `__name__` IS `"__main__"`, so an
      inlined CLI guard RUNS. `order.py`, `infer_25d.py` and `infer_ordered.py` all end in
      `raise SystemExit(main())` (argparse would then parse the kernel's argv) and `tta.py` ends
      in a `self_check()` + `SystemExit`. Any of those kills the bundle cell before a single
      model loads. None of the pre-TTA modules had a guard, which is why the v7-era stripper
      never needed this.
    """
    out = []
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if _FUTURE.match(line):
            i += 1
            continue
        if _REL_IMPORT.match(line):
            # swallow the whole statement whether it continues with parentheses
            # (`from .x import (\n A,\n)`) or with a backslash (`from .x import a, \\\n b`) --
            # dataset_25d.py uses the backslash form, which paren-depth alone does not catch.
            depth = line.count("(") - line.count(")")
            cont = line.rstrip().endswith("\\")
            i += 1
            while (depth > 0 or cont) and i < len(lines):
                depth += lines[i].count("(") - lines[i].count(")")
                cont = lines[i].rstrip().endswith("\\")
                i += 1
            continue
        if _MAIN_GUARD.match(line):
            # swallow the whole suite: every following blank line or indented line
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i][:1] in (" ", "\t")):
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out).strip("\n")


def _apply_renames(src: str, module: str) -> str:
    """Rewrite the identifiers RENAMES lists for `module` (whole-word, so no partial hits)."""
    for old, new in RENAMES.get(module, {}).items():
        src = re.sub(rf"\b{re.escape(old)}\b", new, src)
    return src


def _toplevel_names(src: str) -> set:
    """Module-level bindings: def / class / plain assignment / annotated assignment."""
    names = set()
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def check_collisions(bodies: dict) -> list:
    """FAIL THE BUILD on an unexpected module-level name collision in the flat bundle.

    `bodies` is {module: stripped+renamed source}. Every duplicate that is not in
    ALLOWED_DUPLICATES is returned; `main()` turns that into a RuntimeError. This gate is the
    reason the MODE_TAG trap above is a build error rather than a wrong submission: a future
    module that reuses a name will break the build loudly instead of silently rebinding a
    predicate at Kaggle rerun time.
    """
    first, bad = {}, []
    for module, body in bodies.items():
        for name in sorted(_toplevel_names(body)):
            if name in first and name not in ALLOWED_DUPLICATES:
                bad.append(f"{name!r}: {first[name]} then {module}")
            first.setdefault(name, module)
    return bad


# `dataset_hr.ROOT` does `os.path.dirname(__file__)`; a notebook cell has no __file__.
FILE_SHIM = '''# --- notebook shim: dataset_hr.ROOT is derived from __file__, which a cell lacks ---
import os as _os
try:
    __file__
except NameError:
    __file__ = _os.path.join(_os.getcwd(), "rsna_bundle.py")
'''


ISOLATED_TEMPLATE = '''import linecache as _linecache
import sys as _sys
import types as _types
{var} = r\'\'\'{src}\'\'\'
{mod} = _types.ModuleType("{mod}")
{mod}.__file__ = "{mod}.py"
_linecache.cache["{mod}.py"] = (len({var}), None, {var}.splitlines(True), "{mod}.py")  # readable tracebacks
_sys.modules["{mod}"] = {mod}
exec(compile({var}, "{mod}.py", "exec", dont_inherit=True), {mod}.__dict__)'''


def isolated_block(src: str, mod: str) -> str:
    """The bundle text that runs `src` VERBATIM as module `mod` (own globals, no __future__ inheritance).
    The source travels as a raw triple-quoted literal; the build fails unless that literal round-trips exactly."""
    if "'''" in src or src.endswith("\\"):
        raise RuntimeError(f"{mod}: source cannot be embedded as a r'''...''' literal")
    if ast.literal_eval("r'''" + src + "'''") != src:
        raise RuntimeError(f"{mod}: embedded literal does not round-trip")
    return ISOLATED_TEMPLATE.format(var="_SRC_" + mod.upper(), mod=mod, src=src)


def build_bundle() -> str:
    parts = ["from __future__ import annotations", "", FILE_SHIM]
    bodies = {}
    for m in MODULES:
        with open(os.path.join(PKG, m)) as f:
            raw = f.read()
        if m in ISOLATED_MODULES:
            body = isolated_block(raw, ISOLATED_MODULES[m])
            bodies[m] = body
            parts.append(f"# ===================== isolated: src/rsna/{m} -> module "
                         f"`{ISOLATED_MODULES[m]}` (own namespace) =====================")
            parts.append(body)
            parts.append("")
            continue
        body = _strip_module(_apply_renames(raw, m))
        bodies[m] = body
        parts.append(f"# ===================== inlined: src/rsna/{m} =====================")
        parts.append(body)
        parts.append("")
    bad = check_collisions(bodies)
    if bad:
        raise RuntimeError("flat-bundle name collision(s) -- add a RENAMES entry or, if the "
                           "duplicate is provably harmless, an ALLOWED_DUPLICATES entry:\n  "
                           + "\n  ".join(bad))
    return "\n".join(parts)


DRIVER = r'''
# ========================= MULTI-ARM SUBMISSION DRIVER =========================
# Loads EVERY checkpoint found under the attached weights dataset(s), detects each one's ARM
# from the checkpoint itself, runs each arm through its own decode-once predictor, and
# per-column rank-averages the arms. Every survival pattern of the v7 driver is kept:
# 0.5 file FIRST, mount discovery, self-size to the hidden test, device probe with CPU
# fallback, capability-aware autocast (bf16 sm_80+ / fp16 T4-V100 / fp32 else, with a
# per-batch fp32 recompute on a dead engine or a non-finite output), advisory (never fatal)
# fingerprint, final schema assert. Every stage prints [time] stage=... elapsed=...s.
import gc
import glob
import os
import re
import time
import traceback

import numpy as np
import pandas as pd
import torch

_T_START = time.time()


def _stage(name, t0):
    """One line per stage. The efficiency metric charges TOTAL notebook wall-clock
    (reports/efficiency_prize.md §2), so the Kaggle log has to show where it went."""
    print(f"[time] stage={name} elapsed={time.time() - t0:.1f}s", flush=True)


OUT = os.environ.get("RSNA_OUT", "/kaggle/working/submission.csv")
DEVICE = os.environ.get("RSNA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = int(os.environ.get("RSNA_WORKERS", "2"))
BATCH = int(os.environ.get("RSNA_BATCH", "8"))
WEIGHTS_DIR = os.environ.get("RSNA_WEIGHTS_DIR", "/kaggle/input")
MEMBER_MODE = os.environ.get("RSNA_MEMBER_MODE", "prob").strip().lower()  # within-arm combiner

ARM_ORDER = ("base", "hr", "25d", "ordered", "v2", "plane", "reader")
# OPT-IN arms keep weight 0 unless RSNA_ARM_WEIGHTS names them, so a weights string written before they existed
# runs exactly as it did. `reader` = goodpjw2008's public 2.5D ConvNeXt-T reader (module `reader_gpjw`); it is
# last in ARM_ORDER, so it runs after every other arm.
OPT_IN_ARMS = ("reader",)
# Arms whose frame may lack single studies: those rows average the other arms only, weights renormalised
# (rank_average_frames_partial). Every other arm covers every study or fails as a whole, as before.
ROW_OPTIONAL_ARMS = ("reader",)
DUMP_ARMS = os.environ.get("RSNA_DUMP_ARMS")  # local tests only: pickle each finished arm frame there

# Arms with a slot-sampling TTA wrapper (reports/tta_tooling.md §7, reports/tta_v2.md).
# `base`/`hr` go through `tta.dataset_factory`; `v2` does NOT -- it needs none, because its
# positions come from `dataset_v2.plan_slots`, which now takes a `phase` directly, and
# `predict_studies_v2(tta=S)` assembles the S bags from ONE decode per study.
# `plane` has its own per-plane sampler; `25d`/`ordered` sample through code
# `tta.dataset_factory` has no wrapper for -- and `ordered`'s coverage="center" does not go
# through `_pick_positions` at all, so a variant there would be a silent no-op.
# Those three are pinned to S=1 and say so in the log.
TTA_CAPABLE = ("base", "hr", "v2")


def parse_arm_weights(spec, default=1.0):
    """"base=1,hr=1,plane=0.5" -> {"base":1.0,"hr":1.0,"plane":0.5}; unnamed arms keep `default`, except the
    OPT-IN arms (`reader`), which stay at 0 unless named."""
    w = {a: (0.0 if a in OPT_IN_ARMS else float(default)) for a in ARM_ORDER}
    for tok in str(spec or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            print(f"[config] ignoring malformed RSNA_ARM_WEIGHTS token {tok!r}")
            continue
        k, v = tok.split("=", 1)
        k = k.strip().lower()
        if k not in w:
            print(f"[config] ignoring unknown arm {k!r} in RSNA_ARM_WEIGHTS")
            continue
        try:
            w[k] = float(v)
        except ValueError:
            print(f"[config] ignoring non-numeric weight in {tok!r}")
    return w


ARM_WEIGHTS = parse_arm_weights(os.environ.get("RSNA_ARM_WEIGHTS"))
_arms_env = os.environ.get("RSNA_ARMS")
ARMS_ENABLED = [a for a in ARM_ORDER
                if (a in {t.strip().lower() for t in _arms_env.split(",")} if _arms_env else True)]


def parse_tta_spec(spec, default=1):
    """"3" -> every arm S=3;  "base:3,hr:2" -> per arm;  "2,hr:1" -> default 2, hr overridden.

    S is the number of slot-sampling variants an arm evaluates. Variant 0 IS the baseline eval
    sampling (`tta.sampling_variants(S)[0].is_identity`), so S=1 runs the literal pre-TTA code
    path -- that is what makes the RSNA_TTA-unset output byte-identical to the pre-TTA notebook.
    """
    s = {a: int(default) for a in ARM_ORDER}
    for tok in str(spec or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok or "=" in tok:
            k, v = re.split(r"[:=]", tok, 1)
            k = k.strip().lower()
            if k not in s:
                print(f"[config] ignoring unknown arm {k!r} in RSNA_TTA")
                continue
            try:
                s[k] = max(1, int(v))
            except ValueError:
                print(f"[config] ignoring non-integer S in RSNA_TTA token {tok!r}")
            continue
        try:                                    # a bare number is the default for every arm
            s = {a: max(1, int(tok)) for a in ARM_ORDER}
        except ValueError:
            print(f"[config] ignoring malformed RSNA_TTA token {tok!r}")
    for a in ARM_ORDER:                          # arms without a wrapper stay on the single pass
        if a not in TTA_CAPABLE and s[a] > 1:
            print(f"[config] RSNA_TTA: arm {a!r} has no TTA wrapper -> forced back to S=1")
            s[a] = 1
    return s


# Slot-sampling TTA: S deterministic re-samplings per arm, rank-averaged inside the arm before
# the across-arm combiner runs. S=1 is byte-identical to no TTA; S>1 costs ~S x decode+forward.
# The next line is the packagers' sed anchor (same shape as RSNA_ARM_WEIGHTS): they rewrite the
# default "1" to the S they want baked in. The anchor text is deliberately NOT repeated in this
# comment, so `grep -c` on it is exactly 1 in the generated .py and .ipynb -- see
# reports/multi_notebook_v2.md for the exact sed line.
TTA_S = parse_tta_spec(os.environ.get("RSNA_TTA", "1"))
TTA_COMBINE = os.environ.get("RSNA_TTA_COMBINE", "rank").strip().lower()
if max(TTA_S.values()) > 1:
    print(f"[config] TTA: {dict(TTA_S)} slot sampling(s) per arm, {TTA_COMBINE}-averaged "
          f"across variants; arms without a wrapper {tuple(a for a in ARM_ORDER if a not in TTA_CAPABLE)} "
          f"stay at S=1")


# Where the attached weights dataset actually mounts. Checked FIRST, shallowly.
WEIGHTS_HINTS = ("rsna-knee-weights-v1", "datasets/abdullllkhan/rsna-knee-weights-v1")
# NEVER descend into these: they are the DICOM trees (~819k files in the competition mount).
PRUNE_DIRS = {"train_series", "test_series", "competitions"}


def _is_member_ckpt(fn):
    return fn.endswith(".pt") and "_fold" in fn and not fn.endswith("_last.pt")


def _shallow_pt(root, max_depth=2):
    """Checkpoints at most `max_depth` directory levels under `root`. Listing directories is
    cheap; it is descending into train_series/ that costs minutes."""
    hits = []
    if not os.path.isdir(root):
        return hits
    for dp, dn, fns in os.walk(root):
        rel = os.path.relpath(dp, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        dn[:] = [] if depth >= max_depth else [d for d in dn if d not in PRUNE_DIRS]
        hits += [os.path.join(dp, fn) for fn in fns if _is_member_ckpt(fn)]
    return hits


def find_weights(search):
    """Collect trained checkpoints (best-by-fold; skip *_last.pt) from the attached dataset.

    Fast path: the known weights-dataset mount points, listed shallowly. The old unbounded
    `os.walk("/kaggle/input")` cost a MEASURED 341 s per run, because /kaggle/input carries
    the whole competition dataset (~819k training DICOMs) and we were walking all of it to
    find ten .pt files in a 12 MB attached dataset -- 0.0048 AUC-equivalent burned on every
    submission (reports/efficiency_prize.md §4). Only if the fast path comes up empty do we
    walk recursively, and even then the DICOM trees are pruned.
    """
    t0 = time.time()
    cands = [os.path.join(search, h) for h in WEIGHTS_HINTS]
    cands += [os.path.join("/kaggle/input", h) for h in WEIGHTS_HINTS]
    cands += sorted(glob.glob(os.path.join(search, "*weights*")))
    cands += sorted(glob.glob(os.path.join(search, "datasets", "*", "*weights*")))
    cands.append(search)  # the mount root itself -- still depth-bounded, still pruned
    hits, seen = [], set()
    for d in cands:
        rd = os.path.realpath(d)
        if rd in seen or not os.path.isdir(rd):
            continue
        seen.add(rd)
        hits += _shallow_pt(d)
    how = "shallow"
    if not hits:  # last resort: recursive, but never into a DICOM tree
        how = "pruned-walk"
        for dp, dn, fns in os.walk(search):
            dn[:] = [d for d in dn if d not in PRUNE_DIRS]
            hits += [os.path.join(dp, fn) for fn in fns if _is_member_ckpt(fn)]
    out = sorted(set(hits))
    print(f"[time] stage=weights_search elapsed={time.time() - t0:.1f}s via={how} "
          f"found={len(out)}", flush=True)
    return out


def peek_checkpoint(path):
    """Read a checkpoint's metadata (args/cfg/top-level keys) without keeping the weights.

    mmap keeps this cheap (no full read into RAM); some torch builds/checkpoints refuse it,
    hence the plain-load fallback."""
    st = None
    try:
        st = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        st = torch.load(path, map_location="cpu", weights_only=False)
    try:
        meta = {
            "args": dict(st.get("args") or {}),
            "cfg": dict(st.get("cfg") or {}),
            "keys": set(st.keys()),
        }
    finally:
        del st
        gc.collect()
    return meta


def detect_arm(meta):
    """Which predictor this checkpoint belongs to -> (arm, size, k_desc).

    Decided from the checkpoint's OWN metadata, never from its file name:
      * v2      -- train_v2 writes args["mode"]="v2"; its bag is 5 fixed (plane x fluid) SLOTS
                  of 64 slices total, evenly spaced in anatomical rank over span 0.02-0.98,
                  140 mm physical crop + per-series percentile pixels, canonicalised to a
                  right knee, plus `slot_id`/`pos`/`valid` side inputs the model consumes.
      * ordered -- train_ordered writes args["mode"]="ordered" (and args["order"] +
                  args["coverage"]); its slots are chosen in ANATOMICAL rank order.
      * 25d    -- train_25d writes args["mode"]="2.5d"; its three input channels are
                  [prev, cur, next] in anatomical order, not a replicated grayscale slice.
      * plane  -- train_plane writes args["plane_mode"]/args["k_per_plane"] and a
                  `plane_weights` tensor; its cfg carries k_per_plane/fusion.
      * hr     -- train_hr writes args["size"]; anything other than 224 needs the size-aware
                  loader (a 384 model fed 224 crops is silently wrong, not merely worse).
      * base   -- the 224 baseline: args has `k`, no `size`. Also where a size==224 train_hr
                  checkpoint lands, since read_slice_hr(224) == dataset._read_slice_dicom
                  bit-for-bit (reports/decode_raw.md) -- it joins the v7 224 decode pass
                  instead of opening a second one.

    ORDER MATTERS. `v2`, `ordered` and `25d` are tested FIRST because all three carry
    args["size"] exactly like the plain HR arm: under the old `size != 224 -> hr` rule a 384
    checkpoint from any of them silently landed in `hr` and was then fed the WRONG INPUT
    (lexicographic slot positions / gray x 3 / a lexicographic 24-slice bag with no slot
    embeddings) -- plausible numbers from a distribution the network never saw
    (reports/arm_25d.md §6, reports/arm_ordered.md §6, reports/arm_v2.md §8(3)). `v2` is
    tested before the other two only for readability: all three predicates read args["mode"]
    and are mutually exclusive ("v2" / "ordered" / "2.5d"), and each comes from its own arm's
    module, so this stays one source of truth.
    """
    args, cfg, keys = meta["args"], meta["cfg"], meta["keys"]
    size = int(args.get("size", 224))
    if is_v2_meta(args, cfg, keys):
        return ("v2", size,
                f"K={int(args.get('k_eval', 64))} slots={args.get('slot_counts')}")
    if is_ordered_meta(args, cfg, keys):
        return ("ordered", size,
                f"k={int(args.get('k', 24))} coverage={args.get('coverage', 'even')}")
    if is_25d_meta(args, cfg, keys):
        return ("25d", size,
                f"k={int(args.get('k', 24))} step={int(args.get('neighbour_step', 1))}")
    is_plane = ("plane_mode" in args or "k_per_plane" in args
                or "k_per_plane" in cfg or "fusion" in cfg or "plane_weights" in keys)
    if is_plane:
        kpp = args.get("k_per_plane") or cfg.get("k_per_plane") or (8, 8, 8)
        return "plane", size, f"k_per_plane={tuple(int(v) for v in kpp)}"
    if size != 224:
        return "hr", size, f"k={int(args.get('k', 24))}"
    return "base", 224, f"k={int(args.get('k', 24))}"


def probe_device(device):
    """Verbatim from infer.run_submission: some Kaggle GPUs (P100 sm_60) are not in this
    torch build's arch list -> CUDA crashes and only the 0.5 file survives. Probe, else CPU."""
    if not str(device).startswith("cuda"):
        return device
    try:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            sm = f"sm_{cap[0]}{cap[1]}"
            archs = list(torch.cuda.get_arch_list())
            if sm in archs:
                _t = torch.zeros(8, device=device)
                _ = (_t + 1).sum().item()
                return device
            print(f"[device] GPU {sm} not in supported {archs}; using CPU")
        else:
            print("[device] cuda not available; using CPU")
    except Exception as e:
        print(f"[device] CUDA probe failed ({type(e).__name__}: {e}); using CPU")
    return "cpu"


def arm_batch(arm, size):
    """Per-arm batch size; a 384 bag is ~3x the pixels of a 224 one, so shrink by default."""
    env = os.environ.get("RSNA_BATCH_" + arm.upper())
    if env:
        return max(1, int(env))
    if int(size) >= 512:
        return max(1, BATCH // 8)
    if int(size) >= 384:
        return max(1, BATCH // 4)
    return max(1, BATCH)


def run_arm(arm, paths, root, device, study_ids):
    """Load one arm's members, fingerprint-check them (ADVISORY), predict, free the models.

    Returns a [ID_COL + 12] probability/rank frame, or None if the arm fails -- a failing arm
    must never take the other arms (or the 0.5 file) down with it.
    """
    size = 224
    members = None
    S = int(TTA_S.get(arm, 1))
    if arm == "reader":
        print("[tta] arm=reader S=1 (pinned: the validated reader configuration is ONE k=12 pass; "
              "RSNA_TTA does not apply to it)")
    else:
        print(f"[tta] arm={arm} S={S}" + ("" if S > 1 else
              (" (baseline sampling, TTA off)" if arm in TTA_CAPABLE else
               " (no TTA wrapper for this arm)")))
    try:
        _t = time.time()
        if arm == "reader":
            # goodpjw2008's public reader (module `reader_gpjw`): 3 checkpoints (name/args/sha256-checked, strict
            # weights_only=True load), k=12, fp16 autocast, probability mean, S=1 -- the validated configuration.
            # It returns a frame of the VALID studies only; its guard verdict decides whether the arm is kept.
            b = max(1, int(os.environ.get("RSNA_BATCH_READER", "4")))  # the released infer.py: bs=4
            nw = int(os.environ.get("RSNA_WORKERS_READER") or os.environ.get("RSNA_WORKERS") or 4)
            deadline = _T_START + float(os.environ.get("RSNA_READER_DEADLINE_H", "8.0")) * 3600.0
            res = reader_gpjw.predict_reader(paths, root, "test", study_ids, device=device, batch=b,
                                             workers=nw, deadline=deadline)
            try:
                reader_gpjw.write_reader_outputs(res, os.path.dirname(os.path.abspath(OUT)))
            except Exception as e:
                print(f"[reader] diagnostics not written ({type(e).__name__}: {e})")
            _stage("predict[reader]", _t)
            return res["frame"] if res["guard_ok"] else None
        if arm == "base":
            members = load_members(paths, device=device)
            size = 224
            for i, m in enumerate(members):
                if not check_fingerprint(m, device):
                    print(f"[fingerprint] base member {i} mismatch -> ADVISORY "
                          f"(cross-arch drift expected); proceeding with REAL predictions")
            _stage("member_load[base]", _t)
            b = arm_batch(arm, size)
            print(f"[arm base] {len(members)} member(s), batch={b}, workers={WORKERS}")
            _t = time.time()
            if S > 1:
                # S passes of the LITERAL v7 combiner, per-column rank-averaged across variants
                out = predict_studies_base_tta(root, "test", members, variants=S, batch=b,
                                               workers=WORKERS, device=device,
                                               study_ids=study_ids, variant_combine=TTA_COMBINE)
            else:
                out = predict_studies(root, "test", members, batch=b, workers=WORKERS,
                                      device=device, study_ids=study_ids)
            _stage("predict[base]", _t)
            return out
        if arm == "hr":
            members = load_members_hr(paths, device=device)
            size = max(int(m.size) for m in members)
            for m in members:
                if not check_fingerprint_hr(m, device):
                    print(f"[fingerprint] {m.name} mismatch -> ADVISORY "
                          f"(cross-arch drift expected); proceeding with REAL predictions")
            _stage("member_load[hr]", _t)
            b = arm_batch(arm, size)
            print(f"[arm hr] {len(members)} member(s), max size={size}, batch={b}, "
                  f"workers={WORKERS}, member-mode={MEMBER_MODE}")
            _t = time.time()
            if S > 1:
                out = predict_studies_tta(root, "test", members, variants=S, batch=b,
                                          workers=WORKERS, device=device, study_ids=study_ids,
                                          mode=MEMBER_MODE, variant_combine=TTA_COMBINE)
            else:
                out = predict_studies_hr(root, "test", members, batch=b, workers=WORKERS,
                                         device=device, study_ids=study_ids, mode=MEMBER_MODE)
            _stage("predict[hr]", _t)
            return out
        if arm == "25d":
            members = load_members_25d(paths, device=device)
            size = max(int(m.size) for m in members)
            for m in members:
                if not check_fingerprint_25d(m, device):
                    print(f"[fingerprint] {m.name} mismatch -> ADVISORY "
                          f"(cross-arch drift expected); proceeding with REAL predictions")
            _stage("member_load[25d]", _t)
            b = arm_batch(arm, size)
            print(f"[arm 25d] {len(members)} member(s), max size={size}, batch={b}, "
                  f"workers={WORKERS}, member-mode={MEMBER_MODE}")
            _t = time.time()
            # No order_parquet / dicom_root: on the Kaggle mount the slices ARE DICOM, so the
            # resolver reads the three ordering tags per series from the mount itself.
            out = predict_studies_25d(root, "test", members, batch=b, workers=WORKERS,
                                      device=device, study_ids=study_ids, mode=MEMBER_MODE)
            _stage("predict[25d]", _t)
            return out
        if arm == "ordered":
            members = load_members_ordered(paths, device=device)
            size = max(int(m.size) for m in members)
            for m in members:
                if not check_fingerprint_ordered(m, device):
                    print(f"[fingerprint] {m.name} mismatch -> ADVISORY "
                          f"(cross-arch drift expected); proceeding with REAL predictions")
            _stage("member_load[ordered]", _t)
            b = arm_batch(arm, size)
            print(f"[arm ordered] {len(members)} member(s), max size={size}, batch={b}, "
                  f"workers={WORKERS}, member-mode={MEMBER_MODE}")
            _t = time.time()
            # Same as 25d: anatomical order comes from the mount's own DICOM headers, so
            # data/series_order.parquet must NOT be shipped in the weights dataset.
            out = predict_studies_ordered(root, "test", members, batch=b, workers=WORKERS,
                                          device=device, study_ids=study_ids, mode=MEMBER_MODE)
            _stage("predict[ordered]", _t)
            return out
        if arm == "v2":
            members = load_members_v2(paths, device=device)
            size = max(int(m.size) for m in members)
            for m in members:
                if not check_fingerprint_v2(m, device):
                    print(f"[fingerprint] {m.name} mismatch -> ADVISORY "
                          f"(cross-arch drift expected); proceeding with REAL predictions")
            _stage("member_load[v2]", _t)
            b = arm_batch(arm, size)
            print(f"[arm v2] {len(members)} member(s), max size={size}, batch={b}, "
                  f"workers={WORKERS}, member-mode={MEMBER_MODE}")
            _t = time.time()
            # NOTE the argument order: `predict_studies_v2` takes the MEMBERS first
            # (reports/arm_v2.md §8 edit 4), unlike every other arm's predictor.
            # No side_parquet / order_parquet / dicom_root: on the Kaggle mount the slices ARE
            # DICOM, so slice order comes from `order.series_ranks_from_dicom` and laterality
            # from `laterality.side_from_series_meta` over the headers the decode already read
            # -- neither data/series_order.parquet nor data/study_side.parquet is shipped.
            # Members are grouped first by DECODE GROUP `(crop_mm, pct)` -- one decode per
            # group per study, at `decode_size = max(CACHE_SIZE, max member size)` of that
            # group -- and inside a group by MemberV2.signature = (size, mean, std), which
            # differ only by the INTER_AREA resize in `_stack_to_tensor`. So a 384 + 336 mix
            # at one crop still costs ONE decode, while a 110 mm + 140 mm mix costs TWO (the
            # `[predict] ... decode groups: [(crop_mm, decode_size, n_members)]` banner says
            # how many, and `[time] stage=v2_decode[c<crop>@<size>]` says what each cost).
            # Before 2026-09-20 a crop mix was decoded ONCE with member 0's crop_mm, which is
            # how the c110 models were fed 140 mm crops -- reports/crop_fix.md.
            # `tta=S` shares that same decode across the S slot-phase variants too, so S
            # multiplies the v2 FORWARD only -- `[time] stage=v2_decode` stays flat while
            # `stage=v2_forward` scales with S (reports/tta_v2.md). S=1 is the literal
            # pre-TTA call.
            out = predict_studies_v2(members, root, "test", batch=b, workers=WORKERS,
                                     device=device, study_ids=study_ids, mode=MEMBER_MODE,
                                     tta=S, tta_combine=TTA_COMBINE)
            _stage("predict[v2]", _t)
            return out
        if arm == "plane":
            members = load_members_plane(paths, device=device)
            size = max(int(m.size) for m in members)
            for m in members:
                if not check_fingerprint_plane(m, device):
                    print(f"[fingerprint] {m.name} mismatch -> ADVISORY "
                          f"(cross-arch drift expected); proceeding with REAL predictions")
            _stage("member_load[plane]", _t)
            b = arm_batch(arm, size)
            print(f"[arm plane] {len(members)} member(s), max size={size}, batch={b}, "
                  f"workers={WORKERS}, member-mode={MEMBER_MODE}")
            _t = time.time()
            out = predict_studies_plane(root, "test", members, batch=b, workers=WORKERS,
                                        device=device, study_ids=study_ids, mode=MEMBER_MODE)
            _stage("predict[plane]", _t)
            return out
        raise ValueError(f"unknown arm {arm!r}")
    finally:
        # Free the arm's weights before the next arm loads (Kaggle RAM is the binding limit
        # once three arms of EfficientNetV2-S ensembles are in play).
        if members is not None:
            for m in members:
                try:
                    m.model.to("cpu")
                except Exception:
                    pass
            del members
        gc.collect()
        if str(device).startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def combine_frames(frames, used_arms, tag=""):
    """The across-arm combiner -> (preds, used_arms). Without a ROW_OPTIONAL arm this is the pre-reader code
    path statement for statement (rank_average_frames over the inner join; missing studies -> 0.5 via reindex).
    A ROW_OPTIONAL arm (the reader) may lack single studies: the join is taken over the OTHER arms, and a study
    the reader lacks averages the other arms only, weights renormalised (rank_average_frames_partial, which
    returns rank_average_frames itself when the reader covers every study)."""
    w = [ARM_WEIGHTS[a] for a in used_arms]
    print(f"[combine] {tag}per-column rank average over arms {used_arms} with weights {w}")
    opt = [a in ROW_OPTIONAL_ARMS for a in used_arms]
    if all(opt):  # only the reader survived: nothing to fall back to, plain average
        opt = [False] * len(opt)
    req = [i for i, o in enumerate(opt) if not o]
    if len(req) > 1:
        rf, ra = [frames[i] for i in req], [used_arms[i] for i in req]
        counts = [len(f) for f in rf]
        common = set(rf[0][ID_COL].astype(str))
        for f in rf[1:]:
            common &= set(f[ID_COL].astype(str))
        print(f"[combine] {tag}arm row counts {dict(zip(ra, counts))}; common studies {len(common)}")
        if len(common) < max(counts):
            print(f"[combine] WARNING: {max(counts) - len(common)} study(ies) missing from at "
                  f"least one arm; they fall back to 0.5 via reindex")
        if not common:  # never ship an empty ensemble: keep the widest single arm instead
            best = req[int(np.argmax(counts))]
            print(f"[combine] EMPTY intersection -> falling back to arm {used_arms[best]} alone")
            frames, used_arms, w, opt = [frames[best]], [used_arms[best]], [ARM_WEIGHTS[used_arms[best]]], [False]
    if not any(opt):
        return rank_average_frames(frames, weights=w), used_arms
    ids_req = set(frames[req[0]][ID_COL].astype(str))
    for i in req[1:]:
        ids_req &= set(frames[i][ID_COL].astype(str))
    for i, o in enumerate(opt):
        if o:
            n_have = len(ids_req & set(frames[i][ID_COL].astype(str)))
            print(f"[combine] {tag}arm {used_arms[i]} covers {n_have}/{len(ids_req)} studies"
                  + ("" if n_have == len(ids_req) else
                     f"; {len(ids_req) - n_have} fall back to the other arms (weights renormalised)"))
    return rank_average_frames_partial(frames, w, opt, log=None), used_arms


def _fallback_test_df():
    root = find_data_root("/kaggle/input") or os.environ.get("RSNA_INPUT_ROOT")
    for cand in [root, "/kaggle/input/rsna-knee-abnormality-detection",
                 "/kaggle/input/rsna-knee-abnormalities-detection"]:
        if cand and os.path.exists(os.path.join(cand, "test.csv")):
            return pd.read_csv(os.path.join(cand, "test.csv"))
    raise FileNotFoundError("no test.csv found for fallback")


try:
    _t = time.time()
    root = os.environ.get("RSNA_INPUT_ROOT") or find_data_root("/kaggle/input")
    if root is None:
        raise FileNotFoundError("could not locate data mount (test.csv + test_series/)")
    print(f"[mount] input_root={root}")
    test_df = pd.read_csv(os.path.join(root, "test.csv"))
    write_benchmark_submission(test_df, OUT)  # valid 0.5 file FIRST, before anything expensive
    print("[driver] wrote 0.5 benchmark; test rows =", len(test_df))
    print(f"[self-size] test.csv rows = {len(test_df)}")
    _stage("mount_discovery", _t)

    weights = find_weights(WEIGHTS_DIR)  # prints its own [time] stage=weights_search line
    # The reader's checkpoints live in their own public dataset. Keep them OUT of the generic detection (its peek
    # uses the full unpickler and would route them to `base`) and find them with the reader's own shallow search,
    # which knows both mount layouts (<root>/<slug> and <root>/datasets/<owner>/<slug>).
    reader_paths = [p for p in weights if reader_gpjw.is_reader_file(p)]
    weights = [p for p in weights if not reader_gpjw.is_reader_file(p)]
    try:  # a failure here may cost the reader, never the other arms
        reader_paths += reader_gpjw.find_reader_checkpoints((WEIGHTS_DIR, "/kaggle/input"))
        _by_name = {}
        for _p in sorted(reader_paths):  # one path per file name (the same file can be reached twice)
            _by_name.setdefault(os.path.basename(_p), _p)
        reader_paths = [_by_name[_k] for _k in sorted(_by_name)]
    except Exception:
        print("[weights] reader checkpoint search FAILED -> no reader members:")
        traceback.print_exc()
        reader_paths = []
    if not weights and not reader_paths:
        raise FileNotFoundError(f"no checkpoints under {WEIGHTS_DIR}")
    print(f"[weights] {len(weights)} checkpoint(s) under {WEIGHTS_DIR}")

    _t = time.time()
    by_arm = {a: [] for a in ARM_ORDER}
    for cp in weights:
        try:
            arm, size, kdesc = detect_arm(peek_checkpoint(cp))
        except Exception as e:
            print(f"[weights] SKIP {os.path.basename(cp)}: unreadable ({type(e).__name__}: {e})")
            continue
        by_arm[arm].append(cp)
        print(f"[weights] {os.path.basename(cp):46s} -> arm={arm:7s} size={size:4d} {kdesc}")
    for cp in sorted(reader_paths, key=os.path.basename):
        try:  # member = file name cnxt_v0_fold<k>.pt AND args (name, backbone, res, k_eval, fold), weights_only
            fold = reader_gpjw.check_reader_args(cp, reader_gpjw.reader_ckpt_args(cp))
        except Exception as e:
            print(f"[weights] SKIP {os.path.basename(cp)}: not the validated reader ({type(e).__name__}: {e})")
            continue
        by_arm["reader"].append(cp)
        print(f"[weights] {os.path.basename(cp):46s} -> arm={'reader':7s} size={reader_gpjw.READER_RES:4d} "
              f"k={reader_gpjw.READER_K} fold={fold} {reader_gpjw.READER_NAME} {reader_gpjw.READER_BACKBONE} "
              f"({os.path.dirname(cp)})")
    _stage("arm_detect", _t)

    plan = []
    for arm in ARM_ORDER:
        n = len(by_arm[arm])
        w = ARM_WEIGHTS[arm]
        if n == 0:
            print(f"[plan] arm {arm:7s}: no members -> skipped")
        elif arm not in ARMS_ENABLED:
            print(f"[plan] arm {arm:7s}: {n} member(s) but disabled by RSNA_ARMS -> skipped")
        elif w <= 0:
            print(f"[plan] arm {arm:7s}: {n} member(s) but weight {w} <= 0 -> skipped")
        else:
            print(f"[plan] arm {arm:7s}: {n} member(s), weight {w}")
            plan.append(arm)
    if not plan:
        raise RuntimeError("no arm has usable members (check RSNA_ARMS / RSNA_ARM_WEIGHTS)")

    device = probe_device(DEVICE)
    print(f"[device] using {device}")

    study_ids = test_df[ID_COL].astype(str).tolist()
    frames, used_arms = [], []
    for arm in plan:
        if arm in ROW_OPTIONAL_ARMS and frames:
            # Crash fallback, written BEFORE the reader runs: the other arms' finished submission goes to OUT now,
            # so a hard crash / kill inside the reader leaves exactly that file instead of the 0.5 benchmark.
            # (Python-level reader failures are caught below and drop only the reader, like any arm.)
            try:
                p0, a0 = combine_frames(frames, used_arms, tag="pre-reader ")
                write_submission(p0, test_df, OUT)
                write_submission(p0, test_df, os.path.join(os.path.dirname(os.path.abspath(OUT)),
                                                           "submission_pre_reader.csv"))
                print(f"[driver] pre-reader submission written (arms {a0}): OUT holds it if the reader dies")
            except Exception:
                print("[driver] pre-reader submission FAILED (continuing):")
                traceback.print_exc()
        _t = time.time()
        try:
            f = run_arm(arm, by_arm[arm], root, device, study_ids)
        except Exception:
            print(f"[arm {arm}] FAILED -> dropped from the ensemble:")
            traceback.print_exc()
            _stage(f"arm[{arm}]", _t)
            continue
        _stage(f"arm[{arm}]", _t)
        if f is None or len(f) == 0:
            if arm in ROW_OPTIONAL_ARMS:
                print(f"[arm {arm}] dropped (guard tripped or no valid study) -> the ensemble is the other arms")
                continue
            print(f"[arm {arm}] produced 0 rows -> dropped from the ensemble")
            continue
        frames.append(f)
        used_arms.append(arm)
        print(f"[arm {arm}] OK: {len(f)} studies")
        if DUMP_ARMS:
            os.makedirs(DUMP_ARMS, exist_ok=True)
            f.to_pickle(os.path.join(DUMP_ARMS, f"arm_{arm}.pkl"))
    if not frames:
        raise RuntimeError("every arm failed")

    _t = time.time()
    preds, used_arms = combine_frames(frames, used_arms)

    sub = write_submission(preds, test_df, OUT)  # pct-rank + reindex to test.csv + fillna(0.5)
    _stage("combine_write", _t)
    print(f"[submission] wrote {OUT} rows={len(sub)} cols={list(sub.columns)}")
    print(f"[submission] arms used: {used_arms} | members: "
          f"{ {a: len(by_arm[a]) for a in used_arms} }")
except Exception:
    traceback.print_exc()
    if not os.path.exists(OUT):
        write_benchmark_submission(_fallback_test_df(), OUT)
        print("[driver] fallback 0.5 submission written")

# ---- final schema assertions ----
sub = pd.read_csv(OUT)
assert list(sub.columns) == [ID_COL] + TARGETS, ("bad columns", list(sub.columns))
vals = sub[TARGETS].values
assert (vals >= 0).all() and (vals <= 1).all(), "values out of [0,1]"
assert not np.isnan(vals).any(), "NaN in submission"
assert sub[ID_COL].is_unique, "duplicate StudyInstanceUID"
print("[driver] submission OK: rows=%d cols=%d" % (len(sub), sub.shape[1]))
print(sub.head())
_stage("total", _T_START)
'''


def make_notebook(bundle: str, driver: str) -> dict:
    def code(src):
        return {"cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": src.splitlines(keepends=True)}

    def md(src):
        return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

    cells = [
        md("# RSNA Knee Abnormality Detection - MULTI-ARM inference notebook\n"
           "\n"
           "Self-contained (no `src` import, no network). Package source is inlined below:\n"
           "`decode` / `order` / `laterality` / `dataset` / `dataset_hr` / `tta` /\n"
           "`dataset_25d` / `dataset_ordered` / `dataset_plane` / `dataset_v2` / `model` /\n"
           "`model_plane` / `model_v2` / `infer` / `infer_hr` / `infer_25d` /\n"
           "`infer_ordered` / `infer_plane` / `infer_v2` / `infer_tta` / `combine_partial`, plus\n"
           "`reader_gpjw` in its own namespace: goodpjw2008's public 2.5D ConvNeXt-T reader\n"
           "(Apache-2.0; weights from the attached dataset goodpjw2008/rsna-knee-2-5d-convnext-reader),\n"
           "the opt-in `reader` arm.\n"
           "\n"
           "Every checkpoint in the attached weights dataset is loaded; its ARM is detected\n"
           "from its own `args`/`cfg` (base 224 / hr 384+ / 2.5D / ordered-sampler / v2 slot\n"
           "sampler / plane); each arm runs its own decode-once loader; the arms are combined\n"
           "by a per-column percentile-rank average with per-arm weights (`RSNA_ARM_WEIGHTS`,\n"
           "`RSNA_ARMS`). The 2.5D, ordered and v2 arms read slice order from the mount's own\n"
           "DICOM headers, and v2 also reads laterality from them - neither\n"
           "`data/series_order.parquet` nor `data/study_side.parquet` may be shipped.\n"
           "\n"
           "`RSNA_TTA` (default 1 = off, byte-identical to no TTA) runs the base, hr and v2\n"
           "arms over S slot-sampling variants and rank-averages them inside the arm; S can be\n"
           "set globally (`3`) or per arm (`base:3,v2:2`). Cost is ~linear in S for base/hr;\n"
           "for v2 it is S x forward only (the S bags share one decode per study).\n"
           "\n"
           "Survival patterns kept from v7: 0.5 file first, mount discovery, self-size to the\n"
           "hidden test, `pretrained=False` weight load, device probe with CPU fallback,\n"
           "capability-aware autocast (bf16 on sm_80+, fp16 on T4/V100, fp32 elsewhere, with a\n"
           "per-batch fp32 recompute guard), ADVISORY fingerprint, rank + reindex + fillna(0.5),\n"
           "schema assert. Per-stage `[time] stage=... elapsed=...s` lines throughout.\n"),
        code(bundle),
        code(driver.lstrip("\n")),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    print(f"bundle modules from {PKG}")
    bundle = build_bundle()
    compile(bundle + "\n\n" + DRIVER.lstrip("\n"), "<submission_multi>", "exec")  # syntax gate
    nb = make_notebook(bundle, DRIVER)
    ipynb_path = os.path.join(HERE, "submission_multi.ipynb")
    with open(ipynb_path, "w") as f:
        json.dump(nb, f, indent=1)
    # standalone .py mirror (same source) for local execution/validation
    py_path = os.path.join(HERE, "submission_multi_standalone.py")
    with open(py_path, "w") as f:
        f.write(bundle + "\n\n" + DRIVER.lstrip("\n"))
    print("wrote", ipynb_path)
    print("wrote", py_path)


if __name__ == "__main__":
    main()
