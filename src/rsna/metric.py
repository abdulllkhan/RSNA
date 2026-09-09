"""Competition metric = unweighted mean of the 12 per-column ROC-AUCs.

Verified in reports/metric_and_format.md:
    Score = (1/12) * sum_i roc_auc_score(y_true[:, i], y_pred[:, i])
No sample weighting, no per-class weighting. A column with a single class present
is undefined (cannot happen on the full test set, but guard for it in CV).
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import TARGETS, ID_COL


def per_column_auc(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-column ROC-AUC. Columns with only one class present return NaN."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, (y_true.shape, y_pred.shape)
    n_cols = y_true.shape[1]
    out = np.full(n_cols, np.nan, dtype=np.float64)
    for i in range(n_cols):
        yi = y_true[:, i]
        if len(np.unique(yi)) < 2:
            warnings.warn(f"column {i} has a single class present; AUC undefined -> NaN")
            continue
        out[i] = roc_auc_score(yi, y_pred[:, i])
    return out


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean of the per-column ROC-AUCs (skipping single-class columns)."""
    aucs = per_column_auc(y_true, y_pred)
    if np.all(np.isnan(aucs)):
        return float("nan")
    return float(np.nanmean(aucs))


def auc_report(y_true: np.ndarray, y_pred: np.ndarray, targets=TARGETS) -> pd.Series:
    """Per-finding AUC as a labelled Series plus a 'macro' row."""
    aucs = per_column_auc(y_true, y_pred)
    s = pd.Series(aucs, index=list(targets), dtype=np.float64)
    s["macro"] = float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else float("nan")
    return s


def load_gold(train_csv: str) -> pd.DataFrame:
    """The 58 gold studies: rows of train.csv whose 12 label columns are all non-null (0/1)."""
    df = pd.read_csv(train_csv)
    gold = df[df[TARGETS].notna().all(axis=1)].copy()
    gold[TARGETS] = gold[TARGETS].astype(np.float64)
    return gold[[ID_COL] + TARGETS].reset_index(drop=True)


def score_gold(preds: pd.DataFrame, gold: pd.DataFrame):
    """Score a prediction frame against the gold anchor.

    preds: DataFrame with ID_COL + the 12 TARGET columns (probabilities / ranks; only
           the within-column ranking matters for AUC).
    gold:  DataFrame from load_gold().
    Returns (macro_auc, per_finding Series). Only gold studies present in `preds` are used.
    """
    merged = gold.merge(preds[[ID_COL] + TARGETS], on=ID_COL, how="inner", suffixes=("_true", "_pred"))
    if len(merged) == 0:
        return float("nan"), auc_report(np.zeros((0, 12)), np.zeros((0, 12)))
    y_true = merged[[f"{t}_true" for t in TARGETS]].values
    y_pred = merged[[f"{t}_pred" for t in TARGETS]].values
    rep = auc_report(y_true, y_pred)
    return float(rep["macro"]), rep
