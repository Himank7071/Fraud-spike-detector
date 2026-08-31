"""
Evaluation. One module so that the number in your notebook, the number in the
demo, and the number you say out loud to a judge are computed by the same code.

Judging criterion: "honest metrics including false-positive cost."
"""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

from config import AVG_ORDER_VALUE_INR, BLOCK_THRESHOLD


def evaluate(y_true, y_prob, threshold=BLOCK_THRESHOLD,
             avg_order_value=AVG_ORDER_VALUE_INR, amounts=None):
    """All headline numbers at one decision threshold.

    amounts: optional per-transaction rupee values. When given, false-positive
    cost uses the REAL blocked amounts rather than an assumed average -- always
    prefer this, it is the more defensible number.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    if amounts is not None:
        amounts = np.asarray(amounts, dtype=float)
        fp_cost = float(amounts[(y_pred == 1) & (y_true == 0)].sum())
        fn_cost = float(amounts[(y_pred == 0) & (y_true == 1)].sum())
        cost_basis = "actual transaction amounts"
    else:
        fp_cost = float(fp * avg_order_value)
        fn_cost = float(fn * avg_order_value)
        cost_basis = f"assumed avg order value INR {avg_order_value:,}"

    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "false_positive_cost_inr": fp_cost,
        "missed_fraud_cost_inr": fn_cost,
        "cost_basis": cost_basis,
    }


def report(name, m):
    print(f"\n--- {name} @ threshold {m['threshold']:.2f} "
          f"({m['n']:,} held-out transactions) ---")
    print(f"  precision {m['precision']:.4f}   "
          f"of every 100 we block, {m['precision'] * 100:.0f} really are fraud")
    print(f"  recall    {m['recall']:.4f}   "
          f"we catch {m['recall'] * 100:.0f} of every 100 actual frauds")
    print(f"  f1        {m['f1']:.4f}")
    print(f"  AUC-PR    {m['auc_pr']:.4f}   (threshold-free; the headline "
          "number for imbalanced data)")
    print(f"  ROC-AUC   {m['roc_auc']:.4f}   (reported for comparability; "
          "flattering on imbalanced data)")
    print(f"  confusion  TP {m['tp']:,}  FP {m['fp']:,}  "
          f"FN {m['fn']:,}  TN {m['tn']:,}")
    print(f"  FALSE POSITIVE COST  INR {m['false_positive_cost_inr']:,.0f} "
          f"in legitimate revenue blocked ({m['cost_basis']})")
    print(f"  missed fraud         INR {m['missed_fraud_cost_inr']:,.0f} "
          "in fraud let through")


def threshold_sweep(y_true, y_prob, amounts=None,
                    thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """The table that justifies the chosen threshold to a judge."""
    print(f"\n{'thresh':>7} {'prec':>7} {'recall':>7} {'f1':>7} "
          f"{'FP':>7} {'FN':>7} {'FP cost INR':>14}")
    rows = []
    for t in thresholds:
        m = evaluate(y_true, y_prob, threshold=t, amounts=amounts)
        rows.append(m)
        print(f"{t:>7.2f} {m['precision']:>7.3f} {m['recall']:>7.3f} "
              f"{m['f1']:>7.3f} {m['fp']:>7,} {m['fn']:>7,} "
              f"{m['false_positive_cost_inr']:>14,.0f}")
    return rows
