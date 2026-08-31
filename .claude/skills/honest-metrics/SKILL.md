---
name: honest-metrics
description: Check any reported fraud-detection number for leakage and cost honesty before it goes in front of a judge. Use when reporting precision/recall/AUC-PR, choosing a decision threshold, or comparing models.
---

# Honest metrics

Run this before any number is quoted, committed, or shown to a judge.

## Leakage checklist

- [ ] Split is chronological on `TransactionDT`, never shuffled
- [ ] Feature pipeline, imputer, scaler, encoders fit on **train only**
- [ ] Early stopping watched the validation slice, not the test set
- [ ] Raw `TransactionDT` is not a feature
- [ ] Test set was touched exactly once, at final evaluation
- [ ] Threshold was chosen on validation, not tuned against test

Any unchecked box means the number is not held-out and must not be reported as such.

## Reporting rules

- **AUC-PR is the headline**, not ROC-AUC. On 3.5% fraud, ROC-AUC flatters badly
  (baseline scored 0.839 ROC-AUC against 0.224 AUC-PR — same model).
- Never report accuracy. "Always legit" scores 96.5%.
- Always state false-positive cost in rupees, computed from **actual transaction
  amounts** where available, not an assumed average. Say which basis was used.
- Report the confusion matrix counts alongside the rates.
- Show the threshold sweep table. The chosen threshold must be justified by the
  precision/recall/cost tradeoff, not asserted.

## Phrasing

State numbers plainly, including the bad ones. "We block 47 legitimate
transactions per 10,000, costing ₹56,400 in blocked revenue; we chose this
threshold deliberately" beats any hedge.
