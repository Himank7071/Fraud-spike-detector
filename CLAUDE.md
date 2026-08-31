# FraudGuard

Fraud detection for Razorpay Buildathon Track 02 (AI Risk Manager). Solo build,
6-day window. Scores individual transactions **and** detects aggregate spikes.

## Judging criteria — everything serves these four

1. Honest precision/recall on a **held-out** test set
2. False-positive cost quantified in rupees
3. Audit trail of every decision
4. One failure handled gracefully (uncertain scores → manual review, not auto-block)

Defense-only. Nothing offense-capable.

## Architecture

| Layer | What | Where |
|---|---|---|
| 1 | XGBoost per-transaction scoring + SHAP explanations | `src/train_xgb.py` |
| 2 | Sliding-window spike detector (aggregate patterns) | Day 3 |
| 3 | Flask API + dashboard + SQLite audit trail | Day 4–5 |

Decisions: `>0.70` BLOCK · `0.40–0.60` MANUAL_REVIEW · `<0.40` ALLOW

## Hard rules — violating these invalidates our metrics

- **Never shuffle before splitting.** All splits are chronological on `TransactionDT`.
- **Never fit on test.** Feature pipeline, imputers, encoders — all fit on train only.
- **Early stopping uses the validation slice** (last 15% of train), never the test set.
- **Raw `TransactionDT` is never a feature.** Only derived `hour` / `dayofweek`.
- **Don't impute missing values for XGBoost.** Missingness is signal here.
- `test_transaction.csv` / `test_identity.csv` have **no labels** — usable only as
  a replay stream for the demo, never for metrics.

## Data

`IEEE CIS Fraud Detection/` (gitignored, from Kaggle). 590,540 transactions,
3.5% fraud, 434 columns after joining identity on `TransactionID`.
Split: 413,378 train (days 1–120.8) / 177,162 held out (days 120.8–183).

## Running

```bash
cd src && ../.venv/bin/python <script>.py
```

Order: `prep_data.py` → `eda.py` → `baseline.py` → `train_xgb.py`

Python 3.12 venv. xgboost needs `brew install libomp` on macOS.

## Files

- `config.py` — paths, thresholds, ₹1200 AOV. Single source of truth.
- `features.py` — shared pipeline. **Training and the Flask API must both use
  this**, or serving silently drifts from training.
- `metrics.py` — precision/recall/F1/AUC-PR + false-positive cost + threshold sweep.

## Context

Builder is a DTU CS undergrad — first real ML project. Prefer working code over
clever code. Flag leakage risks proactively. Flat Flask, no blueprints, no ORM.
