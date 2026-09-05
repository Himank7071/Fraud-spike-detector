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
| 1 | XGBoost per-transaction scoring + SHAP explanations | `src/train_xgb.py`, `src/explain.py` |
| 2 | Sliding-window spike detector (aggregate patterns) | `src/spike_detector.py` |
| 3 | Flask API + dashboard + SQLite audit trail | `src/app.py`, `src/db.py`, `src/templates/dashboard.html` |

Decisions: `>=0.70` BLOCK · `0.40–0.70` MANUAL_REVIEW · `<0.40` ALLOW. The
0.60–0.70 band (between the documented review ceiling and the block floor) is
folded into MANUAL_REVIEW on purpose — an uncertain score falls open to a
human, never to either extreme (judging criterion #4).

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

Pipeline order: `prep_data.py` → `eda.py` → `baseline.py` → `train_xgb.py`
(each writes to `artifacts/`; only rerun if the data or model changes).

Demo, two terminals:
```bash
cd src && ../.venv/bin/python app.py            # Flask API + dashboard, :5000
cd src && ../.venv/bin/python replay_demo.py --reset   # streams the demo feed
```
Open `http://127.0.0.1:5000/`. `replay_demo.py` streams Kaggle's unlabeled
`test_transaction.csv`/`test_identity.csv` (never a metrics source — see hard
rules) and injects one clearly-tagged synthetic card-testing spike to trigger
Layer 2 live. `--reset` wipes the SQLite audit trail first for a clean run;
the dashboard's "Reset demo" button does the same mid-session.

Python 3.12 venv. xgboost needs `brew install libomp` on macOS.

## Files

- `config.py` — paths, thresholds, ₹1200 AOV, spike-detector and API settings.
  Single source of truth.
- `features.py` — shared pipeline. **Training and the Flask API must both use
  this**, or serving silently drifts from training.
- `metrics.py` — precision/recall/F1/AUC-PR + false-positive cost + threshold sweep.
- `explain.py` — per-transaction SHAP top-3, in plain-English phrases.
- `spike_detector.py` — in-memory sliding window over `card1`; fires when
  enough risky-scored transactions land on one card too fast.
- `db.py` — SQLite audit trail (every decision) + review queue + spike log.
  Raw `sqlite3`, no ORM.
- `app.py` — Flask API + dashboard route. `score_transaction()` is the one
  place live data is scored; it always goes through `features.py`.
- `replay_demo.py` — demo-only replay stream (see Running, above).

## Context

Builder is a DTU CS undergrad — first real ML project. Prefer working code over
clever code. Flag leakage risks proactively. Flat Flask, no blueprints, no ORM.
