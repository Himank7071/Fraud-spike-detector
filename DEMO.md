# Demo script

Seven beats, ~5 minutes. Say the numbers out loud — including the bad ones.

## Setup (before judges arrive)

```bash
cd src && ../.venv/bin/python app.py        # terminal 1 — leave running
```
Open `http://127.0.0.1:5000/` in a browser. Click **Reset demo** for a clean
audit trail. Keep a second terminal ready for `replay_demo.py`.

If a number looks stale, re-run `train_xgb.py` before demoing — no stale slides.

## 1. Metrics first (held-out, never touched during training)

Point at the KPI row. Say the numbers plainly:

> "Precision 48.4%, recall 46.4%, F1 0.47. The headline number on 3.5% fraud
> is AUC-PR, not ROC-AUC — we get 0.491, which is 2.2x our logistic-regression
> baseline's 0.224. ROC-AUC is 0.883, but that number flatters imbalanced data,
> so we don't lead with it."

## 2. Live stream

```bash
cd src && ../.venv/bin/python replay_demo.py --reset
```
Narrate: this replays Kaggle's **unlabeled** competition test set — a stream,
never a metrics source. Watch the live feed populate in real time.

## 3. Explain a flag

Click any BLOCK or MANUAL_REVIEW row in the feed. The modal shows the top-3
SHAP contributors with signed bars.

> "Flagged because [feature] pushed risk up, [feature] pushed it down — this
> is the model's actual decision decomposed, not a guess after the fact."

## 4. Trigger a spike — the differentiator, don't cut it

`replay_demo.py` injects ~20 synthetic transactions sharing one `card1`
partway through the stream. Call it out as it happens:

> "This burst is synthetic — I'm injecting it live to demonstrate detection.
> Individually each of these might only score 'manual review'; the pattern
> — one card, many transactions, single minutes — is what Layer 2 catches
> that Layer 1 can't."

Watch the red spike banner fire and the entry land in **Spike alerts**.

## 5. Manual review queue — the graceful-failure criterion

Point at an item sitting in the queue (score between 0.40 and 0.70).

> "This wasn't auto-blocked. An uncertain score falls open to a human, not to
> either extreme — that's deliberate, not a gap." Approve or reject it live.

## 6. Audit trail

Every row in the live feed is a permanent SQLite record — transaction,
timestamp, score, decision, and the reason, queryable after the fact.

## 7. False-positive cost, stated out loud

> "At our 0.70 threshold: ₹5,31,421 in legitimate revenue blocked across
> 3,036 transactions, computed from actual transaction amounts, not an
> assumed average. Against that, ₹5,98,025 in fraud we still miss. We chose
> this threshold as the precision/recall/cost tradeoff — the sweep table in
> `artifacts/xgb_metrics.json` is what justifies it, not a gut call."

## If something breaks live

Let it fail into the manual review queue and say so — that's the graceful-
degradation story, not a perfect run. Never present the injected spike as a
historical finding; always name it as synthetic.
