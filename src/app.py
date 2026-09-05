"""
Day 4-5: Layer 3 -- Flask API + SQLite audit trail + dashboard.

score_transaction() below is the single place live data is scored: it calls
features.transform() with the pipeline fitted in train_xgb.py, so serving can
never silently drift from training (see features.py's own docstring on why
that matters). Flat Flask: one file, no blueprints, no ORM, per project
convention.

Run:
    cd src && ../.venv/bin/python app.py
Then open http://127.0.0.1:5000/
"""
import json
import time

import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, render_template, request

import db
import features
from config import (API_HOST, API_PORT, ARTIFACTS, BLOCK_THRESHOLD, ID_COL,
                    REVIEW_LOW, TIME_COL)
from explain import Explainer
from spike_detector import SpikeDetector

app = Flask(__name__)

MODEL_PATH = ARTIFACTS / "xgb_model.json"

print("Loading model + feature pipeline ...")
_model = xgb.XGBClassifier()
_model.load_model(MODEL_PATH)
_meta = features.load_meta()
_explainer = Explainer(_model, _meta["feature_names"])
_spikes = SpikeDetector()
db.init_db()
print("Ready.")


def decide(score):
    """>=0.70 BLOCK, <0.40 ALLOW.

    Everything in between -- including the 0.60-0.70 band that sits between
    the documented MANUAL_REVIEW ceiling and the BLOCK floor -- is treated as
    MANUAL_REVIEW. That is deliberate: this is the "one failure handled
    gracefully" criterion. An uncertain score should fall open to a human,
    not fall open to either extreme.
    """
    if score >= BLOCK_THRESHOLD:
        return "BLOCK"
    if score < REVIEW_LOW:
        return "ALLOW"
    return "MANUAL_REVIEW"


def score_transaction(raw, is_synthetic=False):
    """raw: dict of one transaction's raw columns, same schema as the joined
    transaction+identity CSVs (a subset is fine -- missing columns become
    NaN in features.transform, which XGBoost handles natively).

    Returns the full audit record that also gets written to SQLite.
    """
    df = pd.DataFrame([raw])
    txn_id = raw.get(ID_COL) or raw.get("transaction_id") \
        or f"live-{int(time.time() * 1000)}"
    amount = float(raw.get("TransactionAmt") or 0)
    card1 = raw.get("card1")

    X = features.transform(df, _meta)
    score = float(_model.predict_proba(X)[:, 1][0])
    decision = decide(score)

    top_features = _explainer.explain_row(X, top_k=3)
    reason = "; ".join(f["phrase"] for f in top_features)

    dt = raw.get(TIME_COL)
    sim_hour = float((dt / 3600) % 24) if dt is not None else None

    # Layer 2: feed this score into the spike detector regardless of decision
    # -- a spike is about the pattern of risky scores, not just outright
    # blocks.
    spike_alert = False
    fired = _spikes.observe(card1, time.time(), score)
    if fired:
        spike_alert = True
        db.log_spike_alert(fired["entity_col"], fired["entity_key"],
                           fired["count"], fired["window_seconds"])
    elif _spikes.active_alert_for(card1):
        spike_alert = True

    row_id = db.log_decision(
        txn_id, amount, card1, score, decision, reason, top_features,
        spike_alert=spike_alert, is_synthetic=is_synthetic, sim_hour=sim_hour,
    )

    return {
        "id": row_id,
        "transaction_id": str(txn_id),
        "amount": amount,
        "card1": card1,
        "score": score,
        "decision": decision,
        "reason": reason,
        "top_features": top_features,
        "spike_alert": spike_alert,
        "spike_fired": fired,
        "is_synthetic": is_synthetic,
    }


# --- routes ------------------------------------------------------------

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/score", methods=["POST"])
def api_score():
    raw = request.get_json(force=True)
    is_synthetic = bool(raw.pop("_synthetic", False))
    return jsonify(score_transaction(raw, is_synthetic=is_synthetic))


@app.route("/api/feed")
def api_feed():
    limit = int(request.args.get("limit", 50))
    return jsonify(db.recent_feed(limit))


@app.route("/api/queue")
def api_queue():
    return jsonify(db.review_queue())


@app.route("/api/queue/<int:row_id>/resolve", methods=["POST"])
def api_resolve(row_id):
    status = (request.get_json(force=True) or {}).get("status")
    db.resolve_review(row_id, status)
    return jsonify({"ok": True})


@app.route("/api/spikes")
def api_spikes():
    return jsonify(db.recent_spike_alerts())


@app.route("/api/summary")
def api_summary():
    return jsonify(db.summary_counts())


@app.route("/api/metrics")
def api_metrics():
    with open(ARTIFACTS / "xgb_metrics.json") as f:
        return jsonify(json.load(f))


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Wipe the audit trail for a clean demo run. The spike detector's
    in-memory windows are cleared too so old demo runs can't leak into a new
    one."""
    db.reset()
    _spikes._windows.clear()
    _spikes.alerts.clear()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT, debug=False)
