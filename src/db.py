"""
Day 4: SQLite audit trail. Judging criterion #3 -- "audit trail of every
decision" -- is met by this file alone: every score the API ever produces is
written here before the response goes out, with the reason attached.

Flat script, raw sqlite3, no ORM (per project convention). SQLite is the
right tool here: single process, single file, zero setup, and it survives a
Flask restart mid-demo without losing history.
"""
import json
import sqlite3
import time
from contextlib import contextmanager

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id  TEXT NOT NULL,
    logged_at       REAL NOT NULL,          -- wall-clock time.time() when scored
    sim_hour        REAL,                   -- hour-of-day derived from the txn, for the demo clock
    amount          REAL,
    card1           TEXT,
    score           REAL NOT NULL,
    decision        TEXT NOT NULL,          -- BLOCK | MANUAL_REVIEW | ALLOW
    reason          TEXT,                   -- human sentence, from SHAP top-3
    top_features    TEXT,                   -- JSON list, full SHAP breakdown
    spike_alert     INTEGER NOT NULL DEFAULT 0,
    is_synthetic    INTEGER NOT NULL DEFAULT 0,
    review_status   TEXT                    -- NULL | pending | approved | rejected
);
CREATE INDEX IF NOT EXISTS idx_audit_logged_at ON audit_log(logged_at);
CREATE INDEX IF NOT EXISTS idx_audit_review ON audit_log(review_status);

CREATE TABLE IF NOT EXISTS spike_alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at       REAL NOT NULL,
    entity_col      TEXT NOT NULL,
    entity_key      TEXT NOT NULL,
    count           INTEGER NOT NULL,
    window_seconds  INTEGER NOT NULL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def log_decision(transaction_id, amount, card1, score, decision, reason,
                 top_features, spike_alert=False, is_synthetic=False,
                 sim_hour=None):
    review_status = "pending" if decision == "MANUAL_REVIEW" else None
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO audit_log
               (transaction_id, logged_at, sim_hour, amount, card1, score,
                decision, reason, top_features, spike_alert, is_synthetic,
                review_status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (str(transaction_id), time.time(), sim_hour, amount,
             str(card1) if card1 is not None else None, score, decision,
             reason, json.dumps(top_features), int(spike_alert),
             int(is_synthetic), review_status),
        )
        return cur.lastrowid


def log_spike_alert(entity_col, entity_key, count, window_seconds):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO spike_alerts
               (logged_at, entity_col, entity_key, count, window_seconds)
               VALUES (?,?,?,?,?)""",
            (time.time(), entity_col, str(entity_key), count, window_seconds),
        )


def recent_feed(limit=50):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def review_queue():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE review_status = 'pending' "
            "ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def resolve_review(row_id, status):
    assert status in ("approved", "rejected")
    with get_conn() as conn:
        conn.execute(
            "UPDATE audit_log SET review_status = ? WHERE id = ?",
            (status, row_id),
        )


def recent_spike_alerts(limit=20):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM spike_alerts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def summary_counts():
    with get_conn() as conn:
        row = conn.execute(
            """SELECT
                 COUNT(*) AS n,
                 SUM(CASE WHEN decision='BLOCK' THEN 1 ELSE 0 END) AS n_block,
                 SUM(CASE WHEN decision='MANUAL_REVIEW' THEN 1 ELSE 0 END) AS n_review,
                 SUM(CASE WHEN decision='ALLOW' THEN 1 ELSE 0 END) AS n_allow,
                 SUM(spike_alert) AS n_spike_flagged
               FROM audit_log"""
        ).fetchone()
        return dict(row)


def reset():
    """Wipe the audit trail for a clean demo run. Not called automatically."""
    with get_conn() as conn:
        conn.execute("DELETE FROM audit_log")
        conn.execute("DELETE FROM spike_alerts")


if __name__ == "__main__":
    init_db()
    print(f"Initialized {DB_PATH}")
