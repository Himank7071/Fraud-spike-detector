"""
Day 3: Layer 2, the sliding-window spike detector.

Layer 1 (XGBoost) scores one transaction at a time. It will happily give
twenty small charges on a stolen card a score of 0.45 each -- individually
unremarkable, "manual review" at worst. A human fraud analyst would spot the
pattern instantly: the same card, twenty times, in ten minutes. That pattern
is invisible to a per-transaction model and is exactly what this layer adds.

Design
------
Per entity (default: card1) we keep a deque of recent "risky" transaction
timestamps -- risky meaning its Layer 1 score cleared SPIKE_RISKY_THRESHOLD,
the MANUAL_REVIEW floor, not just outright blocks. As each new transaction
arrives:
  1. Drop timestamps older than SPIKE_WINDOW_SECONDS from now.
  2. If this transaction is risky, append it.
  3. If the deque length reaches SPIKE_COUNT_TRIGGER, fire an alert.

This is O(1) amortised per event (each timestamp is pushed and popped once)
and needs no history beyond the window, so it runs fine embedded in the Flask
process on a live stream. It is a pure aggregate-pattern signal, independent
of and complementary to the per-transaction score.
"""
from collections import defaultdict, deque

from config import (SPIKE_COUNT_TRIGGER, SPIKE_GROUP_COL,
                    SPIKE_RISKY_THRESHOLD, SPIKE_WINDOW_SECONDS)


class SpikeDetector:
    """Stateful, in-memory. One instance lives for the life of the Flask
    process (or a replay run) and sees every transaction in order."""

    def __init__(self, window_seconds=SPIKE_WINDOW_SECONDS,
                 count_trigger=SPIKE_COUNT_TRIGGER,
                 risky_threshold=SPIKE_RISKY_THRESHOLD,
                 group_col=SPIKE_GROUP_COL):
        self.window_seconds = window_seconds
        self.count_trigger = count_trigger
        self.risky_threshold = risky_threshold
        self.group_col = group_col
        self._windows = defaultdict(deque)   # entity_key -> deque[timestamp]
        self.alerts = []                     # audit history of fired alerts

    def _prune(self, dq, now):
        while dq and now - dq[0] > self.window_seconds:
            dq.popleft()

    def observe(self, entity_key, timestamp, score):
        """Feed one scored transaction in. Returns an alert dict if this
        transaction just pushed the entity over the trigger count, else None.

        entity_key: e.g. the card1 value (None/NaN entities are skipped --
                    can't detect a pattern with no shared key).
        timestamp:  wall-clock seconds (time.time() in the live API; for a
                    historical replay, use the transaction's own clock).
        score:      this transaction's Layer 1 fraud probability.
        """
        if entity_key is None:
            return None

        dq = self._windows[entity_key]
        self._prune(dq, timestamp)

        if score < self.risky_threshold:
            return None
        dq.append(timestamp)

        if len(dq) >= self.count_trigger:
            alert = {
                "entity_col": self.group_col,
                "entity_key": entity_key,
                "count": len(dq),
                "window_seconds": self.window_seconds,
                "triggered_at": timestamp,
            }
            self.alerts.append(alert)
            return alert
        return None

    def active_alert_for(self, entity_key):
        """Is this entity currently inside an open spike window?"""
        dq = self._windows.get(entity_key)
        if not dq:
            return False
        return len(dq) >= self.count_trigger

    def recent_alerts(self, n=20):
        return self.alerts[-n:][::-1]
