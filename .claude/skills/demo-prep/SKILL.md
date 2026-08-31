---
name: demo-prep
description: The judge demo sequence for FraudGuard and what each step must prove. Use on Day 6, when rehearsing the demo, or when deciding whether a feature is demo-critical.
---

# Demo prep

Seven beats, in order. Each maps to a judging criterion.

1. **Metrics first** — precision, recall, F1, AUC-PR, false-positive cost on the
   held-out set. Numbers upfront, nothing hidden.
2. **Live stream** — replay ~100 test transactions through the API, dashboard
   updating in real time.
3. **Explain a flag** — click a blocked transaction, show its SHAP top-3.
   "Flagged because amount was 4x this card's average, device was new, 2am."
4. **Trigger a spike** — inject ~20 fraud transactions sharing a `card1`,
   show the aggregate alert fire. This is the differentiator; do not cut it.
5. **Manual review queue** — show a 0.4–0.6 transaction sitting there and say
   why it was not auto-blocked. This is the graceful-failure criterion.
6. **Audit trail** — every decision logged, timestamped, with reason.
7. **False-positive cost, stated out loud** — in rupees, with the threshold
   choice defended.

## Rules

- A synthetic injected spike is fine — it demonstrates detection capability. Say
  it is injected. Never present it as a historical finding.
- If a number changed since the last run, re-run before demoing. No stale slides.
- If something breaks live, show the manual-review queue absorbing it. Graceful
  degradation is the point, not a perfect run.
