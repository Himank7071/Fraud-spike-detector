"""
Day 6 demo script: streams the Kaggle competition test set through the live
Flask API, then injects a synthetic card-testing spike so Layer 2 fires live.

Hard rule respected: test_transaction.csv / test_identity.csv have no labels,
so this is a replay stream for the demo ONLY -- nothing here is or ever
becomes a reported metric. Metrics all come from artifacts/xgb_metrics.json,
computed once in train_xgb.py against the chronological held-out split.

Usage (with the API already running in another terminal):
    cd src && ../.venv/bin/python replay_demo.py --reset
"""
import argparse
import time

import numpy as np
import pandas as pd
import requests

from config import (DEMO_IDENTITY, DEMO_TRANSACTION, ID_COL, TARGET,
                    TIME_COL, TRAIN_PARQUET)

API = "http://127.0.0.1:5000"
SAMPLE_ROWS_TO_READ = 50_000   # head of the file is plenty for a demo stream
N_NORMAL = 70
N_SYNTHETIC = 20
N_TEMPLATE_CANDIDATES = 25


def load_demo_frame():
    print(f"Reading first {SAMPLE_ROWS_TO_READ:,} rows of the demo replay "
          "stream (Kaggle's UNLABELED test set -- never used for metrics) ...")
    tx = pd.read_csv(DEMO_TRANSACTION, nrows=SAMPLE_ROWS_TO_READ)
    idf = pd.read_csv(DEMO_IDENTITY)
    # Kaggle quirk: test_identity.csv ships "id-01".."id-38" (hyphens) while
    # train_identity.csv -- what the model was fit on -- used "id_01" etc.
    # (underscores). Without this rename every identity column here silently
    # fails to match a trained feature name and gets treated as 100% missing.
    idf = idf.rename(columns={c: c.replace("-", "_") for c in idf.columns})

    df = tx.merge(idf, on=ID_COL, how="left")
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    print(f"  {len(df):,} rows loaded, "
          f"{df['id_01'].notna().mean():.1%} have identity data")
    return df


def json_safe(row):
    """A pandas row mixes numpy scalar dtypes that plain json.dumps (what
    requests uses) cannot serialise. Cast everything to native Python."""
    out = {}
    for k, v in row.items():
        if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
            out[k] = None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def row_to_payload(row, synthetic=False):
    d = json_safe(row)
    d["_synthetic"] = synthetic
    return d


def pick_fraud_template(n_candidates=N_TEMPLATE_CANDIDATES):
    """The anonymised Vesta (V*) and count (C*) columns dominate this model's
    top features by gain (see artifacts/train_log.txt) -- they cannot be
    hand-crafted into a convincing fraud pattern by guessing values. Instead,
    borrow the full feature vector of a REAL confirmed fraud row from
    TRAINING data (never the held-out test set -- that stays untouched) and
    reuse it as the base for the synthetic burst, swapping only the
    identifying fields (card1, id, amount, time). The pattern is real; only
    the delivery is synthetic and clearly tagged as such.

    We score a handful of candidates through the live API itself (the only
    place scoring happens, per features.py's single-pipeline rule) and keep
    the one the model is most confident about, so the injected spike reliably
    clears the spike detector's risky-transaction bar.
    """
    print("Selecting a high-confidence fraud template for the synthetic "
          "spike (from TRAINING data, never the held-out test set) ...")
    train = pd.read_parquet(TRAIN_PARQUET)
    fraud = train[train[TARGET] == 1]
    richness = fraud.notna().sum(axis=1).sort_values(ascending=False)
    candidates = fraud.loc[richness.index[:n_candidates]]

    best_score, best_row = -1.0, None
    for _, row in candidates.iterrows():
        result = post(row_to_payload(row, synthetic=True))
        if result["score"] > best_score:
            best_score, best_row = result["score"], row
    print(f"  template selected: score {best_score:.3f}, "
          f"card1={best_row['card1']}\n")
    return best_row


def make_synthetic_burst(base_row, n, shared_card1):
    """~20 small, escalating charges sharing one card1, seconds apart -- the
    classic card-testing probe. Individually each may look unremarkable to
    Layer 1; the pattern is what Layer 2 exists to catch. Always tagged
    synthetic and never presented as a real historical finding (demo-prep
    rule)."""
    rows = []
    base = json_safe(base_row)
    for i in range(n):
        r = dict(base)
        r["card1"] = shared_card1
        r[ID_COL] = f"SYNTH-{shared_card1}-{i}"
        r["TransactionAmt"] = float(np.random.choice([1.0, 5.0, 9.99, 49.0]))
        r[TIME_COL] = base[TIME_COL] + i * 2
        r["DeviceType"] = "mobile"
        r["DeviceInfo"] = None
        rows.append(r)
    return rows


def post(payload):
    r = requests.post(f"{API}/api/score", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true",
                    help="wipe the audit trail + spike state before streaming")
    ap.add_argument("--n-normal", type=int, default=N_NORMAL)
    ap.add_argument("--n-synthetic", type=int, default=N_SYNTHETIC)
    ap.add_argument("--delay", type=float, default=0.25)
    args = ap.parse_args()

    template = pick_fraud_template()

    if args.reset:
        requests.post(f"{API}/api/reset", timeout=10)
        print("Audit trail reset -- template probes above won't show on the "
              "dashboard.\n")

    df = load_demo_frame()
    sample = (df.sample(n=min(args.n_normal, len(df)), random_state=42)
                .sort_values(TIME_COL))

    print(f"\nStreaming {len(sample)} normal transactions to {API} ...")
    counts = {"BLOCK": 0, "MANUAL_REVIEW": 0, "ALLOW": 0}
    injected_at = len(sample) // 2

    for i, (_, row) in enumerate(sample.iterrows()):
        result = post(row_to_payload(row))
        counts[result["decision"]] += 1
        print(f"  [{i + 1:>3}/{len(sample)}] {result['transaction_id']:<10} "
              f"score={result['score']:.3f}  {result['decision']}")
        time.sleep(args.delay)

        if i == injected_at:
            shared_card1 = 900000 + int(template.name) % 100000  # a card1
            # value that won't collide with any real one in this sample
            print(f"\n  >>> INJECTING synthetic card-testing spike on "
                  f"card1={shared_card1} ({args.n_synthetic} txns, cloned "
                  "from a real confirmed-fraud pattern, clearly tagged "
                  "synthetic) -- watch the dashboard alert fire\n")
            burst = make_synthetic_burst(template, args.n_synthetic,
                                         shared_card1)
            for j, r in enumerate(burst):
                result = post({**r, "_synthetic": True})
                tag = " <-- SPIKE FIRED" if result.get("spike_fired") else ""
                print(f"    [synthetic {j + 1:>2}/{args.n_synthetic}] "
                      f"{result['transaction_id']:<20} "
                      f"score={result['score']:.3f} {result['decision']}{tag}")
                time.sleep(0.15)
            print()

    print("Done. Decision mix over the normal stream:", counts)
    print("Open http://127.0.0.1:5000/ to see the dashboard.")


if __name__ == "__main__":
    main()
