"""
Shared feature pipeline. Fit on TRAIN ONLY, then applied unchanged to the
held-out test set and later to live transactions in the Flask API.

This file exists so that training and serving cannot drift apart. If the API
built features differently from training, the model would silently score
garbage -- one of the most common and hardest-to-spot bugs in deployed ML.

Two rules encoded here:
  1. Raw TransactionDT is NEVER a feature. It is a timestamp that only
     increases, and the test set lives entirely in the future. A tree would
     happily split on it and learn nothing transferable. We derive cyclical
     time features (hour of day, day of week) instead.
  2. Category encodings are learned from train only. Categories that appear
     for the first time in test/production map to -1 ("unseen"), which is
     honest. Fitting the encoder on all data first would leak test information.
"""
import pickle

import numpy as np
import pandas as pd

from config import ARTIFACTS, ID_COL, TARGET, TIME_COL

FEATURE_META = ARTIFACTS / "feature_meta.pkl"

DAY = 60 * 60 * 24


def engineer(df):
    """Derived features. Pure function of a single row's own fields, so it is
    safe to call on one live transaction exactly as on the training frame."""
    out = df.copy()

    # Cyclical time -- fraud has a strong hour-of-day signature (card testing
    # bots run at night). Derived from DT but bounded, unlike DT itself.
    out["hour"] = (out[TIME_COL] / 3600) % 24
    out["dayofweek"] = (out[TIME_COL] / DAY) % 7

    # Amount is heavily right-skewed; log makes it usable by linear models.
    out["log_amt"] = np.log1p(out["TransactionAmt"])

    # The cents portion of an amount. Fraudulent card-testing charges cluster
    # on round or repeated decimals -- a genuinely predictive quirk here.
    out["amt_cents"] = ((out["TransactionAmt"] % 1) * 1000).round().astype("float32")

    # How much of the identity record is missing, as one number.
    id_cols = [c for c in out.columns if c.startswith("id_")]
    if id_cols:
        out["id_missing_frac"] = out[id_cols].isna().mean(axis=1).astype("float32")

    return out


def fit(train_df):
    """Learn the column list and category vocabularies from the training set."""
    df = engineer(train_df)
    drop = {ID_COL, TIME_COL, TARGET}

    cat_cols = [c for c in df.columns
                if c not in drop and df[c].dtype == object]
    num_cols = [c for c in df.columns
                if c not in drop and c not in cat_cols]

    # value -> integer code, learned on train only
    vocab = {}
    for c in cat_cols:
        cats = df[c].dropna().unique().tolist()
        vocab[c] = {v: i for i, v in enumerate(sorted(map(str, cats)))}

    meta = {"cat_cols": cat_cols, "num_cols": num_cols, "vocab": vocab,
            "feature_names": num_cols + cat_cols}
    with open(FEATURE_META, "wb") as f:
        pickle.dump(meta, f)
    print(f"Feature pipeline fitted: {len(num_cols)} numeric + "
          f"{len(cat_cols)} categorical = {len(meta['feature_names'])} features")
    return meta


def load_meta():
    with open(FEATURE_META, "rb") as f:
        return pickle.load(f)


def transform(df, meta=None):
    """Apply the fitted pipeline. Works on a full frame or a single row.

    Built as one dict-of-columns -> single DataFrame construction rather than
    ~440 sequential `X[c] = ...` inserts. The row-by-row version is not just
    slow -- it fragments the block manager badly enough that pandas warns on
    nearly every insert -- and this function is now also the hot path the
    live Flask API calls once per incoming transaction, where that overhead
    is no longer a one-off training cost.
    """
    meta = meta or load_meta()
    df = engineer(df)

    # Any column the model expects but this frame lacks (common for a live
    # transaction posted with partial fields) becomes NaN, not an error.
    missing = [c for c in meta["feature_names"] if c not in df.columns]
    if missing:
        df = pd.concat(
            [df, pd.DataFrame(np.nan, index=df.index, columns=missing)], axis=1
        )

    cols = {}
    for c in meta["num_cols"]:
        cols[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    for c in meta["cat_cols"]:
        # -1 = category never seen during training. NaN stays NaN.
        cols[c] = df[c].astype(object).map(
            lambda v: meta["vocab"][c].get(str(v), -1) if pd.notna(v) else np.nan
        ).astype("float32")

    X = pd.DataFrame(cols, index=df.index)
    return X[meta["feature_names"]]
