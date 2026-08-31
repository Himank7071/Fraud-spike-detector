"""
Day 2: the XGBoost model. This is Layer 1 of the architecture.

THE LEAKAGE TRAP THIS SCRIPT AVOIDS
-----------------------------------
XGBoost stops training when performance stops improving on a "validation set".
The obvious thing is to hand it the test set -- and it is wrong. The model would
then be tuned, indirectly, on the very data we report metrics from, and those
metrics become optimistic fiction.

So we split the TRAINING period again, chronologically:
    days   1.0 - 102.7   -> fit           (85% of train)
    days 102.7 - 120.8   -> validation    (15% of train, for early stopping)
    days 120.8 - 183.0   -> HELD-OUT TEST (untouched until the final number)

Same rule as before: later data never trains a model that is scored on earlier
data.
"""
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import features
from config import (ARTIFACTS, SEED, TARGET, TIME_COL, TEST_PARQUET,
                    TRAIN_PARQUET)
from metrics import evaluate, report, threshold_sweep

MODEL_PATH = ARTIFACTS / "xgb_model.json"
VAL_FRACTION = 0.15
DAY = 60 * 60 * 24


def main():
    print("Loading parquet ...")
    train_full = pd.read_parquet(TRAIN_PARQUET)
    test = pd.read_parquet(TEST_PARQUET)

    # --- inner chronological split for early stopping -------------------
    cut = int(len(train_full) * (1 - VAL_FRACTION))
    fit_df = train_full.iloc[:cut]
    val_df = train_full.iloc[cut:]
    assert fit_df[TIME_COL].max() < val_df[TIME_COL].min(), "TIME LEAK"
    print(f"  fit  {len(fit_df):,} rows | days "
          f"{fit_df[TIME_COL].min()/DAY:.1f} -> {fit_df[TIME_COL].max()/DAY:.1f}")
    print(f"  val  {len(val_df):,} rows | days "
          f"{val_df[TIME_COL].min()/DAY:.1f} -> {val_df[TIME_COL].max()/DAY:.1f}")
    print(f"  test {len(test):,} rows | days "
          f"{test[TIME_COL].min()/DAY:.1f} -> {test[TIME_COL].max()/DAY:.1f}")

    # Feature pipeline fitted on the FIT portion only.
    meta = features.fit(fit_df)
    X_fit = features.transform(fit_df, meta)
    X_val = features.transform(val_df, meta)
    X_test = features.transform(test, meta)
    y_fit = fit_df[TARGET].values
    y_val = val_df[TARGET].values
    y_test = test[TARGET].values
    amounts_test = test["TransactionAmt"].values

    # --- class imbalance -------------------------------------------------
    # scale_pos_weight tells XGBoost to treat one fraud example as worth ~27
    # legit ones when computing gradients. Without it the model minimises loss
    # by predicting "legit" for everything.
    spw = (y_fit == 0).sum() / (y_fit == 1).sum()
    print(f"\nscale_pos_weight = {spw:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=1000,          # upper bound; early stopping picks the real number
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,              # each tree sees 80% of rows  ) both reduce
        colsample_bytree=0.8,       # each tree sees 80% of cols  ) overfitting
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        eval_metric="aucpr",        # optimise precision-recall, not accuracy
        early_stopping_rounds=50,
        tree_method="hist",         # fast histogram algorithm
        n_jobs=-1,
        random_state=SEED,
    )

    print("Training ...")
    t0 = time.time()
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=100)
    print(f"  trained in {time.time() - t0:.0f}s")
    print(f"  early stopping chose {model.best_iteration + 1} trees "
          f"(val AUC-PR {model.best_score:.4f})")

    # --- the only time we touch the test set -----------------------------
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 70)
    print("XGBOOST RESULTS -- held-out test set (last 30% by time, never seen)")
    print("=" * 70)
    m = evaluate(y_test, y_prob, amounts=amounts_test)
    report("XGBoost", m)
    sweep = threshold_sweep(y_test, y_prob, amounts=amounts_test)

    baseline = np.load(ARTIFACTS / "baseline_probs.npy")
    b = evaluate(y_test, baseline, amounts=amounts_test)
    print(f"\nBaseline AUC-PR {b['auc_pr']:.4f}  ->  "
          f"XGBoost AUC-PR {m['auc_pr']:.4f}   "
          f"({m['auc_pr'] / b['auc_pr']:.1f}x better)")
    print(f"At threshold 0.70, false positives "
          f"{b['fp']:,} -> {m['fp']:,}  "
          f"(INR {b['false_positive_cost_inr']:,.0f} -> "
          f"INR {m['false_positive_cost_inr']:,.0f} blocked)")

    model.save_model(MODEL_PATH)
    np.save(ARTIFACTS / "xgb_test_probs.npy", y_prob)
    with open(ARTIFACTS / "xgb_metrics.json", "w") as f:
        json.dump({"headline": m, "sweep": sweep,
                   "baseline_auc_pr": b["auc_pr"],
                   "best_iteration": int(model.best_iteration + 1)}, f, indent=2)
    print(f"\nSaved model -> {MODEL_PATH.name}")

    # --- what the model actually keyed on --------------------------------
    imp = (pd.Series(model.feature_importances_, index=meta["feature_names"])
           .sort_values(ascending=False).head(15))
    print("\nTop 15 features by gain:")
    print(imp.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
