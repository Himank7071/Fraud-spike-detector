"""
Day 1, step 3: logistic regression baseline.

The point of a baseline is not to be good. It is to give every later number a
floor to beat. If XGBoost cannot clearly beat this, something is wrong with the
XGBoost setup -- and you will know, instead of being impressed by a number with
nothing to compare it to.

Logistic regression cannot handle NaN and is sensitive to feature scale, so it
needs median imputation + standardisation. Both are fitted on TRAIN ONLY;
fitting them on all data would leak the test distribution into training.
"""
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import features
import pandas as pd
from config import TARGET, TEST_PARQUET, TRAIN_PARQUET
from metrics import evaluate, report, threshold_sweep


def main():
    print("Loading parquet ...")
    train = pd.read_parquet(TRAIN_PARQUET)
    test = pd.read_parquet(TEST_PARQUET)

    meta = features.fit(train)          # fitted on train only
    X_train = features.transform(train, meta)
    X_test = features.transform(test, meta)
    y_train, y_test = train[TARGET].values, test[TARGET].values
    amounts_test = test["TransactionAmt"].values

    print(f"X_train {X_train.shape} | X_test {X_test.shape}")

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        # class_weight balanced -- without it the model just predicts 'legit'
        # for everything and looks 96.5% accurate while catching zero fraud.
        ("lr", LogisticRegression(max_iter=300, class_weight="balanced",
                                  n_jobs=-1)),
    ])

    print("\nFitting logistic regression (a few minutes on 413k x 440) ...")
    t0 = time.time()
    pipe.fit(X_train, y_train)
    print(f"  fitted in {time.time() - t0:.0f}s")

    y_prob = pipe.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 70)
    print("BASELINE RESULTS -- held-out test set (last 30% by time)")
    print("=" * 70)
    m = evaluate(y_test, y_prob, amounts=amounts_test)
    report("Logistic regression", m)
    threshold_sweep(y_test, y_prob, amounts=amounts_test)

    np.save("../artifacts/baseline_probs.npy", y_prob)
    print("\nBaseline AUC-PR to beat on Day 2: "
          f"{m['auc_pr']:.4f}")


if __name__ == "__main__":
    main()
