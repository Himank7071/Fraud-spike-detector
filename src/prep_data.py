"""
Day 1, step 1: load the raw CSVs, join identity, split by time, save parquet.

Why parquet: re-reading a 652 MB CSV takes ~60s every time. Parquet loads the
same data in a few seconds and preserves dtypes, so every later script is fast.

CRITICAL: the split is chronological, never shuffled. TransactionDT is a seconds
offset from a fixed reference point, so sorting by it gives real time order.
Shuffling would let the model train on future transactions while being tested on
past ones -- that is data leakage and it inflates your metrics into a lie.
"""
import pandas as pd

from config import (ID_COL, TARGET, TIME_COL, TRAIN_FRACTION, TRAIN_IDENTITY,
                    TRAIN_PARQUET, TRAIN_TRANSACTION, TEST_PARQUET)


def build_dtypes(path, target_included):
    """Sample the file to pick memory-efficient dtypes before the full read.

    pandas defaults every number to 64-bit. For 394 columns x 590k rows that is
    ~1.9 GB. float32 halves it with no meaningful precision loss for this data.
    """
    sample = pd.read_csv(path, nrows=10_000)
    dtypes = {}
    for col in sample.columns:
        if col in (ID_COL, TIME_COL):
            dtypes[col] = "int64"          # keep full precision on id/time
        elif target_included and col == TARGET:
            dtypes[col] = "int8"
        elif sample[col].dtype == object:
            dtypes[col] = "object"         # categoricals handled later
        else:
            dtypes[col] = "float32"
    return dtypes


def load_joined():
    print("Reading train_transaction.csv ...")
    tx = pd.read_csv(TRAIN_TRANSACTION,
                     dtype=build_dtypes(TRAIN_TRANSACTION, True))
    print(f"  transactions: {tx.shape[0]:,} rows x {tx.shape[1]} cols")

    print("Reading train_identity.csv ...")
    idf = pd.read_csv(TRAIN_IDENTITY,
                      dtype=build_dtypes(TRAIN_IDENTITY, False))
    print(f"  identity:     {idf.shape[0]:,} rows x {idf.shape[1]} cols")

    # LEFT join: most transactions have no identity record. Those become NaN,
    # which is correct -- "we have no device info" is itself a signal, and
    # XGBoost handles NaN natively. Do not fill these with zeros.
    df = tx.merge(idf, on=ID_COL, how="left")
    coverage = df["id_01"].notna().mean()
    print(f"  joined:       {df.shape[0]:,} rows x {df.shape[1]} cols "
          f"({coverage:.1%} have identity data)")
    return df


def time_split(df):
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    cut = int(len(df) * TRAIN_FRACTION)
    cut_time = df.loc[cut, TIME_COL]

    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()

    # Sanity check: no train transaction may occur at or after the cut.
    assert train[TIME_COL].max() < test[TIME_COL].min(), "TIME LEAK in split!"

    day = 60 * 60 * 24
    print(f"\nChronological split at TransactionDT = {cut_time:,} "
          f"(day {cut_time / day:.1f} of the dataset)")
    print(f"  train: {len(train):,} rows | days "
          f"{train[TIME_COL].min()/day:.1f} -> {train[TIME_COL].max()/day:.1f} "
          f"| fraud rate {train[TARGET].mean():.3%}")
    print(f"  test:  {len(test):,} rows | days "
          f"{test[TIME_COL].min()/day:.1f} -> {test[TIME_COL].max()/day:.1f} "
          f"| fraud rate {test[TARGET].mean():.3%}")
    return train, test


def main():
    df = load_joined()
    train, test = time_split(df)

    print("\nWriting parquet ...")
    train.to_parquet(TRAIN_PARQUET, index=False)
    test.to_parquet(TEST_PARQUET, index=False)
    print(f"  {TRAIN_PARQUET.name}  "
          f"({TRAIN_PARQUET.stat().st_size / 1e6:.0f} MB)")
    print(f"  {TEST_PARQUET.name}   "
          f"({TEST_PARQUET.stat().st_size / 1e6:.0f} MB)")
    print("\nDone. Held-out test set is now frozen -- do not touch it until "
          "final evaluation.")


if __name__ == "__main__":
    main()
