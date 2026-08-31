"""
Day 1, step 2: EDA. Every number printed here is one you may need to defend
to a judge, so this script produces facts, not pretty pictures.
"""
import pandas as pd

from config import ARTIFACTS, TARGET, TIME_COL, TEST_PARQUET, TRAIN_PARQUET

pd.set_option("display.width", 120)
DAY = 60 * 60 * 24


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    train = pd.read_parquet(TRAIN_PARQUET)
    test = pd.read_parquet(TEST_PARQUET)

    section("1. CLASS IMBALANCE  (the central difficulty of this problem)")
    n_pos = int(train[TARGET].sum())
    n_neg = len(train) - n_pos
    print(f"train: {n_neg:,} legit / {n_pos:,} fraud  ({train[TARGET].mean():.3%} fraud)")
    print(f"test:  {len(test) - int(test[TARGET].sum()):,} legit / "
          f"{int(test[TARGET].sum()):,} fraud  ({test[TARGET].mean():.3%} fraud)")
    print(f"\nscale_pos_weight = neg/pos = {n_neg / n_pos:.2f}")
    print("A model that predicts 'never fraud' scores "
          f"{1 - train[TARGET].mean():.1%} accuracy. This is why accuracy is a")
    print("useless metric here and we report precision/recall/AUC-PR instead.")

    section("2. MISSINGNESS")
    miss = train.isna().mean().sort_values(ascending=False)
    print(f"columns >90% missing: {(miss > 0.9).sum()} of {len(miss)}")
    print(f"columns >50% missing: {(miss > 0.5).sum()} of {len(miss)}")
    print(f"columns with no missing values: {(miss == 0).sum()}")
    print("\nworst 8:")
    print(miss.head(8).to_string(float_format=lambda v: f"{v:.1%}"))
    print("\nWe do NOT drop or impute these. XGBoost routes NaN down its own")
    print("branch, and 'field absent' is itself predictive of fraud here.")

    section("3. FRAUD RATE BY CATEGORY  (where the signal lives)")
    for col in ["ProductCD", "card4", "card6", "DeviceType", "P_emaildomain"]:
        if col not in train.columns:
            continue
        g = (train.groupby(col, dropna=False)[TARGET]
             .agg(n="size", fraud_rate="mean")
             .sort_values("fraud_rate", ascending=False))
        g = g[g["n"] >= 500].head(6)
        print(f"\n{col}:")
        print(g.to_string(formatters={"fraud_rate": lambda v: f"{v:.2%}",
                                      "n": lambda v: f"{v:,}"}))

    section("4. TRANSACTION AMOUNT")
    print(train.groupby(TARGET)["TransactionAmt"]
          .describe()[["count", "mean", "50%", "75%", "max"]]
          .to_string(float_format=lambda v: f"{v:,.2f}"))

    section("5. FRAUD RATE OVER TIME  (does the pattern drift?)")
    tmp = train[[TIME_COL, TARGET]].copy()
    tmp["week"] = (tmp[TIME_COL] / (DAY * 7)).astype(int)
    weekly = tmp.groupby("week")[TARGET].agg(n="size", rate="mean")
    print(weekly.to_string(formatters={"rate": lambda v: f"{v:.2%}",
                                       "n": lambda v: f"{v:,}"}))
    print("\nFraud rate is not constant. This is why a time-based split is the")
    print("honest one: the model is tested on a period it has never seen,")
    print("exactly as it would be in production.")

    section("6. HOURLY PATTERN")
    tmp = train[[TIME_COL, TARGET]].copy()
    tmp["hour"] = ((tmp[TIME_COL] / 3600) % 24).astype(int)
    hourly = tmp.groupby("hour")[TARGET].agg(n="size", rate="mean")
    peak = hourly["rate"].idxmax()
    low = hourly["rate"].idxmin()
    print(f"peak fraud hour: {peak}:00 ({hourly.loc[peak, 'rate']:.2%})")
    print(f"safest hour:     {low}:00 ({hourly.loc[low, 'rate']:.2%})")
    print(f"ratio: {hourly.loc[peak, 'rate'] / hourly.loc[low, 'rate']:.1f}x")
    print("\nStrong hour-of-day signal -> justifies the derived 'hour' feature.")


if __name__ == "__main__":
    main()
