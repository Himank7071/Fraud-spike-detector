"""
Column-by-column profile of the dataset.

Vesta masked what every column MEANS, but they could not mask how each column
BEHAVES. This script recovers structure empirically: which columns are missing
together, how many distinct values each holds, and how predictive each one is
on its own. That is enough to work with them intelligently.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import TARGET, TIME_COL, TRAIN_PARQUET

pd.set_option("display.width", 150)


def family(col):
    if col.startswith("V"): return "V (Vesta engineered)"
    if col.startswith("C"): return "C (counting)"
    if col.startswith("D"): return "D (timedelta)"
    if col.startswith("M"): return "M (match flags)"
    if col.startswith("id_"): return "id_ (network/device)"
    if col.startswith("card"): return "card (payment card)"
    if col.startswith("addr"): return "addr (address)"
    if col.startswith("dist"): return "dist (distance)"
    if "emaildomain" in col: return "email"
    return "other"


def univariate_auc(s, y):
    """How well does this ONE column separate fraud, by itself?

    0.50 = useless. Below 0.50 means it separates in the inverse direction,
    which is still signal. We report distance from 0.50.
    """
    m = s.notna()
    if m.sum() < 1000 or s[m].nunique() < 2:
        return np.nan
    try:
        return roc_auc_score(y[m], s[m])
    except Exception:
        return np.nan


def main():
    df = pd.read_parquet(TRAIN_PARQUET)
    y = df[TARGET]
    cols = [c for c in df.columns if c not in (TARGET, TIME_COL, "TransactionID")]

    print("=" * 78)
    print("1. COLUMN FAMILIES  (grouped by naming prefix)")
    print("=" * 78)
    fam = pd.Series({c: family(c) for c in cols})
    summary = []
    for name, group in fam.groupby(fam):
        gc = group.index.tolist()
        summary.append({
            "family": name,
            "n_cols": len(gc),
            "avg_missing": df[gc].isna().mean().mean(),
            "dtype": "object" if df[gc[0]].dtype == object else "numeric",
        })
    print(pd.DataFrame(summary).sort_values("n_cols", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.1%}"))

    print("\nOfficial meanings (all Vesta ever published):")
    print("  C1-C14   counting -- 'how many addresses are associated with this")
    print("           payment card', etc. Actual meaning masked.")
    print("  D1-D15   timedelta -- 'days between previous transaction', etc.")
    print("  M1-M9    match -- 'names on card and address match', etc.")
    print("  Vxxx     Vesta engineered: ranking, counting, entity relations.")
    print("  id_*     network connection (IP, ISP, proxy) + digital signature")
    print("           (browser/OS/version). Pairwise dictionary NOT provided.")

    print("\n" + "=" * 78)
    print("2. THE V-COLUMN BLOCKS  (recovering hidden structure)")
    print("=" * 78)
    print("339 V columns is unmanageable -- until you notice they are missing in")
    print("GROUPS. Columns computed from the same underlying entity go missing")
    print("together. Identical missing-count => almost certainly one block.\n")
    vcols = [c for c in cols if c.startswith("V")]
    nan_counts = df[vcols].isna().sum()
    blocks = nan_counts.groupby(nan_counts).apply(lambda s: list(s.index))
    print(f"{len(vcols)} V columns collapse into {len(blocks)} blocks by NaN pattern:\n")
    for i, (n_missing, members) in enumerate(blocks.items(), 1):
        print(f"  block {i:2d}: {len(members):3d} cols | "
              f"{n_missing / len(df):5.1%} missing | "
              f"{members[0]}...{members[-1]}")
        if i >= 15:
            print(f"  ... and {len(blocks) - 15} more blocks")
            break
    print("\nThis is how the winners cut 339 columns down: keep a few per block")
    print("(or PCA each block), since members are largely redundant.")

    print("\n" + "=" * 78)
    print("3. WHICH COLUMNS ACTUALLY PREDICT FRAUD ALONE")
    print("=" * 78)
    aucs = {}
    for c in cols:
        s = df[c]
        if s.dtype == object:
            s = s.astype("category").cat.codes.replace(-1, np.nan)
        aucs[c] = univariate_auc(s, y)
    auc_s = pd.Series(aucs).dropna()
    strength = (auc_s - 0.5).abs().sort_values(ascending=False)

    print("Top 20 single columns (AUC distance from 0.50 = coin flip):\n")
    top = pd.DataFrame({
        "auc": auc_s[strength.index[:20]],
        "strength": strength[:20],
        "missing": df[strength.index[:20]].isna().mean(),
        "n_unique": [df[c].nunique() for c in strength.index[:20]],
        "family": [family(c) for c in strength.index[:20]],
    })
    print(top.to_string(float_format=lambda v: f"{v:.3f}"))

    print(f"\ncolumns that are basically noise alone (|AUC-0.5| < 0.02): "
          f"{(strength < 0.02).sum()} of {len(strength)}")
    print("Weak alone does not mean useless -- trees combine weak features.")

    print("\n" + "=" * 78)
    print("4. THE MODEL'S TOP FEATURES, CHARACTERISED")
    print("=" * 78)
    print("Our XGBoost leaned hardest on these. We cannot know their meaning,")
    print("but we can describe their behaviour:\n")
    for c in ["V258", "V257", "V70", "V294", "V201", "C4", "C8", "id_12", "C14"]:
        if c not in df.columns:
            continue
        s = df[c]
        fr_hi = y[s > s.median()].mean() if s.dtype != object else np.nan
        fr_lo = y[s <= s.median()].mean() if s.dtype != object else np.nan
        print(f"  {c:6s} {family(c):22s} missing {s.isna().mean():5.1%} | "
              f"{s.nunique():5d} distinct | alone AUC {aucs.get(c, np.nan):.3f}")
        if not np.isnan(fr_hi):
            print(f"         fraud rate: {fr_lo:.2%} when below median, "
                  f"{fr_hi:.2%} when above")

    print("\n" + "=" * 78)
    print("5. THE D COLUMNS ARE CLOCKS, NOT NUMBERS")
    print("=" * 78)
    print("D1-D15 are timedeltas measured backwards from each transaction. They")
    print("drift upward with TransactionDT, so raw values are time-unstable --")
    print("a model can learn 'D1 > 300 means late in dataset' which will not")
    print("generalise. Correlation of each D column with time:\n")
    dcols = [c for c in cols if c.startswith("D") and df[c].dtype != object]
    corr = df[dcols + [TIME_COL]].corr()[TIME_COL].drop(TIME_COL).sort_values(
        ascending=False)
    print(corr.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("\nHigh positive correlation => that column is partly just a clock.")
    print("The standard fix (used by the winners) is D_normalised = ")
    print("day_of_transaction - D_column, which converts 'days ago' into a")
    print("stable reference date that does not drift.")


if __name__ == "__main__":
    main()
