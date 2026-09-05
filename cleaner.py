"""
Data cleaning for the Gas Sensor Array Drift datathon.

train.csv: 10,310 labelled measurements, batches 1-9, columns:
    measurement_id, batch, gas_class, concentration, feat_1..feat_128
test.csv: 2,288 unlabelled measurements (hidden batch 10), columns:
    measurement_id, batch, concentration, feat_1..feat_128

The raw files have no missing values and no duplicate ids, so cleaning here
focuses on: enforcing dtypes, guarding against future data issues (missing/
inf/duplicate/out-of-range values), and flagging outliers for the modelling
stage rather than dropping them (every row is scarce and batch 10 in test
is the whole point of the competition, so we never touch rows by "looks
weird").
"""

import numpy as np
import pandas as pd

# feat_1 .. feat_128: 8 response descriptors x 16 sensors, per the data dictionary.
FEATURE_COLS = [f"feat_{i}" for i in range(1, 129)]
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"


def load(path: str, is_train: bool) -> pd.DataFrame:
    """Read a csv and make sure it actually has the columns we expect
    before we do anything else with it."""
    df = pd.read_csv(path)

    # test.csv has no gas_class column (that's the hidden label we're
    # trying to predict), so only require it for train.csv.
    expected = ["measurement_id", "batch", "concentration", *FEATURE_COLS]
    if is_train:
        expected.insert(2, "gas_class")
    missing_cols = set(expected) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing_cols)}")

    return df


def clean(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """Fix dtypes and verify the data is well-formed. This does NOT drop
    or impute rows -- if something looks wrong it raises so a human can
    look at it, rather than silently "fixing" data in a competition."""
    df = df.copy()

    # --- schema / dtypes ---
    # measurement_id is just a label, batch is a grouping key (not a number
    # to do arithmetic on), gas_class is the integer target class (1-6).
    df["measurement_id"] = df["measurement_id"].astype(str)
    df["batch"] = df["batch"].astype("category")
    if is_train:
        df["gas_class"] = df["gas_class"].astype(int)

    # Force every feature/concentration column to numeric. Anything that
    # can't be parsed (e.g. stray text) becomes NaN here, which the
    # missing-value check right below will catch.
    numeric_cols = ["concentration", *FEATURE_COLS]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # --- integrity checks (raise loudly rather than silently dropping rows) ---

    # Same measurement appearing twice would double-count it during training/scoring.
    dup_ids = df["measurement_id"].duplicated().sum()
    if dup_ids:
        raise ValueError(f"{dup_ids} duplicate measurement_id values found")

    # NaNs here mean either a genuinely missing value or a value that
    # failed the pd.to_numeric conversion above.
    n_missing = df[numeric_cols].isna().sum().sum()
    if n_missing:
        raise ValueError(
            f"{n_missing} missing/non-numeric values found in numeric columns; "
            "inspect before proceeding, do not silently impute"
        )

    # inf/-inf values would silently break distance- or scale-based models
    # (and mean, std, etc.) further down the pipeline.
    n_inf = np.isinf(df[numeric_cols].to_numpy()).sum()
    if n_inf:
        raise ValueError(f"{n_inf} infinite values found in numeric columns")

    # Sanity-check the label and a physically-meaningful feature are in
    # their expected ranges (per the competition's data dictionary).
    if is_train and (df["gas_class"] < 1).any() or is_train and (df["gas_class"] > 6).any():
        raise ValueError("gas_class values outside expected range 1-6")

    if (df["concentration"] <= 0).any():
        raise ValueError("non-positive concentration values found")

    # --- constant / zero-variance features (useless for modelling, safe to flag) ---
    # A column with only one distinct value carries no information for any
    # model, so it's worth knowing about (though we don't drop it here --
    # train and test could theoretically differ, and dropping is a
    # modelling-stage decision, not a cleaning one).
    constant_cols = [c for c in FEATURE_COLS if df[c].nunique() <= 1]
    if constant_cols:
        print(f"warning: constant feature columns (no signal): {constant_cols}")

    return df


def report_outliers(df: pd.DataFrame, z_thresh: float = 6.0) -> pd.DataFrame:
    """Flag rows with any feature beyond z_thresh std devs from its column mean.

    Returned as a report only -- outliers are not dropped. With only ~10k rows
    across 6 classes and a drift-focused test set, aggressive trimming risks
    removing exactly the drift signal the competition is testing for.
    """
    # z-score each feature column independently (value - mean) / std, then
    # flag any row where at least one feature is more than z_thresh
    # std devs from that column's mean.
    z = (df[FEATURE_COLS] - df[FEATURE_COLS].mean()) / df[FEATURE_COLS].std()
    flagged = (z.abs() > z_thresh).any(axis=1)
    return df.loc[flagged, ["measurement_id", "batch"]]


def main():
    # Run both files through the same load -> validate -> clean pipeline.
    train = clean(load(TRAIN_PATH, is_train=True), is_train=True)
    test = clean(load(TEST_PATH, is_train=False), is_train=False)

    # Quick eyeball summary: shapes, class balance, and which batches
    # ended up where (train should be batches 1-9, test should be batch 10
    # only -- that's the drift split the competition is built around).
    print(f"train: {train.shape}, test: {test.shape}")
    print("train gas_class distribution:\n", train["gas_class"].value_counts().sort_index())
    print("train batches:", sorted(train["batch"].cat.categories.astype(int)))
    print("test batches:", sorted(test["batch"].astype(int).unique()))

    outliers = report_outliers(train)
    print(f"{len(outliers)} training rows flagged as outliers (>6 std on some feature), kept in place")

    # Write out the typed/validated data for the modelling stage to consume,
    # so modelling code doesn't need to repeat these checks.
    train.to_csv("train_clean.csv", index=False)
    test.to_csv("test_clean.csv", index=False)
    print("wrote train_clean.csv, test_clean.csv")


if __name__ == "__main__":
    main()
