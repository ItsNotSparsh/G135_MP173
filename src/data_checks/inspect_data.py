from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")
TARGET = "spoilage_pct"
SENSOR_COLS = [
    "storage_days",
    "temp_set_c",
    "humidity_pct",
    "power_cuts_hours",
    "bag_weight_kg",
    "door_openings_per_day",
]

train = pd.read_csv(DATA_DIR / "train.csv")
later = pd.read_csv(DATA_DIR / "later.csv")
holdout = pd.read_csv(DATA_DIR / "holdout_features.csv")

print("Train shape:", train.shape)
print("Later shape:", later.shape)
print("Holdout shape:", holdout.shape)
print("Train columns:", train.columns.tolist())
print("Holdout columns:", holdout.columns.tolist())

print("\nMissing values:")
print(train.isna().sum())

print("\nExact duplicate rows:", train.duplicated().sum())

print("\nNumeric ranges and averages by chamber:")
numeric_cols = train.select_dtypes(include="number").columns
for col in numeric_cols:
    print(f"\n{col}")
    print(train.groupby("chamber")[col].agg(["min", "mean", "max"]).to_string())

print("\nPossible sentinel values:")
sentinels = {-999, -1, 999, 9999}
for col in numeric_cols:
    matches = train.loc[train[col].isin(sentinels), col]
    if not matches.empty:
        print(f"\n{col}")
        print(matches.value_counts().to_string())

print("\nTarget distribution:")
print(train[TARGET].describe().to_string())

print("\nSmallest target values:")
print(train[TARGET].sort_values().head(20).to_list())

print("\nLargest target values:")
print(train[TARGET].sort_values(ascending=False).head(20).to_list())

print("\nText categories in train and later:")
for col in ["crop", "chamber"]:
    print(f"\n{col} — train:")
    print(sorted(train[col].dropna().unique().tolist()))
    print(f"{col} — later:")
    print(sorted(later[col].dropna().unique().tolist()))

print("\nCorrelation with target:")
print(
    train.select_dtypes(include="number")
    .corr()[TARGET]
    .sort_values(ascending=False)
    .to_string()
)

print("\nDifference between unloading spoilage and target:")
diff = (train["spoilage_found_at_unloading"] - train[TARGET]).abs()
print("Exact matches:", (diff == 0).sum())
print("Maximum absolute difference:", diff.max())
print("Mean absolute difference:", diff.mean())

print("\nRepeated record IDs:")
repeated_ids = train.loc[
    train["record_id"].duplicated(keep=False), "record_id"
]
print(repeated_ids.value_counts().sort_index().to_string())

print("\nChamber 1 power-cut values divided by 60:")
chamber_1_power = train.loc[
    train["chamber"] == "Chamber 1", "power_cuts_hours"
]
print("Mean after dividing by 60:", (chamber_1_power / 60).mean())
print("Max after dividing by 60:", (chamber_1_power / 60).max())

print("\nLongest unchanged sensor runs by chamber:")

dated = train.copy()
dated["date"] = pd.to_datetime(dated["date"])

for chamber in sorted(dated["chamber"].unique()):
    chamber_data = dated[dated["chamber"] == chamber]
    print(f"\n{chamber}")

    for col in SENSOR_COLS:
        # All readings on a date must match for that date to count.
        daily = (
            chamber_data.groupby("date")[col]
            .agg(
                first="first",
                last="last",
                unique="nunique",
                rows="size",
            )
            .sort_index()
        )
        daily["same_reading"] = (
            daily["unique"].eq(1)
            & daily["first"].eq(daily["last"])
        )

        # A date continues a run only if it is the next calendar day
        # and every reading on both dates has the same value.
        previous_date = daily.index.to_series().shift()
        consecutive_day = (
            daily.index.to_series() - previous_date
        ).dt.days.eq(1)

        previous_value = daily["first"].shift()
        same_as_previous = daily["first"].eq(previous_value)

        continues = (
            daily["same_reading"]
            & daily["same_reading"].shift(fill_value=False)
            & consecutive_day
            & same_as_previous
        )
        daily["run_id"] = (~continues).cumsum()

        runs = (
            daily[daily["same_reading"]]
            .groupby("run_id")
            .agg(
                value=("first", "first"),
                start_date=("first", lambda values: values.index.min()),
                end_date=("first", lambda values: values.index.max()),
                days=("first", "size"),
            )
            .sort_values("days", ascending=False)
            .head(5)
        )

        print(f"\n{col}")
        if runs.empty:
            print("No repeated daily readings.")
        else:
            print(runs.to_string(index=False))