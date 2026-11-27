from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.data_checks.clean_data import clean_data
from src.train import FEATURES, TARGET, build_model


def main():
    raw_train = pd.read_csv(Path("data/raw/train.csv"))
    cleaned_train = clean_data(raw_train)

    later = pd.read_csv(Path("data/raw/later.csv"))
    later = later.sort_values(["date", "record_id"]).reset_index(drop=True)

    split_at = int(len(later) * 0.70)
    later_for_training = later.iloc[:split_at]
    later_for_check = later.iloc[split_at:]

    v1 = joblib.load(Path("models/model_v1.joblib"))
    v1_predictions = v1.predict(later_for_check[FEATURES])
    v1_mae = mean_absolute_error(later_for_check[TARGET], v1_predictions)

    combined_training = pd.concat(
        [cleaned_train, later_for_training],
        ignore_index=True,
    )

    best_depth = v1.get_params()["regressor__max_depth"]
    v2 = build_model(max_depth=best_depth)
    v2.fit(combined_training[FEATURES], combined_training[TARGET])

    v2_predictions = v2.predict(later_for_check[FEATURES])
    v2_mae = mean_absolute_error(later_for_check[TARGET], v2_predictions)

    improvement = (v1_mae - v2_mae) / v1_mae
    decision = "promote" if improvement >= 0.15 else "keep"

    print(f"Later rows used for v2 training: {len(later_for_training)}")
    print(f"Later rows used for checking: {len(later_for_check)}")
    print(f"Total v2 training rows: {len(combined_training)}")
    print(f"V1 MAE on check: {v1_mae:.6f}")
    print(f"V2 MAE on check: {v2_mae:.6f}")
    print(f"Relative improvement: {improvement:.2%}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
