import hashlib
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_checks.clean_data import clean_data


NUMERIC_FEATURES = [
    "storage_days",
    "temp_set_c",
    "humidity_pct",
    "power_cuts_hours",
    "bag_weight_kg",
    "door_openings_per_day",
]
CATEGORICAL_FEATURES = ["crop", "chamber"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "spoilage_pct"

EXPERIMENT_NAME = "MP173-cold-storage-spoilage"
RANDOM_STATE = 42
N_ESTIMATORS = 100
DEPTH_OPTIONS = [5, 10, None]


def build_model(max_depth):
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main():
    train_path = Path("data/raw/train.csv")
    data_sha256 = hashlib.sha256(train_path.read_bytes()).hexdigest()
    cleaned_train = clean_data(pd.read_csv(train_path))

    X = cleaned_train[FEATURES]
    y = cleaned_train[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    for depth in DEPTH_OPTIONS:
        depth_label = "unlimited" if depth is None else str(depth)

        with mlflow.start_run(run_name=f"random-forest-depth-{depth_label}") as run:
            model = build_model(max_depth=depth)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            mae = float(mean_absolute_error(y_val, predictions))

            mlflow.log_params({
                "model": "RandomForestRegressor",
                "max_depth": depth_label,
                "n_estimators": N_ESTIMATORS,
                "random_state": RANDOM_STATE,
                "data_sha256": data_sha256,
            })
            mlflow.log_metric("mae_validation", mae)

            run_id = run.info.run_id
            results.append((mae, depth_label, run_id))

            print(f"Depth {depth_label}: MAE = {mae:.3f} | Run ID: {run_id}")

    best_mae, best_depth, best_run_id = min(results, key=lambda result: result[0])
    print(f"\nBest depth: {best_depth}")
    print(f"Best validation MAE: {best_mae:.3f} percentage points")
    print(f"Best run ID: {best_run_id}")


if __name__ == "__main__":
    main()