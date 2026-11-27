import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

from src.data_checks.clean_data import clean_data
from src.train import EXPERIMENT_NAME, FEATURES, TARGET, build_model


MODEL_NAME = "MP173-spoilage-model"
TRAIN_PATH = Path("data/raw/train.csv")
LATER_PATH = Path("data/raw/later.csv")
MODEL_PATH = Path("models/model.joblib")


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    existing_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if any(version.version == "2" for version in existing_versions):
        raise RuntimeError(
            "Model version 2 already exists. Check the registry before running again."
        )

    cleaned_train = clean_data(pd.read_csv(TRAIN_PATH))
    later = (
        pd.read_csv(LATER_PATH)
        .sort_values(["date", "record_id"])
        .reset_index(drop=True)
    )

    split_at = int(len(later) * 0.70)
    later_for_training = later.iloc[:split_at]
    later_for_check = later.iloc[split_at:]

    v1 = joblib.load(MODEL_PATH)
    best_depth = v1.get_params()["regressor__max_depth"]

    combined_training = pd.concat(
        [cleaned_train, later_for_training],
        ignore_index=True,
    )

    v1_mae = float(
        mean_absolute_error(
            later_for_check[TARGET],
            v1.predict(later_for_check[FEATURES]),
        )
    )

    v2 = build_model(max_depth=best_depth)
    v2.fit(combined_training[FEATURES], combined_training[TARGET])

    v2_mae = float(
        mean_absolute_error(
            later_for_check[TARGET],
            v2.predict(later_for_check[FEATURES]),
        )
    )

    improvement = (v1_mae - v2_mae) / v1_mae
    if improvement < 0.15:
        raise RuntimeError(
            f"V2 improved MAE by {improvement:.2%}; promotion requires 15%."
        )

    data_sha256 = hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest()

    with mlflow.start_run(run_name="v2-retrain") as run:
        mlflow.log_params({
            "data_sha256": data_sha256,
            "model": "RandomForestRegressor",
            "max_depth": str(best_depth),
            "later_training_rows": len(later_for_training),
            "later_check_rows": len(later_for_check),
            "total_training_rows": len(combined_training),
        })
        mlflow.log_metric("mae_v1_on_check", v1_mae)
        mlflow.log_metric("mae_v2_on_check", v2_mae)
        mlflow.log_metric("relative_improvement", improvement)

        logged_model = mlflow.sklearn.log_model(
            sk_model=v2,
            name="spoilage_pipeline_v2",
            input_example=combined_training[FEATURES].head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        run_id = run.info.run_id

    registered = mlflow.register_model(logged_model.model_uri, MODEL_NAME)
    client.set_registered_model_alias(
        MODEL_NAME, "champion", registered.version
    )

    joblib.dump(v2, MODEL_PATH)

    print(f"V1 MAE on check: {v1_mae:.6f}")
    print(f"V2 MAE on check: {v2_mae:.6f}")
    print(f"Improvement: {improvement:.2%}")
    print(f"V2 run ID: {run_id}")
    print(f"Registered version: {registered.version}")
    print(f"Champion version: {client.get_model_version_by_alias(MODEL_NAME, 'champion').version}")
    print(f"Updated model: {MODEL_PATH}")


if __name__ == "__main__":
    main()