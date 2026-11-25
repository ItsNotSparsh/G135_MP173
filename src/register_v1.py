import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from src.data_checks.clean_data import clean_data
from src.train import EXPERIMENT_NAME, FEATURES, TARGET, build_model


WINNING_RUN_ID = "9eb583ff4f67453bb9a2b6519ca26080"
MODEL_NAME = "MP173-spoilage-model"
MODEL_PATH = Path("models/model.joblib")


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient()
    winning_run = client.get_run(WINNING_RUN_ID)
    best_depth = int(winning_run.data.params["max_depth"])
    validation_mae = winning_run.data.metrics["mae_validation"]

    train_path = Path("data/raw/train.csv")
    data_sha256 = hashlib.sha256(train_path.read_bytes()).hexdigest()

    if data_sha256 != winning_run.data.params["data_sha256"]:
        raise ValueError("train.csv differs from the file used in the winning run.")

    cleaned_train = clean_data(pd.read_csv(train_path))
    X = cleaned_train[FEATURES]
    y = cleaned_train[TARGET]

    with mlflow.start_run(run_name="v1-refit") as run:
        model = build_model(max_depth=best_depth)
        model.fit(X, y)

        mlflow.log_params({
            **winning_run.data.params,
            "source_validation_run_id": WINNING_RUN_ID,
            "training_rows": len(X),
        })
        mlflow.log_metric("selection_mae_validation", validation_mae)

        logged_model = mlflow.sklearn.log_model(
            sk_model=model,
            name="spoilage_pipeline",
            input_example=X.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        v1_run_id = run.info.run_id

    registered = mlflow.register_model(logged_model.model_uri, MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "champion", registered.version)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Training rows: {len(X)}")
    print(f"Winning validation run ID: {WINNING_RUN_ID}")
    print(f"V1 refit run ID: {v1_run_id}")
    print(f"Registered model: {MODEL_NAME}")
    print(f"Champion version: {registered.version}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()