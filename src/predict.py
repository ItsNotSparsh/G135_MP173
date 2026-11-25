from pathlib import Path

import joblib
import pandas as pd

from src.train import FEATURES


HOLDOUT_PATH = Path("data/raw/holdout_features.csv")
MODEL_PATH = Path("models/model.joblib")
OUTPUT_PATH = Path("predictions/holdout_predictions.csv")


def main():
    holdout = pd.read_csv(HOLDOUT_PATH)
    model = joblib.load(MODEL_PATH)

    predictions = model.predict(holdout[FEATURES])

    output = pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": predictions,
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"Predictions written: {len(output)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()