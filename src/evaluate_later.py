from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.train import FEATURES, TARGET


def main():
    later = pd.read_csv(Path("data/raw/later.csv"))
    model = joblib.load(Path("models/model.joblib"))

    later = later.copy()
    later["prediction"] = model.predict(later[FEATURES])

    results = {}

    print("V1 MAE by chamber:")
    for chamber, rows in later.groupby("chamber"):
        mae = float(mean_absolute_error(rows[TARGET], rows["prediction"]))
        results[chamber] = mae
        print(f"{chamber}: {mae:.6f}")

    print(f"Worst-served chamber: {max(results, key=results.get)}")
    print(f"Rows evaluated: {len(later)}")


if __name__ == "__main__":
    main()