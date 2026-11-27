from pathlib import Path

import numpy as np
import pandas as pd

from src.data_checks.clean_data import clean_data
from src.train import NUMERIC_FEATURES


def calculate_psi(reference: pd.Series, current: pd.Series) -> float:
    """Calculate PSI using ten bins based on the training data."""
    reference_values = reference.dropna().to_numpy()
    current_values = current.dropna().to_numpy()

    edges = np.quantile(reference_values, np.linspace(0, 1, 11))
    edges = np.unique(edges)
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)

    reference_share = reference_counts / reference_counts.sum()
    current_share = current_counts / current_counts.sum()

    reference_share = np.clip(reference_share, 1e-6, None)
    current_share = np.clip(current_share, 1e-6, None)

    psi = (current_share - reference_share) * np.log(
        current_share / reference_share
    )
    return float(psi.sum())


def main():
    raw_train = pd.read_csv(Path("data/raw/train.csv"))
    cleaned_train = clean_data(raw_train)
    later = pd.read_csv(Path("data/raw/later.csv"))

    results = {}

    print("PSI by feature:")
    for column in NUMERIC_FEATURES:
        psi = calculate_psi(cleaned_train[column], later[column])
        results[column] = psi
        print(f"{column}: {psi:.6f}")

    most_drifted = max(results, key=results.get)
    print(f"Most drifted feature: {most_drifted}")


if __name__ == "__main__":
    main()