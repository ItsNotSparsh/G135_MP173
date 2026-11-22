import pandas as pd


TARGET_COLUMN = "spoilage_pct"
LEAKAGE_COLUMN = "spoilage_found_at_unloading"
UNIT_MIXUP_COLUMN = "power_cuts_hours"
AFFECTED_CHAMBER = "Chamber 1"
UNIT_CONVERSION_FACTOR = 60


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the training data."""
    cleaned = df.copy()

    # Remove exact duplicate rows, keeping the first occurrence.
    cleaned = cleaned.drop_duplicates()

    # Convert Chamber 1 power-cut readings from minutes to hours.
    chamber_1 = cleaned["chamber"].eq(AFFECTED_CHAMBER)
    cleaned.loc[chamber_1, UNIT_MIXUP_COLUMN] = (
        cleaned.loc[chamber_1, UNIT_MIXUP_COLUMN] / UNIT_CONVERSION_FACTOR
    )

    # Drop the post-event leakage column from model features.
    # Keep the target column so training can use it as y.
    cleaned = cleaned.drop(columns=[LEAKAGE_COLUMN])

    return cleaned