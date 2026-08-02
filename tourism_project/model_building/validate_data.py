"""Validate tourism.csv schema and print a dataset summary."""
import sys
import pandas as pd

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

DATA_PATH = "tourism_project/data/tourism.csv"


def validate_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]

    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    print("Dataset validation passed.")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    if extra_cols:
        print(f"Extra columns found (will review in cleaning): {extra_cols}")

    print("\n--- Target distribution (ProdTaken) ---")
    print(df["ProdTaken"].value_counts(normalize=True).round(3))

    print("\n--- Missing values ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\n--- Numeric summary ---")
    display_cols = ["Age", "MonthlyIncome", "DurationOfPitch", "NumberOfTrips"]
    print(df[display_cols].describe().round(2))

    return df


if __name__ == "__main__":
    try:
        validate_dataset()
    except Exception as exc:
        print(f"Validation failed: {exc}")
        sys.exit(1)
