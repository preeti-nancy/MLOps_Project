"""Clean tourism data, encode features, and create train/test splits."""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
ARTIFACT_DIR = "tourism_project/model_building/artifacts"


def clean_data(df):
    df = df.copy()
    # CustomerID is not predictive
    df = df.drop(columns=["CustomerID"], errors="ignore")
    # Fix known data quality issue
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["TypeofContact"] = df["TypeofContact"].replace({"Self Inquiry": "Self Enquiry"})
    return df


def main():
    df = pd.read_csv(DATA_PATH, index_col=0)
    print(f"Raw dataset shape: {df.shape}")

    df = clean_data(df)
    print(f"Cleaned dataset shape: {df.shape}")

    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    X_train.to_csv(f"{ARTIFACT_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{ARTIFACT_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{ARTIFACT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{ARTIFACT_DIR}/y_test.csv", index=False)
    joblib.dump(list(X_encoded.columns), f"{ARTIFACT_DIR}/feature_columns.pkl")

    print("\nData preparation complete.")
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Purchase rate (train): {y_train.mean():.2%}")
    print(f"Purchase rate (test): {y_test.mean():.2%}")


if __name__ == "__main__":
    main()
