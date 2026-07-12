from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "cleaned.csv"

ONE_HOT_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def main():
    df = pd.read_csv(DATA_PATH)

    df = df.drop(columns=["customerID"])

    # TotalCharges is blank (not NaN) only for the 11 rows with tenure == 0,
    # i.e. customers who haven't been billed yet. Their true total charges
    # to date is 0, so filling with 0 reflects reality rather than guessing
    # a value for missing data.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    print("=" * 60)
    print("UNIQUE VALUES PER CATEGORICAL COLUMN")
    print("=" * 60)
    categorical_cols = df.select_dtypes(include="object").columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].unique())

    print("\n" + "=" * 60)
    print("SHAPE AFTER CLEANING")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("DTYPES AFTER CLEANING")
    print("=" * 60)
    print(df.dtypes)

    # "No phone service" and "No internet service" are fully determined by
    # PhoneService == "No" and InternetService == "No" respectively, so
    # keeping them as separate categories is redundant and can distort
    # feature importance. Collapse them into "No".
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    internet_dependent_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in internet_dependent_cols:
        df[col] = df[col].replace("No internet service", "No")

    print("\n" + "=" * 60)
    print("UNIQUE VALUES AFTER COLLAPSING REDUNDANT CATEGORIES")
    print("=" * 60)
    for col in ["MultipleLines"] + internet_dependent_cols:
        print(f"\n{col}:")
        print(df[col].unique())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned dataframe to {OUTPUT_PATH}")

    df_encoded = pd.get_dummies(df, columns=ONE_HOT_COLS, drop_first=True)

    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Fit the scaler on X_train only, then apply it to both splits.
    # Fitting on X_test too would leak test-set statistics into training.
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    print("\n" + "=" * 60)
    print("TRAIN/TEST SPLIT")
    print("=" * 60)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("\nChurn rate in y_train:", y_train.mean() * 100)
    print("Churn rate in y_test:", y_test.mean() * 100)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    print(f"\nSaved X_train, X_test, y_train, y_test to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
