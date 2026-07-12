from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("SHAPE")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("COLUMNS + DTYPES")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("FIRST 5 ROWS")
    print("=" * 60)
    print(df.head())

    print("\n" + "=" * 60)
    print("NULL COUNTS PER COLUMN")
    print("=" * 60)
    print(df.isnull().sum())

    print("\n" + "=" * 60)
    print("CHURN RATE (class balance baseline)")
    print("=" * 60)
    print(df["Churn"].value_counts())
    print(df["Churn"].value_counts(normalize=True) * 100)

    print("\n" + "=" * 60)
    print("TOTALCHARGES DTYPE CHECK")
    print("=" * 60)
    print("dtype:", df["TotalCharges"].dtype)
    blank_mask = df["TotalCharges"].astype(str).str.strip() == ""
    print("blank string count:", blank_mask.sum())
    print(df.loc[blank_mask, ["customerID", "tenure", "TotalCharges"]])


if __name__ == "__main__":
    main()
