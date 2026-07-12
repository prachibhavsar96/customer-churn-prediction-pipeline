from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_PATH = MODELS_DIR / "churn_model.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

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
INTERNET_DEPENDENT_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def fit_scaler_on_raw_data():
    # X_train.csv already has tenure/MonthlyCharges/TotalCharges scaled --
    # feature_engineering.py applies StandardScaler before saving but never
    # persists the fitted scaler. Fitting a *new* scaler directly on those
    # already-scaled columns would fit on data with mean~0/std~1, producing
    # a broken scaler that double-transforms real incoming raw values.
    # Instead, reproduce feature_engineering.py's exact cleaning + encoding
    # + split (same random_state=42) to recover the true raw X_train numeric
    # columns, and fit the scaler on those -- this is the scaler that
    # actually matches what produced the saved CSVs.
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    for col in INTERNET_DEPENDENT_COLS:
        df[col] = df[col].replace("No internet service", "No")

    df_encoded = pd.get_dummies(df, columns=ONE_HOT_COLS, drop_first=True)
    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]

    X_train_raw, _, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    scaler.fit(X_train_raw[NUMERIC_COLS])
    return scaler


def main():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    feature_columns = list(X_train.columns)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
    print(f"Saved {len(feature_columns)} feature columns to {FEATURE_COLUMNS_PATH}")

    scaler = fit_scaler_on_raw_data()
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Scaler fitted on raw columns {NUMERIC_COLS}")
    print("Scaler mean_:", scaler.mean_)
    print("Scaler scale_:", scaler.scale_)

    print("\n" + "=" * 60)
    print("RELOAD SANITY CHECK")
    print("=" * 60)
    reloaded_model = joblib.load(MODEL_PATH)
    reloaded_columns = joblib.load(FEATURE_COLUMNS_PATH)
    reloaded_scaler = joblib.load(SCALER_PATH)

    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")

    assert list(X_test.columns) == reloaded_columns, "Column order mismatch between X_test and saved feature columns!"
    assert reloaded_scaler.mean_.shape[0] == len(NUMERIC_COLS), "Reloaded scaler shape mismatch!"

    sample_X = X_test.iloc[:5][reloaded_columns]
    sample_y = y_test.iloc[:5].reset_index(drop=True)
    predictions = reloaded_model.predict(sample_X)

    print("\nFirst 5 X_test rows -- predicted vs actual Churn:")
    print(f"{'Row':>5}{'Predicted':>12}{'Actual':>10}")
    for i in range(5):
        print(f"{i:>5}{predictions[i]:>12}{sample_y[i]:>10}")


if __name__ == "__main__":
    main()
