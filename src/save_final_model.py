from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CLEANED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "cleaned.csv"
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
    "tenure_bucket",
]
SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
]
# Matches train_improved.py's LogisticRegression_Balanced exactly (5 numeric
# columns, including total_services) so the reproduced metrics line up.
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "total_services", "avg_monthly_spend"]

EXPECTED_METRICS = {
    "accuracy": 0.7360,
    "precision": 0.5017,
    "recall": 0.7941,
    "f1": 0.6149,
    "roc_auc": 0.8420,
}


def add_engineered_features(df):
    df = df.copy()

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 36, df["tenure"].max()],
        labels=["New", "Mid", "Loyal"],
    ).astype(str)

    df["total_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)

    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"]
    df.loc[df["tenure"] == 0, "avg_monthly_spend"] = df.loc[df["tenure"] == 0, "MonthlyCharges"]

    return df


def main():
    df = pd.read_csv(CLEANED_PATH)
    df = add_engineered_features(df)

    df_encoded = pd.get_dummies(df, columns=ONE_HOT_COLS, drop_first=True)

    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    feature_columns = list(X_train.columns)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
    print(f"Saved {len(feature_columns)} feature columns to {FEATURE_COLUMNS_PATH}")

    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Scaler fitted on columns {NUMERIC_COLS}")

    print("\n" + "=" * 60)
    print("TEST SET METRICS")
    print("=" * 60)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    for key, value in metrics.items():
        expected = EXPECTED_METRICS[key]
        match = "OK" if abs(value - expected) < 0.001 else "MISMATCH"
        print(f"{key}: {value:.4f}  (expected {expected:.4f}) [{match}]")

    print("\n" + "=" * 60)
    print("RELOAD SANITY CHECK")
    print("=" * 60)
    reloaded_model = joblib.load(MODEL_PATH)
    reloaded_columns = joblib.load(FEATURE_COLUMNS_PATH)
    reloaded_scaler = joblib.load(SCALER_PATH)

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
