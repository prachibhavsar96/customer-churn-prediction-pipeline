from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

CLEANED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "cleaned.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

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
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "total_services", "avg_monthly_spend"]

DEFAULT_THRESHOLD = 0.5


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


def build_test_set(feature_columns, scaler):
    # data/processed/X_test.csv is stale (23 cols, from the old
    # feature_engineering.py pipeline) -- the current model expects the
    # 27-column engineered feature set from save_final_model.py. Reproduce
    # that exact pipeline (same engineered features, same random_state=42
    # split) here, then apply the *loaded* scaler (transform only, never
    # re-fit) so X_test is scaled identically to what the model trained on.
    df = pd.read_csv(CLEANED_PATH)
    df = add_engineered_features(df)

    df_encoded = pd.get_dummies(df, columns=ONE_HOT_COLS, drop_first=True)
    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_test = X_test.reindex(columns=feature_columns, fill_value=0).copy()
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    return X_test, y_test


def compute_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    model = joblib.load(MODELS_DIR / "churn_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.joblib")

    X_test, y_test = build_test_set(feature_columns, scaler)
    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.10, 0.9001, 0.05).round(2)
    results = [
        {"threshold": t, **compute_metrics(y_test, y_proba, t)}
        for t in thresholds
    ]

    print("=" * 60)
    print("THRESHOLD SWEEP (0.10 to 0.90, step 0.05)")
    print("=" * 60)
    header = f"{'Threshold':>10}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}"
    print(header)
    for r in results:
        print(
            f"{r['threshold']:>10.2f}{r['accuracy']:>10.4f}{r['precision']:>11.4f}"
            f"{r['recall']:>9.4f}{r['f1']:>9.4f}"
        )

    best = max(results, key=lambda r: r["f1"])
    default_metrics = compute_metrics(y_test, y_proba, DEFAULT_THRESHOLD)

    print("\n" + "=" * 60)
    print("BEST THRESHOLD (by F1)")
    print("=" * 60)
    print(
        f"Best threshold: {best['threshold']}, giving "
        f"accuracy={best['accuracy']:.4f}, precision={best['precision']:.4f}, "
        f"recall={best['recall']:.4f}, f1={best['f1']:.4f}"
    )
    print(
        f"\nCurrent default threshold: {DEFAULT_THRESHOLD}, giving "
        f"accuracy={default_metrics['accuracy']:.4f}, precision={default_metrics['precision']:.4f}, "
        f"recall={default_metrics['recall']:.4f}, f1={default_metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
