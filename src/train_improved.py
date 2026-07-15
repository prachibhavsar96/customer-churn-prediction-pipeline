from pathlib import Path

import matplotlib
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLEANED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "cleaned.csv"
ROOT_DIR = Path(__file__).resolve().parent.parent

# Verified numbers from the 5 previously logged runs, included so the
# final comparison table can show all 8 models without retraining them.
PRIOR_METRICS = {
    "LogisticRegression": {
        "accuracy": 0.8055, "precision": 0.6582, "recall": 0.5561,
        "f1": 0.6029, "roc_auc": 0.8422,
    },
    "RandomForest": {
        "accuracy": 0.7935, "precision": 0.6436, "recall": 0.4973,
        "f1": 0.5611, "roc_auc": 0.8287,
    },
    "XGBoost": {
        "accuracy": 0.7764, "precision": 0.5891, "recall": 0.5214,
        "f1": 0.5532, "roc_auc": 0.8183,
    },
    "RandomForest_Tuned": {
        "accuracy": 0.8062, "precision": 0.6747, "recall": 0.5214,
        "f1": 0.5882, "roc_auc": 0.8435,
    },
    "XGBoost_Tuned": {
        "accuracy": 0.7984, "precision": 0.6471, "recall": 0.5294,
        "f1": 0.5824, "roc_auc": 0.8356,
    },
}

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


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    return fig


def train_and_log(model, run_name, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        cm = confusion_matrix(y_test, y_pred)

        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        print("=" * 60)
        print(f"{run_name.upper()} METRICS")
        print("=" * 60)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        print(f"\n{run_name} confusion matrix:")
        print(cm)

        fig = plot_confusion_matrix(cm, f"{run_name} Confusion Matrix")
        cm_path = ROOT_DIR / f"confusion_matrix_{run_name.lower()}.png"
        fig.savefig(cm_path)
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))
        cm_path.unlink()

        model_info = mlflow.sklearn.log_model(model, name="model")
        print(f"\nModel artifact logged at: {model_info.model_uri}")

        print(f"MLflow run ID: {mlflow.active_run().info.run_id}\n")

        return metrics


def main():
    df = pd.read_csv(CLEANED_PATH)
    df = add_engineered_features(df)

    df_encoded = pd.get_dummies(df, columns=ONE_HOT_COLS, drop_first=True)

    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]

    # Same split logic as feature_engineering.py so results are comparable.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("customer_churn_prediction")

    print("Engineered feature set shape:", X_train.shape)

    results = {}

    results["LogisticRegression_Engineered"] = train_and_log(
        LogisticRegression(random_state=42, max_iter=1000),
        "LogisticRegression_Engineered",
        {"model_type": "LogisticRegression", "max_iter": 1000, "features": "engineered"},
        X_train, X_test, y_train, y_test,
    )

    results["LogisticRegression_Balanced"] = train_and_log(
        LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"),
        "LogisticRegression_Balanced",
        {"model_type": "LogisticRegression", "max_iter": 1000, "class_weight": "balanced", "features": "engineered"},
        X_train, X_test, y_train, y_test,
    )

    # SMOTE is fit on X_train/y_train only -- X_test/y_test are never
    # touched by the oversampler, so the test set stays a clean, untouched
    # measure of real-world performance.
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"\nSMOTE resampled X_train shape: {X_train_smote.shape} (churn rate: {y_train_smote.mean() * 100:.2f}%)")

    results["LogisticRegression_SMOTE"] = train_and_log(
        LogisticRegression(random_state=42, max_iter=1000),
        "LogisticRegression_SMOTE",
        {"model_type": "LogisticRegression", "max_iter": 1000, "resampling": "SMOTE", "features": "engineered"},
        X_train_smote, X_test, y_train_smote, y_test,
    )

    print("=" * 60)
    print("MODEL COMPARISON (ALL 8 MODELS, SORTED BY F1 DESC)")
    print("=" * 60)
    all_results = {**PRIOR_METRICS, **results}
    sorted_results = sorted(all_results.items(), key=lambda item: item[1]["f1"], reverse=True)

    header = f"{'Model':<28}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'ROC AUC':>10}"
    print(header)
    for name, m in sorted_results:
        print(
            f"{name:<28}{m['accuracy']:>10.4f}{m['precision']:>11.4f}"
            f"{m['recall']:>9.4f}{m['f1']:>9.4f}{m['roc_auc']:>10.4f}"
        )


if __name__ == "__main__":
    main()
